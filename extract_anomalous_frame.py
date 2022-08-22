from copy import deepcopy
import os

import numpy as np

from tbad.autoencoder.data import extract_global_features, change_coordinate_system
from tbad.autoencoder.data import scale_trajectories
from tbad.rnn_autoencoder.data import remove_short_trajectories, aggregate_rnn_ae_evaluation_data
from tbad.rnn_autoencoder.data import compute_rnn_ae_reconstruction_errors, summarise_reconstruction_errors
from tbad.combined_model.fusion import load_pretrained_combined_model


# Compute Reconstruction Error
def get_reconstruction_errors(pretrained_model_path, trajectories):
    """
    Computed reconstruction errors

    Parameters
    ----------
    pretrained_model_path                 : str
                                            path to pretrained model
    trajectories                          : dict
                                            dict of video_id as a key and Trajectory() object as a value

    Returns
    ----------
    recontruction_error                   : numpy.ndarray
                                           array of recontruction error value for each frame
    """
    # Model Intialization
    model_info = os.path.basename(os.path.split(pretrained_model_path)[0])
    message_passing = 'mp' in model_info
    pretrained_combined_model, global_scaler, local_scaler, out_scaler = load_pretrained_combined_model(
        pretrained_model_path, message_passing)
    multiple_outputs = pretrained_combined_model.multiple_outputs
    input_length, rec_length = pretrained_combined_model.input_length, pretrained_combined_model.reconstruction_length
    input_gap, pred_length = 0, pretrained_combined_model.prediction_length
    reconstruct_reverse = pretrained_combined_model.reconstruct_reverse
    loss = pretrained_combined_model.loss
    print("Model Summary")
    print("-------------------")
    print(f"multiple_outputs : {multiple_outputs}")
    print(f"input_length, rec_length : {input_length, rec_length}")
    print(f"input_gap, pred_lenght : {input_gap, pred_length}")
    print(f"reconstructe_reverse : {reconstruct_reverse}")
    print(f"loss : {loss}")

    # Extract information about the models
    reconstruct_original_data = 'down' in model_info
    global_normalisation_strategy = 'zero_one'
    if '_G3stds_' in model_info:
        global_normalisation_strategy = 'three_stds'
    elif '_Grobust_' in model_info:
        global_normalisation_strategy = 'robust'

    local_normalisation_strategy = 'zero_one'
    if '_L3stds_' in model_info:
        local_normalisation_strategy = 'three_stds'
    elif '_Lrobust_' in model_info:
        local_normalisation_strategy = 'robust'

    out_normalisation_strategy = 'zero_one'
    if '_O3stds_' in model_info:
        out_normalisation_strategy = 'three_stds'
    elif '_Orobust_' in model_info:
        out_normalisation_strategy = 'robust'

    # Data
    trajectories = remove_short_trajectories(trajectories, input_length=input_length,
                                             input_gap=input_gap, pred_length=pred_length)

    # Extract global and local feature from the data

    video_resolution = 1920, 1080
    overlapping_trajectories = True
    global_trajectories = extract_global_features(deepcopy(trajectories), video_resolution=video_resolution)
    global_trajectories = change_coordinate_system(global_trajectories, video_resolution=video_resolution,
                                                   coordinate_system='global', invert=False)

    trajectories_ids, frames, X_global = \
        aggregate_rnn_ae_evaluation_data(global_trajectories,
                                         input_length=input_length,
                                         input_gap=input_gap,
                                         pred_length=pred_length,
                                         overlapping_trajectories=overlapping_trajectories)
    X_global, _ = scale_trajectories(X_global, scaler=global_scaler, strategy=global_normalisation_strategy)

    local_trajectories = deepcopy(trajectories)
    local_trajectories = change_coordinate_system(local_trajectories, video_resolution=video_resolution,
                                                  coordinate_system='bounding_box_centre', invert=False)
    _, _, X_local = aggregate_rnn_ae_evaluation_data(local_trajectories, input_length=input_length,
                                                     input_gap=input_gap, pred_length=pred_length,
                                                     overlapping_trajectories=overlapping_trajectories)
    X_local, _ = scale_trajectories(X_local, scaler=local_scaler, strategy=local_normalisation_strategy)

    original_trajectories = deepcopy(trajectories)
    _, _, X_original = aggregate_rnn_ae_evaluation_data(original_trajectories, input_length=input_length,
                                                        input_gap=input_gap, pred_length=pred_length,
                                                        overlapping_trajectories=overlapping_trajectories)

    if reconstruct_original_data:
        out_trajectories = trajectories
        out_trajectories = change_coordinate_system(out_trajectories, video_resolution=video_resolution,
                                                    coordinate_system='global', invert=False)
        _, _, X_out = aggregate_rnn_ae_evaluation_data(out_trajectories, input_length=input_length,
                                                       input_gap=input_gap, pred_length=pred_length,
                                                       overlapping_trajectories=True)
        X_out, _ = scale_trajectories(X_out, scaler=out_scaler, strategy=out_normalisation_strategy)

    X_input = [X_global, X_local]
    if pred_length == 0:
        if multiple_outputs:
            _, _, reconstructed_X = pretrained_combined_model.predict(X_input, batch_size=1024)
        else:
            reconstructed_X = pretrained_combined_model.predict(X_input, batch_size=1024)
    else:
        if multiple_outputs:
            _, _, reconstructed_X, _, _, predicted_y = \
                pretrained_combined_model.predict(X_input, batch_size=1024)
        else:
            reconstructed_X, predicted_y = pretrained_combined_model.predict(X_input, batch_size=1024)

    if reconstruct_reverse:
        reconstructed_X = reconstructed_X[:, ::-1, :]

    X = X_out if reconstruct_original_data else np.concatenate((X_global, X_local), axis=-1)
    reconstruction_errors = compute_rnn_ae_reconstruction_errors(X[:, :rec_length, :], reconstructed_X, loss)
    reconstruction_ids, reconstruction_frames, reconstruction_errors = \
        summarise_reconstruction_errors(reconstruction_errors, frames[:, :rec_length], trajectories_ids[:, :rec_length])

    return reconstruction_errors


# Detect Anomalous
def detect_most_anomalous_or_most_normal_frames(reconstruction_errors, anomalous=True, fraction=0.20):
    """
    Tell which frame is normal or anomalous

    Parameters
    ----------
    reconstruction_errors                   : numpy.ndarray
                                              array of recontruction error for each frame
    anomalous                               : bool
                                              True, for returning True for anomalous frame indices and vice versa

    fraction                                : float
                                              fraction of frame to blame on


    Returns
    ----------
    anomalous_or_normal_frame               : numpy.ndarray
                                              array contain True or False value to tell which is normal or anomalous

    """

    reconstruction_errors_sorted = np.sort(reconstruction_errors)
    num_frames_to_blame = round(len(reconstruction_errors_sorted) * fraction)
    if anomalous:
        threshold = np.round(reconstruction_errors_sorted[-num_frames_to_blame], decimals=8)
        anomalous_or_normal_frames = reconstruction_errors >= threshold
    else:
        threshold = np.round(reconstruction_errors_sorted[num_frames_to_blame - 1], decimals=8)
        anomalous_or_normal_frames = (0 < reconstruction_errors) & (reconstruction_errors <= threshold)

    return anomalous_or_normal_frames


# Inference Pipeline
def extract_anomalous_frame(pretrained_model_path, trajectories):
    """
    Tell which frame is normal or anomalous

    Parameters
    ----------
    pretrained_model_path                           : str
                                                      path to pretrained model
    trajectories                                    : dict
                                                      dict of video_id as a key and Trajectory() object as a value

    Returns
    ----------
    np.where(anomalous_frames == True)               : numpy.ndarray
                                                       array contain which frame number is anomalous

    """
    # Perfrom reconstruction error
    reconstruction_errors = get_reconstruction_errors(pretrained_model_path, trajectories)

    # Find the anomalous in the frame
    anomalous_frames = detect_most_anomalous_or_most_normal_frames(reconstruction_errors,
                                                                   anomalous=True,
                                                                   fraction=0.20)

    count = 0
    for pred in anomalous_frames:
        if pred == True:
            count += 1

    print(f"Total Anomalous Frames : {count}")

    return np.where(anomalous_frames == True)







