import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os
import pickle
from tbad.autoencoder.data import Trajectory
from natsort import natsorted


def get_trajectories_object(video_id_dict):
    """
    Convert the dict format of trajectory data to be in Trajectory() object format

    Parameters
    ----------
    video_id_dict                 : dict
                                    dict of frame number as a key and skeleton coordinates as a value

    Returns
    ----------
    trajectories                  : Trajectory() object
                                    data in Trajectory() object format


    """
    trajectories = {}
    frame_num_list = []
    video_frame_list = []
    for video_id in video_id_dict.keys():

        for frame_num in video_id_dict[video_id]:
            trajectory_id = video_id + "_" + str(frame_num)
            frame_num_list.append(frame_num)
            video_frame_list.append(flatten(video_id_dict[video_id][frame_num]))

    frame_num_array = np.array(frame_num_list).astype(np.int32)
    video_frame_array = np.array(video_frame_list).astype(np.float32)
    trajectories[video_id] = Trajectory(video_id + "_" + "0", frame_num_array, video_frame_array)

    return trajectories


def flatten(l):
    """Flatten the list from [[x1, y1], [x2. y2], ...] to [x1, y1, x2, y2, ...]"""

    return [item for sublist in l for item in sublist]


def convert_normalized_frame(frame_size, keypoints, confidence_threshold, ignore_confidence):
    """
    Convert the normalized size of frame to match the original shape of an image

    Parameters
    ----------
    frame_size                 : numpy.ndarray
                                 input frame image                  :
    keypoints                  : list
                                 list of keypoints [normalized(y), normalized(x), confidence_scores]
    confidence_threshold       : float
                                 value of confidence threshold (0.0-1.0)
    ignore_confidence          : bool
                               : True for not using the confidence threshold for returning skeleton points

    Returns
    ----------
    fitting_size_list           : list
                                  list of new coordinate after conversion with it frame number
                                  [[x1, y1], [x2, y2], ....]
    """

    y, x, c = frame_size.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    fitting_size_list = []

    if ignore_confidence:
        for kp in shaped:
            ky, kx, kp_conf = kp
            fitting_size_list.append([kx, ky])

    else:
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                fitting_size_list.append([kx, ky])

            if len(fitting_size_list) < 17:
                add_more = 17 - len(fitting_size_list)
                for _ in range(add_more):
                    fitting_size_list.append([0, 0])

    return fitting_size_list


def prep_input(dataset_dir):
    """
    Convert the normalized size of frame to match the original shape of an image

    Parameters
    ----------
    dataset_dir                 : str
                                : dataset path contained frame from the video that will be using for skeleton detection

    Returns
    ----------
    trajectories                : dict
                                  dict of video_id as a key and Trajectory() object as a value
    """

    model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    movenet = model.signatures['serving_default']

    abort_test = 0
    frame_num = 0

    fitting_size_dict = {}
    video_id_dict = {}
    video_list = os.listdir(dataset_dir)
    video_list = natsorted(video_list)
    for video_id in video_list:

        if "." not in video_id:

            video_ls = natsorted(os.listdir(dataset_dir + "/" + video_id))

            for i in range(0, len(video_ls) - 1):

                frame = video_ls[i]

                org_img = cv2.imread(dataset_dir + "/" + video_id + "/" + frame)

                # Resize image
                img = tf.image.resize_with_pad(tf.expand_dims(org_img, axis=0), 288, 512)
                input_img = tf.cast(img, dtype=tf.int32)

                # Perform detection
                results = movenet(input_img)
                # in each keypoint will have these 3 values
                # [norm(y) corrdinate, norm(x) coordinate, confidence_scores]
                keypoints_with_scores = results["output_0"].numpy()[:, :, :51].reshape((6, 17, 3))

                for keypoints in keypoints_with_scores:
                    fitting_size_dict[frame_num] = convert_normalized_frame(org_img, keypoints, 0.1, True)

                frame_num += 1

            video_id_dict[video_id] = fitting_size_dict

    trajectories = get_trajectories_object(video_id_dict)
    test_frame = "000216"
    print(trajectories)
    print(len(trajectories))
    print(trajectories[test_frame].trajectory_id)
    print(trajectories[test_frame].frames)
    print(len(trajectories[test_frame].frames))
    print(trajectories[test_frame].coordinates)
    print(len(trajectories[test_frame].coordinates))

    return trajectories


# if __name__ == "__main__":
#     main()

trajectories = prep_input("./Corridor_Datasets")
with open('trajectories.pkl', 'wb') as f:
    pickle.dump(trajectories, f)

