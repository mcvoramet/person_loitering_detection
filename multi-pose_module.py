import tensorflow as tf
import tensorflow_hub as hub
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import time
import pandas as pd

# Joint connection map of each keypoints that will connect to each other
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}


def draw_keypoints(frame, keypoints, confidence_threshold):
    """
    Draw the keypoints on an image

    Parameters
    ----------
    frame                 : numpy.ndarray
                            input frame image
    keypoints             : list
                            list of keypoints [norm(y) corrdinate, norm(x) coordinate, confidence_scores]
    confidence_threshold  : float
                            value of confidence threshold (0.0-1.0)
    """

    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))
    print(shaped)
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 6, (0, 255, 0), -1)
            print(f"x : {kx}")
            print(f"y : {ky}")



def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 4)


# Function to loop through each person detected and render
def loop_through_people(frame, keypoints_with_scores, edges, confidence_threshold):

    for person in keypoints_with_scores:
        #draw_connections(frame, person, edges, confidence_threshold)
        draw_keypoints(frame, person, confidence_threshold)



def main():

    test_train_frame_path = "../Corridor_Datasets/Test/Loitering_Test_Frames/000216"
    p_time = 0
    model = hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
    movenet = model.signatures['serving_default']

    abort_test = 0

    for frame in os.listdir(test_train_frame_path):

        abort_test += 1
        if abort_test > 60:
            break

        org_img = cv2.imread(test_train_frame_path + "/" + frame)

        # Resize image
        img = tf.image.resize_with_pad(tf.expand_dims(org_img, axis=0), 288, 512)
        input_img = tf.cast(img, dtype=tf.int32)

        # Perform detection
        results = movenet(input_img)
        # in each keypoint will have these 3 values
        # [norm(y) corrdinate, norm(x) coordinate, confidence_scores]
        keypoints_with_scores = results["output_0"].numpy()[:, :, :51].reshape((6, 17, 3))

        # Render keypoints
        loop_through_people(org_img, keypoints_with_scores, EDGES, 0.1)

        # check frame rate
        c_time = time.time()  # current time
        fps = 1/(c_time-p_time)
        p_time = c_time  # previous time

        cv2.putText(org_img, "FPS : " + str(int(fps)), (70, 50), cv2.FONT_HERSHEY_PLAIN, 3,
                    (0, 255, 0), 3)

        cv2.imshow("Image", org_img)

        # 1 ms delay
        cv2.waitKey(1)


if __name__ == "__main__":
    main()

