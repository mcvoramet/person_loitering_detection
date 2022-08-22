from extract_anomalous_frame import extract_anomalous_frame
import pickle
import cv2
import os
import numpy as np
from natsort import natsorted


def inference_pipeline():
    # Model path
    pretrained_model_path = "./pretrained/CVPR19/ShanghaiTech/combined_model/_mp_Grobust_Lrobust_Orobust_concatdown_/01_2018_11_09_10_55_13"

    # Data
    with open('trajectories.pkl', 'rb') as f:
        trajectories = pickle.load(f)

    anomalous_indices = extract_anomalous_frame(pretrained_model_path, trajectories)
    print(anomalous_indices)

    org_img_path = "./Corridor_Datasets/000216"
    org_frame_list = natsorted(os.listdir(org_img_path))

    save_dir_path = "detected_results"
    isExist = os.path.exists(save_dir_path)
    if not isExist:
      os.mkdir(save_dir_path)

    for idx in range(0, len(org_frame_list) - 1):

        if idx in anomalous_indices[0]:

            kp = trajectories["000216"].coordinates[idx].reshape((17, 2))
            img = cv2.imread(org_img_path + "/" + org_frame_list[idx])

            y, x, c = img.shape
            shaped = np.squeeze(np.multiply(kp, [x, y]))

            for kp in shaped:
                kx, ky = kp
                cv2.circle(img, (int(kx), int(ky)), 6, (0, 0, 255), -1)

            cv2.imwrite(f"{save_dir_path}/{idx}.jpg", img)
            print(f"saved frame {idx} successfully")

        else:

            img = cv2.imread(org_img_path + "/" + org_frame_list[idx])
            cv2.imwrite(f"{save_dir_path}/{idx}.jpg", img)
            print(f"saved frame {idx} successfully")


if __name__ == "__main__":
    inference_pipeline()









