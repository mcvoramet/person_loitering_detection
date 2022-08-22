import cv2
import os
from natsort import natsorted

image_size = (1920, 1080)
fps = 24
out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, image_size)
img_dir = "detection_results"
for img_file in natsorted(os.listdir(img_dir)):

    img = cv2.imread(img_dir + "/" + img_file)
    cv2.imshow("Loitering Detection", img)
    out.write(img)
    cv2.waitKey(1)

out.release()

