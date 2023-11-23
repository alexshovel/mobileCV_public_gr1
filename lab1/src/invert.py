# ITMO University
# Mobile Computer Vision course
# 2020
# by Aleksei Denisov
# denisov@itmo.ru

import cv2
import numpy as np

# gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
# Defaults to 1280x720 @ 60fps
# Flip the image by setting the flip_method (most common values: 0 and 2)
# display_width and display_height determine the size of the window on the screen


def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=1280,
    display_height=720,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink max-buffers=1 drop=true"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def my_adaptive_thresh_mean(img, region, C=5):
    max_r = img.shape[0]
    max_c = img.shape[1]
    half_region = (region - 1) / 2
    res_img = []
    for r in range(max_r):
        new_line = []
        start_r = int(0 if (r - half_region) < 0 else r)
        end_r = int(r + half_region if (r + half_region) < max_r else max_r)
        for c in range(max_c):
            start_c = int(0 if (c - half_region) < 0 else c - half_region)
            end_c = int(c + half_region if (r + half_region) < max_r else max_c)
            region = img[start_r:end_r, start_c:end_c]
            treshold = region.mean() + C
            adaptive = 0 if img[r, c] < treshold else 255
            new_line.append(np.uint8(adaptive))

        res_img.append(new_line)
    return np.array(res_img)


def show_camera():
    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    # print(gstreamer_pipeline(flip_method=4))
    cap = cv2.VideoCapture(0) # gstreamer_pipeline(flip_method=4), cv2.CAP_GSTREAMER)

    if cap.isOpened():
        window_handle = cv2.namedWindow("CSI Camera", cv2.WINDOW_AUTOSIZE)
        # Window
        while cv2.getWindowProperty("CSI Camera", 0) >= 0:
            ret_val, frame = cap.read()

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('Original', img)
            thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
            cv2.imshow('Adaptiv CV2 199', thresh1)

            # Show video
            my_thresh1 = my_adaptive_thresh_mean(img, 19, 5)
            cv2.imshow('Adaptiv lab 199', my_thresh1)

            # This also acts as
            keyCode = cv2.waitKey(1) & 0xFF
            # Stop the program on the ESC key
            if keyCode == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        print("Unable to open camera")


if __name__ == "__main__":
    show_camera()
