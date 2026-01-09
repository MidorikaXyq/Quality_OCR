import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


counter = 0
def save_image(img_array):
    global counter
    output_path = "C:/Users/ROG/Downloads/result"
    output_path = os.path.join(output_path, f"{counter}.png")
    success = cv2.imwrite(output_path, img_array)
    if not success:
        raise IOError(f"Failed to write image to {output_path}")
    counter += 1
    print(f"Saved image → {output_path}")

def binary_contrast(img, threshold):
    _, th_inv = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return th_inv

def open_stroke(img, effect, iteration):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (effect, effect))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iteration)

def close_stroke(img, effect, iteration):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (effect, effect))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iteration)

def preprocess(img):
    pts1 = np.float32([[0, 204], [1682+332, 5], [110, 643], [2089, 513]])
    pts2 = np.float32([[0, 0], [1500, 0], [0, 400], [1500, 400]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (1500, 400))

    # sigma = 300
    # img = cv2.bilateralFilter(img,5, sigma, sigma)

    save_image(img)
    return img

img = cv2.imread("C:/Users/ROG/Downloads/raw_in/alldata.png")
img = preprocess(img)

# 3) Show the image and let the user draw the rectangle
#    You MUST click-and-drag a box, then press ENTER (or SPACE) to confirm.
cv2.imshow("graph", img)
cv2.waitKey(0)

cv2.destroyAllWindows()
