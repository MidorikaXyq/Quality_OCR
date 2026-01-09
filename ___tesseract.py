import os
import cv2
from paddleocr import PaddleOCR

counter = 0
output_file = "C:/Users/ROG/Downloads/result/output.csv"
roi = (539, 1478, 4079, 927)

ocr_engine = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

def save_image(img_array):
    global counter
    output_path = "C:/Users/ROG/Downloads/result"
    output_path = os.path.join(output_path, f"{counter}.png")
    success = cv2.imwrite(output_path, img_array)
    if not success:
        raise IOError(f"Failed to write image to {output_path}")
    counter += 1
    print(f"Saved image → {output_path}")

def crop_image(img, roi):
    x, y, w, h = roi
    return img[y:y+h, x:x+w]

def binary_contrast(img, threshold):
    _, th_inv = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_not(th_inv)

def preprocess(image_path):
    th = cv2.imread(image_path)
    if th is None:
        raise FileNotFoundError(f"Cannot load image at {image_path}")

    # th = binary_contrast(th, 81)
    # th = crop_image(th, roi)

    return th

def ocr(img):
    result = ocr_engine.predict(img)
    for res in result:
        res.print()
        res.save_to_img("C:/Users/ROG/Downloads/result")
        res.save_to_json("C:/Users/ROG/Downloads/result/result.json")

if __name__ == "__main__":
    fname = "0.png"
    path = os.path.join("C:/Users/ROG/Downloads/", fname)

    prep = preprocess(path)

    ocr(prep)