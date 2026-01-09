import csv
import math
import os
import cv2
import numpy as np
from paddleocr import PaddleOCR

WIDTH = 1500
HEIGHT = 400

def binary_contrast(img, threshold):
    _, th_inv = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_not(th_inv)

def open_stroke(img, effectx, effecty, iteration):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (effectx, effecty))
    return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iteration)

def close_stroke(img, effectx, effecty, iteration):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (effectx, effecty))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iteration)

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


IN_DIR = r"C:/Users/ROG/Downloads/raw_in"
OUT_DIR = r"C:/Users/ROG/Downloads/raw_out"

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    lang="en",
    ocr_version="PP-OCRv5",
    device="gpu:0",
)

def preprocess(img):
    pts1 = np.float32([[0, 204], [1682+332, 5], [110, 643], [2089, 513]])
    pts2 = np.float32([[0, 0], [WIDTH, 0], [0, HEIGHT], [WIDTH, HEIGHT]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    img = cv2.warpPerspective(img, M, (WIDTH, HEIGHT))
    
    # save_image(img)
    return img



def postprocess(content):
    layers = [[] for _ in range(12)]
    cell_height = HEIGHT / 13.0

    for text, confidence, polys in content:
        y = polys[0][1]
        row = min(range(13), key=lambda i: abs(y - i * cell_height))
        x = polys[0][0]
        if row == 0:
            continue
        layers[row-1].append((x, text))

    for layer in layers:
        layer.sort(key=lambda item: item[0])

    csv_path = os.path.join(OUT_DIR, f"result.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Nome Zona Secondo Cliente", "", ".%", "S/N", "Delta TC (C)", "tempo risposta TC (s)", "Rigidita (V)", "Risultati"])
        for layer in layers:
            texts = [text for _, text in layer]

            first_seven = texts[:7]
            rest = texts[7:]
            merged = " ".join(rest)

            row_cells = first_seven + [merged]

            if len(row_cells) < 8:
                row_cells += [""] * (8 - len(row_cells))

            writer.writerow(row_cells)

    print(f"Wrote to {csv_path}.")





def my_ocr():
    fname = "alldata.png"
    in_path = os.path.join(IN_DIR, fname)
    base_name = os.path.splitext(fname)[0]

    img = cv2.imread(in_path)
    img = preprocess(img)
    result = ocr.predict(img)

    csv_content = []
    for res in result:
        res.print()
        res.save_to_img(save_path=os.path.join(OUT_DIR, f"{base_name}"))

        txt = res["rec_texts"]
        score = res["rec_scores"]
        polys = res["rec_polys"]
        for i, item in enumerate(txt):
            csv_content.append((txt[i], score[i], polys[i]))
        # csv_path = os.path.join(OUT_DIR, f"{base_name}.csv")
        # with open(csv_path, "w", newline="", encoding="utf-8") as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["rec_text", "rec_score"])
        #     writer.writerows(csv_content)
        # print(f"✅ Done {fname}: {len(res['rec_texts'])} entries → {csv_path}")
    return csv_content

postprocess(my_ocr())
















# for fname in sorted(os.listdir(IN_DIR)):
#     if not fname.startswith('all'):
#         continue
#     in_path = os.path.join(IN_DIR, fname)
#     base_name = os.path.splitext(fname)[0]
#
#     img = cv2.imread(in_path)
#     img = preprocess(img)
#     result = ocr.predict(img)
#
#     csv_content = []
#     for res in result:
#         res.print()
#         res.save_to_img(save_path=os.path.join(OUT_DIR, f"{base_name}"))
#
#         txt = res["rec_texts"]
#         score = res["rec_scores"]
#         for i, item in enumerate(txt):
#             csv_content.append((txt[i], score[i]))
#         csv_path = os.path.join(OUT_DIR, f"{base_name}.csv")
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             writer = csv.writer(f)
#             writer.writerow(["rec_text", "rec_score"])
#             writer.writerows(csv_content)
#
#         print(f"✅ Done {fname}: {len(res['rec_texts'])} entries → {csv_path}")
#
#
