import csv
import cv2
import numpy as np
from paddleocr import TextDetection, TextRecognition
import os

IN_DIR        = r"C:/Users/ROG/Downloads/train"
DETECT_OUT_DIR   = r"C:/Users/ROG/Downloads/detect"
RECOG_OUT_DIR    = r"C:/Users/ROG/Downloads/recog"

detect_model     = TextDetection(model_name="PP-OCRv5_server_det", device="gpu:0")
recog_model      = TextRecognition(model_name="PP-OCRv5_server_rec",  device="gpu:0")



def crop_poly(img, poly):
    x, y, w, h = cv2.boundingRect(poly.astype(np.int32))
    return img[y:y+h, x:x+w]

def preprocess(img):
    return open_stroke(img, 2, 1)

for fname in sorted(os.listdir(IN_DIR)):
    if not fname.lower().endswith(".png"):
        continue

    in_path  = os.path.join(IN_DIR, fname)
    base_name = os.path.splitext(fname)[0]

    img = cv2.imread(in_path)

    img = preprocess(img)

    det_results = detect_model.predict(img)

    for i, res in enumerate(det_results):
        res.save_to_img(save_path=os.path.join(DETECT_OUT_DIR, f"{base_name}_{i}.png"))
        res.save_to_json(save_path=os.path.join(DETECT_OUT_DIR, f"{base_name}_{i}.json"))

    recogs = []
    for res in det_results:
        polys = res["dt_polys"]
        for poly in polys:
            crop = crop_poly(img, poly)
            recog = recog_model.predict(crop)
            for r in recog:
                txt = r["rec_text"]
                score = r["rec_score"]
                recogs.append((txt, score))

    csv_path = os.path.join(RECOG_OUT_DIR, f"{base_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rec_text", "rec_score"])
        writer.writerows(reversed(recogs))

    print(f"✅ Done {fname}: {len(recogs)} entries → {csv_path}")