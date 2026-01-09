import cv2

path = r"C:\Users\ROG\Downloads\result\0.png"
img  = cv2.imread(path)
if img is None:
    raise FileNotFoundError(path)

# 2) Create a normal window that you can resize
win = "Select Column ROI"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
# Resize it to something big enough to show your entire screenshot
cv2.resizeWindow(win, 1500, 400)  # tweak these numbers to fit your screen

# 3) Show the image and let the user draw the rectangle
#    You MUST click-and-drag a box, then press ENTER (or SPACE) to confirm.
roi = cv2.selectROI(win, img, showCrosshair=True, fromCenter=False)

# 4) Clean up
cv2.destroyAllWindows()

# 5) roi is (x, y, w, h)
print("ROI coords (x, y, w, h):", roi)


# 282   354   +62