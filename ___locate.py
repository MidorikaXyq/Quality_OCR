import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

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
    _, th_inv = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    return cv2.bitwise_not(th_inv)

def detect_cells(img, scale=15):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # binary
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract horizontal lines
    hor_size = bw.shape[1] // scale
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hor_size, 1))
    hor_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hor_kernel)

    # Extract vertical lines
    ver_size = bw.shape[0] // scale
    ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, ver_size))
    ver_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ver_kernel)

    # Combine to get grid intersections
    grid = cv2.bitwise_and(hor_lines, ver_lines)

    # Find intersection points
    pts = cv2.findNonZero(grid)
    # You can then cluster these points to find the intersections of rows & cols
    # or use connectedComponents on the inverted grid to get cell boxes.
    return hor_lines, ver_lines, grid


def find_table_roi(img, hor_lines, ver_lines):
    # 1) union of lines → full grid
    grid_mask = cv2.bitwise_or(hor_lines, ver_lines)

    # 2) connected components on that mask
    num_lbl, labels, stats, _ = cv2.connectedComponentsWithStats(
        grid_mask, connectivity=8
    )
    # stats[i] = [x, y, w, h, area]

    # 3) ignore background (label 0) and pick the component with max area
    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        raise RuntimeError("No table lines found")
    table_lbl = 1 + int(np.argmax(areas))
    x, y, w, h, _ = stats[table_lbl]

    # 4) return both the ROI and its top-left corner
    return img[y : y + h, x : x + w], (x, y, w, h)


def visualize_line_masks(img, hor_mask, ver_mask, combined_overlay=False):
    """
    Displays the original image, the horizontal‐line mask, the vertical‐line mask,
    and (optionally) an overlay of both on the original.

    Args:
      img            : BGR image as read by cv2.imread()
      hor_mask       : single‐channel binary mask of horizontal lines
      ver_mask       : single‐channel binary mask of vertical   lines
      combined_overlay: if True, shows both masks overlaid on img
    """
    # Convert BGR→RGB for plotting
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Prepare overlay if requested
    if combined_overlay:
        overlay = img_rgb.copy()
        # horizontal in red
        overlay[hor_mask > 0] = [255, 0, 0]
        # vertical   in blue
        overlay[ver_mask > 0] = [0, 0, 255]

    # Plot
    ncols = 3 + int(combined_overlay)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis('off')

    axes[1].imshow(hor_mask, cmap='gray')
    axes[1].set_title("Horizontal Mask")
    axes[1].axis('off')

    axes[2].imshow(ver_mask, cmap='gray')
    axes[2].set_title("Vertical Mask")
    axes[2].axis('off')

    if combined_overlay:
        axes[3].imshow(overlay)
        axes[3].set_title("Overlay (H=red, V=blue)")
        axes[3].axis('off')

    save_image(overlay)
    plt.tight_layout()
    plt.show()


img_path = "C:/Users/ROG/Downloads/peek.png"
img = cv2.imread(img_path)
img = binary_contrast(img, 85)
save_image(img)
hor, ver, grid = detect_cells(img)

visualize_line_masks(img, hor, ver, combined_overlay=True)

table_img, (tx, ty, tw, th) = find_table_roi(img, hor, ver)