import sys
import numpy as np
import cv2

def make_sticker(img_path, out_path, cx, cy):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    r = min(h, w) // 8
    cv2.circle(mask, (cx, cy), r, cv2.GC_PRV_FGD, -1)
    cv2.circle(mask, (cx, cy), max(1, r // 4), cv2.GC_FGD, -1)
    cv2.rectangle(mask, (0, 0), (w, h), cv2.GC_BGD, 2)
    cv2.grabCut(img, mask, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
    alpha = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PRV_FGD), 255, 0).astype('uint8')
    b, g, r_ch = cv2.split(img)
    cv2.imwrite(out_path, cv2.merge([b, g, r_ch, alpha]))

def make_id_card(student_path, out_path):
    img = cv2.imread(student_path)
    if img is None:
        return
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
    cv2.grabCut(img, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    matte = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PRV_FGD), 1.0, 0.0).astype(np.float32)
    matte = cv2.GaussianBlur(matte, (5, 5), 0)
    bg = np.zeros_like(img)
    bg[:] = [255, 0, 0]
    matte_3d = np.expand_dims(matte, axis=2)
    blend = (img * matte_3d + bg * (1.0 - matte_3d)).astype(np.uint8)
    cv2.imwrite(out_path, blend)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)
    mode = sys.argv[1]
    if mode == "sticker" and len(sys.argv) == 6:
        make_sticker(sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
    elif mode == "id" and len(sys.argv) == 4:
        make_id_card(sys.argv[2], sys.argv[3])
