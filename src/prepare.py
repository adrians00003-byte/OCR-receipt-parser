# ocr_core.py
from rapidocr_onnxruntime import RapidOCR
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2

# --- OCR ENGINE ---
_ocr_engine = None

def get_ocr():
    global _ocr_engine
    if _ocr_engine is None:
        _ocr_engine = RapidOCR(
            det_db_thresh=0.25,
            det_db_box_thresh=0.45,
            det_db_unclip_ratio=1.8,
            rec_char_dict_path="latin_pl.txt",
            rec_img_h=48,
            rec_img_w=320,
            rec_score_thresh=0.5,
            use_angle_cls=False
        )
    return _ocr_engine

# --- PRZYGOTOWANIE OBRAZU ---
def prepare_image(pil_img: Image.Image):
    img = ImageOps.grayscale(pil_img)
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Sharpness(img).enhance(1.6)

    w, h = img.size
    if w < 1400:
        img = img.resize((int(w * 1.6), int(h * 1.6)), Image.BICUBIC)

    rgb = np.asarray(img.convert("RGB"))
    return rgb

# --- PROSTOWANIE PARAGONU ---
def detect_receipt_corners(pil_input):
    if isinstance(pil_input, str):
        img = cv2.imread(pil_input)
    else:
        img = cv2.cvtColor(np.array(pil_input), cv2.COLOR_RGB2BGR)

    orig = img.copy()
    h, w = img.shape[:2]

    scale = 800 / max(h, w)
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (15, 15), 0)

    th = cv2.adaptiveThreshold(gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 9)

    kernel = np.ones((25, 25), np.uint8)
    marker = cv2.dilate(th, kernel, iterations=2)

    contours, _ = cv2.findContours(marker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None

    biggest = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(biggest)
    rect = cv2.minAreaRect(hull)
    box = cv2.boxPoints(rect)

    # sort punktów
    box = np.array(box, dtype=np.float32)
    s = box.sum(axis=1)
    diff = box[:, 1] - box[:, 0]
    pts = np.array([
        box[np.argmin(s)],   # tl
        box[np.argmin(diff)],# tr
        box[np.argmax(s)],   # br
        box[np.argmax(diff)],# bl
    ], dtype=np.float32)

    if scale < 1:
        pts /= scale

    pts = pts.astype(np.float32)

    # warp
    wA = np.linalg.norm(pts[1] - pts[0])
    wB = np.linalg.norm(pts[2] - pts[3])
    hA = np.linalg.norm(pts[3] - pts[0])
    hB = np.linalg.norm(pts[2] - pts[1])

    maxW = int(max(wA, wB))
    maxH = int(max(hA, hB))

    dst = np.array([[0,0],[maxW-1,0],[maxW-1,maxH-1],[0,maxH-1]], dtype=np.float32)

    M = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(orig, M, (maxW, maxH))

    pil = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    return pts.astype(int), pil


# --- GŁÓWNE OCR ---
def run_ocr(pil_img: Image.Image):
    rgb = prepare_image(pil_img)
    ocr = get_ocr()
    res, _ = ocr(rgb)
    return res
