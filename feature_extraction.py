'''
feature_extraction.py

Core pipeline for the rule-based Facial Emotion Recognition (FER) system, based on the paper:

    Aslam, A. & Hussain, B. (2021). "Emotion recognition techniques with
    rule based and machine learning approaches." arXiv:2103.00658

Pipeline overview (per image):
    1. Face detection (Viola-Jones / Haar Cascade) + elliptical face mask
    2. Standardized resize to 281x381 
    3. Facial landmark detection (MediaPipe FaceMesh, 468 points): used to
       locate the eyebrow, forehead and mouth regions precisely, instead of
       the fixed percentage-based ROIs used in earlier iterations of this
       project (which broke on faces with different proportions/hairlines).
    4. Five numeric features extracted: Mouth Opening (MO), Eyebrow
       Constriction (EBC), Eyebrow Mean height (EBM), Lip Corner distance
       (LC), forehead Wrinkle pixel count (W).
    5. Rule-based classification into one of 5 emotions (Angry, Disgust,
       Happy, Neutral, Surprise) via Majority Voting over threshold rules,
       following the paper's HES (Human Emotion Sensitivity) table.

Note on the Eye Map / Mouth Map formulas: the paper's original formulas are
chrominance-based (YCbCr color space). Grayscale input images (e.g. JAFFE)
carry no real chrominance information, so this pipeline detects grayscale
input automatically and falls back to a luminance-based approximation for
those cases (see the `is_grayscale` branch below).
'''

import mediapipe as mp
import cv2
import numpy as np
from scipy.signal import find_peaks

mp_face_mesh = mp.solutions.face_mesh

''' 
MediaPipe FaceMesh landmark indices (468-point model) used to locate the eyebrows,
forehead and mouth precisely on each individual face,
instead of guessing fixed percentages of the image height/width.
'''
LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
FOREHEAD_TOP = 10    # top-of-forehead reference point
MOUTH_UPPER = 0
MOUTH_LOWER = 17
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

'''
Fallback mouth ROI bounds (381x281 standardized image), used only as defaults; 
actual mouth bounds are computed per-image from landmarks.
'''
mouth_y1, mouth_y2 = 250, 381
mouth_x1, mouth_x2 = 0, 281


def get_landmarks(img_bgr):
    '''
    Run MediaPipe FaceMesh on a BGR image and return the 468 landmark
    points as (x, y) pixel coordinates, or None if no face is detected.
    '''
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks:
            return None
        h, w = img_bgr.shape[:2]
        lm = results.multi_face_landmarks[0].landmark
        return [(int(p.x * w), int(p.y * h)) for p in lm]


# Numeric feature extractors

def extract_mouth_opening(mouth_map_final):
    '''
    Mouth Opening (MO): distance between the two intensity peaks found in the mouth map's row-sum profile.
    A closed mouth produces a single peak; an open mouth produces two (upper/lower lip):
    the distance between them approximates how wide the mouth is open.
    '''
    sobel_x = cv2.Sobel(mouth_map_final, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(mouth_map_final, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.normalize(np.sqrt(sobel_x**2 + sobel_y**2), None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(sobel_mag, cv2.MORPH_OPEN, kernel)

    row_sums = cleaned.sum(axis=1).astype(float)
    peaks, _ = find_peaks(row_sums, distance=5, prominence=row_sums.max() * 0.1)

    if len(peaks) >= 2:
        top2 = sorted(peaks[np.argsort(row_sums[peaks])[-2:]])
        return abs(top2[1] - top2[0])  # 2 peaks = mouth open
    return 0  # 1 peak or none = mouth closed


def extract_eyebrow_features(eyebrow_gray):
    '''
    Eyebrow Constriction (EBC) and Eyebrow Mean height (EBM)

    EBM ("Method 1: Mean Intensity Tracking"):
    for each column, find the average vertical position of high-intensity pixels 
    (the eyebrow hair after a morphological gradient), then average across all columns and 
    normalized to the region size, due to regions of different sizes (/h)

    EBC ("Method 2: Degree of Curvature Line / DCL"):
    sum of the absolute differences between adjacent column heights (how much the eyebrow's 
    shape bends from one column to the next), normalized by "the length of the line" (number of columns)
    '''
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grad = cv2.morphologyEx(eyebrow_gray, cv2.MORPH_GRADIENT, kernel)
    grad_norm = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    h, w = grad_norm.shape
    threshold = grad_norm.mean() + grad_norm.std()
    heights = []
    for col in range(w):
        bright_rows = np.where(grad_norm[:, col] > threshold)[0]
        if len(bright_rows) > 0:
            heights.append(bright_rows.mean())

    if len(heights) < 2:
        return 0.0, 0.0

    heights = np.array(heights)
    ebm = heights.mean() / h
    ebc = np.sum(np.abs(np.diff(heights))) / len(heights)
    return ebc, ebm


def extract_lip_corners(Y_channel, mouth_y1, mouth_y2, mouth_x1, mouth_x2):
    '''
    Lip Corners (LC): horizontal distance between the two mouth corners: 
    dark regions (low Y / luminance) are amplified with a ^6 power,
    then thresholded at half the row's maximum intensity.

    Pixels with near-zero luminance (gray_lips < 5) are excluded before thresholding.
    These come from the pure-black background left by the elliptical face mask and without this filter
    they get amplified by the ^6 power just like real mouth-corner shadows, corrupting the
    measurement (this was a real bug found during JAFFE evaluation: LC was stuck near the ROI's full width on every image).
    '''
    gray_lips = Y_channel[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
    corners_area = (255.0 - gray_lips) ** 6
    corners_area[gray_lips < 5] = 0  # exclude pure-black mask background
    corners_area = cv2.normalize(corners_area, None, 0, 255, cv2.NORM_MINMAX)
    mid_row = corners_area[corners_area.shape[0] // 2, :]
    threshold = mid_row.max() / 2
    cols_above = np.where(mid_row > threshold)[0]
    if len(cols_above) >= 2:
        return int(cols_above[-1] - cols_above[0])
    return 0


def extract_wrinkle_count(wrinkles_canny):
    '''
    Wrinkles (W): count of edge pixels found by Canny on the forehead ROI. 
    Higher count = more visible forehead lines.
    '''
    return int(np.sum(wrinkles_canny > 0))



# Rule-based classification

'''
HES (Human Emotion Sensitivity) table: which side of each threshold each emotion is expected to fall on. 
True = feature value above its threshold.
'''
RULES = {
    "Disgust":  {"MO": False, "LC": False, "W": True,  "EBC": False, "EBM": False},
    "Surprise": {"MO": True,  "LC": True,  "W": True,  "EBC": True,  "EBM": True},
    "Angry":    {"MO": False, "LC": False, "W": False, "EBC": False, "EBM": False},
    "Neutral":  {"MO": False, "LC": False, "W": False, "EBC": True,  "EBM": True},
    "Happy":    {"MO": True,  "LC": True,  "W": False, "EBC": True,  "EBM": True},
}


def classify_emotion(mo, lc, w, ebc, ebm,
                      mo_thresh=10, lc_thresh=13, w_thresh=42,
                      ebc_thresh=2.97, ebm_thresh=0.478):
    
    '''
    Majority Voting (MV) classification:
    compare each feature to its threshold, then pick the emotion whose expected pattern matches
    the most observed booleans. Requires at least 3/5 agreement to commit to an answer;
    ties and weak matches return "Unclassified".

    Thresholds are NOT the paper's original values (25, 50, 200, 0.5, 0.7)
    because those were calibrated on the authors' own exact implementation and dataset. 
    These defaults were instead found by grid-searching threshold candidates on a 
    training split of this project's own JAFFE evaluation run (see calibrate_thresholds.py) 
    (see the project README for the resulting accuracy and a comparison against trained ML classifiers)

    NOTE: these defaults come from calibrate_thresholds.py's last run.
    If you change the feature-extraction logic above, re-run evaluate_dataset.py -> calibrate_thresholds.py 
    and update these five numbers accordingly.
    '''
    observed = {
        "MO":  mo > mo_thresh,
        "LC":  lc > lc_thresh,
        "W":   w > w_thresh,
        "EBC": ebc > ebc_thresh,
        "EBM": ebm > ebm_thresh,
    }

    scores = {
        emotion: sum(1 for feat in rule if rule[feat] == observed[feat])
        for emotion, rule in RULES.items()
    }

    sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_emotion, best_score = sorted_scores[0]
    second_emotion, second_score = sorted_scores[1]

    '''
    Angry and Neutral differ only in EBC/EBM, so fall back to  EBM alone to break a tie between exactly these two.
    '''
    if best_score == second_score and best_score < 5:
        if {best_emotion, second_emotion} == {"Angry", "Neutral"}:
            return "Neutral" if observed["EBM"] else "Angry"
        return "Unclassified"

    if scores[best_emotion] >= 3:
        return best_emotion
    return "Unclassified"


# Main pipeline

def process_image(image_path, save_output=False):
    '''
    Run the full pipeline on a single image

    Returns a dict with the 5 numeric features and the predicted emotion,
    or None if the image can't be read or no face/landmarks are found.
    Set save_output=True to also write a labeled side-by-side visualization
    to output.jpg (used by the interactive demo, f(), below).
    '''
    img = cv2.imread(image_path)
    if img is None:
        return None

    # Normalize to 3 channels (some input images may already be grayscale on disk, e.g. single-channel TIFFs).
    if len(img.shape) == 2 or img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Face Detection (Viola-Jones)
    gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_full, scaleFactor=1.05, minNeighbors=3, minSize=(60, 60))

    if len(faces) == 0:
        # using the whole image if no face is detected 
        face_crop = img
    else:
        # take the largest detected face box (in case of multiple detections)
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        pad = int(0.15 * h)  # margin so the crop doesn't cut off chin/forehead
        y1, y2 = max(0, y - pad), min(img.shape[0], y + h + pad)
        x1, x2 = max(0, x - pad), min(img.shape[1], x + w + pad)
        face_crop = img[y1:y2, x1:x2]

    # Elliptic Mask: isolate the face, remove background 
    mask = np.zeros(face_crop.shape[:2], dtype=np.uint8)
    center = (face_crop.shape[1] // 2, face_crop.shape[0] // 2)
    axes = (face_crop.shape[1] // 2, face_crop.shape[0] // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    face_masked = cv2.bitwise_and(face_crop, face_crop, mask=mask)

    # Standardized Scaling (381x281, as specified in the paper) 
    img = cv2.resize(face_masked, (281, 381))

    # Facial landark detectionnn
    landmarks = get_landmarks(img)
    if landmarks is None:
        print("No facial landmarks detected.")
        return None

    # Eyebrow band: tight vertical box around both eyebrows
    eyebrow_idx = LEFT_EYEBROW + RIGHT_EYEBROW
    eyebrow_ys = [landmarks[i][1] for i in eyebrow_idx]
    y_start_ebrow = max(0, min(eyebrow_ys) - 12)
    y_end_ebrow = min(img.shape[0], max(eyebrow_ys) + 12)

    # Forehead band: from the top-of-forehead landmark down to just above the eyebrows
    forehead_top_y = landmarks[FOREHEAD_TOP][1]
    y_start_fore = max(0, forehead_top_y + 5)
    y_end_fore = max(y_start_fore + 10, y_start_ebrow - 5)

    # Mouth region: real x/y bounds from the mouth landmarks, instead of a fixed full-width slice
    mouth_x1 = max(0, min(landmarks[MOUTH_LEFT][0], landmarks[MOUTH_RIGHT][0]) - 15)
    mouth_x2 = min(img.shape[1], max(landmarks[MOUTH_LEFT][0], landmarks[MOUTH_RIGHT][0]) + 15)
    mouth_y1 = max(0, min(landmarks[MOUTH_UPPER][1], landmarks[MOUTH_LOWER][1]) - 10)
    mouth_y2 = min(img.shape[0], max(landmarks[MOUTH_UPPER][1], landmarks[MOUTH_LOWER][1]) + 10)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    '''
    Detect grayscale input: if R, G and B channels are (near-)identical, 
    there is no real chrominance information to work with, and the
    paper's Cr/Cb-based Eye Map / Mouth Map formulas become meaningless
    
    JAFFE is a grayscale dataset.
    '''
    is_grayscale = (np.allclose(img[:, :, 0], img[:, :, 1]) and
                     np.allclose(img[:, :, 1], img[:, :, 2]))

    img_mouth_full = np.zeros_like(img)

    if is_grayscale:
        # Grayscale: luminance-only approximation 
        # Eye Map: simple contrast enhancement (no real Cb/Cr to exploit).
        eye_map_final = cv2.equalizeHist(gray)
        img_eyes = cv2.cvtColor(eye_map_final, cv2.COLOR_GRAY2BGR)

        '''
        Mouth Map: inverted luminance in the mouth ROI (dark lips show up bright), 
        then reused by the same peak-detection logic as the color version.
        '''
        mouth_crop = gray[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        mouth_map_final = (cv2.bitwise_not(mouth_crop) if mouth_crop.size > 0
                            else np.zeros((10, 10), dtype=np.uint8))
        mouth_map_final = cv2.normalize(mouth_map_final, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mo = extract_mouth_opening(mouth_map_final)
        img_mouth = cv2.cvtColor(mouth_map_final, cv2.COLOR_GRAY2BGR)

        Y = gray  # luminance channel used directly for LC
        if mouth_crop.size > 0:
            h_m, w_m = mouth_map_final.shape[:2]
            img_mouth_full[mouth_y1:mouth_y1 + h_m, mouth_x1:mouth_x1 + w_m] = img_mouth
    else:
        # Color input (paper's original YCbCr-based formulas)
        ycbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(float)
        Y, Cr, Cb = cv2.split(ycbcr)
        eps = 1e-6

        # Eye Map (paper's ECrR/SCrB formula):
        # EyeMap = Cr^2 * (Cr^2 - Cr/Cb)^2
        eye_map = (Cr**2) * (Cr**2 - (Cr / (Cb + eps)))**2
        eye_map_norm = cv2.normalize(eye_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        eye_map_final = (cv2.pow(eye_map_norm / 255.0, 2) * 255.0).astype(np.uint8)
        img_eyes = cv2.cvtColor(eye_map_final, cv2.COLOR_GRAY2BGR)

        # Mouth Map (paper's Mouthmap formula):
        # Mouthmap = Cr^2 * (Cr^2 - n*Cr/Cb)^2, n = 0.95 * mean(Cr^2)/mean(Cr/Cb)
        n = 0.95 * (np.mean(Cr**2) / (np.mean(Cr / (Cb + 1.0)) + eps))
        mouth_map = (Cr**2) * np.square(Cr**2 - n * (Cr / (Cb + 1.0)))
        mouth_map_final = cv2.normalize(mouth_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        mouth_map_roi = mouth_map_final[mouth_y1:mouth_y2, mouth_x1:mouth_x2]
        mo = extract_mouth_opening(mouth_map_roi)
        img_mouth = cv2.cvtColor(mouth_map_final, cv2.COLOR_GRAY2BGR)

        Y = Y.astype(np.uint8)
        img_mouth_full = img_mouth

    # Eyebrows: EBC / EBM
    eyebrow_roi = gray[y_start_ebrow:y_end_ebrow, :]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eyebrow_grad = cv2.morphologyEx(eyebrow_roi, cv2.MORPH_GRADIENT, kernel)
    eyebrow_final = cv2.normalize(eyebrow_grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ebc, ebm = extract_eyebrow_features(eyebrow_roi)
    # Reconstruct a full-size black canvas for the display grid, pasting the small ROI result back into its real anatomical position.
    eyebrow_full = np.zeros_like(gray)
    eyebrow_full[y_start_ebrow:y_end_ebrow, :] = eyebrow_final
    img_brows = cv2.cvtColor(eyebrow_full, cv2.COLOR_GRAY2BGR)

    # Wrinkles: W 
    ''' Horizontal bounds span the full width of BOTH eyebrows (not just the two inner-corner landmarks)
    so the ROI covers the whole forehead instead of only the narrow glabella strip between the eyes since an 
    earlier version restricted to that strip and got W=0 on ~73% of the JAFFE evaluation set.''' 
    eyebrow_xs = [landmarks[i][0] for i in eyebrow_idx]
    brow_in_left = max(0, min(eyebrow_xs) - 10)
    brow_in_right = min(img.shape[1], max(eyebrow_xs) + 10)
    forehead_roi = gray[y_start_fore:y_end_fore, brow_in_left:brow_in_right]

    if forehead_roi.size > 0:
        wrinkles_canny = cv2.Canny(cv2.GaussianBlur(forehead_roi, (3, 3), 0), 40, 100)
        w = extract_wrinkle_count(wrinkles_canny)
    else:
        w = 0
        wrinkles_canny = np.zeros((10, 10), dtype=np.uint8)

    wrinkles_full = np.zeros_like(gray)
    if forehead_roi.size > 0:
        wrinkles_full[y_start_fore:y_start_fore + wrinkles_canny.shape[0],
                       brow_in_left:brow_in_right] = wrinkles_canny
    img_wrinkles = cv2.cvtColor(wrinkles_full, cv2.COLOR_GRAY2BGR)

    # Lip Corners: LC 
    lc = extract_lip_corners(Y, mouth_y1, mouth_y2, mouth_x1, mouth_x2)

    # Classification
    emotion = classify_emotion(mo, lc, w, ebc, ebm)
    print(f"MO={mo}, EBC={ebc:.3f}, EBM={ebm:.3f}, LC={lc}, W={w} -> {emotion}")

    result = {"MO": mo, "EBC": ebc, "EBM": ebm, "LC": lc, "W": w, "predicted": emotion}

    if save_output:
        def label(image, text):
            canvas = image.copy()
            cv2.putText(canvas, text, (10, canvas.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            return canvas

        res = np.hstack((
            label(img, "Original"),
            label(img_eyes, "Eye Map"),
            label(img_mouth_full, "Mouth Map"),
            label(img_brows, "Eyebrow"),
            label(img_wrinkles, "Wrinkles"),
        ))
        cv2.imwrite("output.jpg", res)

    return result


def f(image_path):
    '''
    Interactive single-image demo: runs the pipeline, prints the result, and opens a popup window with the visualization grid. 
    Not used for batch evaluation (see evaluate_dataset.py for that)
    '''
    result = process_image(image_path, save_output=True)
    if result is None:
        print("No face detected.")
        return
    print(result)
    img_display = cv2.imread("output.jpg")
    cv2.imshow("Feature Extraction demo", img_display)
    cv2.waitKey(0)


if __name__ == "__main__":
    f('face_test.jpg')
