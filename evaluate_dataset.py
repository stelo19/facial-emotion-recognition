'''
evaluate_dataset.py

Batch-runs the feature_extraction pipeline over a labeled dataset and saves
per-image results (the 5 numeric features + predicted emotion) to results.csv. 
That CSV is the input for calibrate_thresholds.py and compare_classifiers.py.

Currently configured for the JAFFE dataset 
(Lyons et al., freely available via Zenodo, DOI 10.5281/zenodo.3451524), 
whose filenames encode the true emotion label directly, e.g. "KA.AN1.39.tiff" -> code "AN" -> Angry.

Only the 5 emotions covered by the paper's HES rule table (Angry, Disgust, Happy, Neutral, Surprise) are kept; 
JAFFE's Fear and Sad images are skipped, since the classifier has no rule that could ever predict them correctly.

To evaluate on a different dataset, change collect_jaffe_images();
the rest of this script (feature extraction + CSV export) stays the same.
'''

import os
import glob
import pandas as pd
from feature_extraction import process_image

# JAFFE filename format: <model>.<EMOTION_CODE><number>.<id>.tiff
JAFFE_LABEL_MAP = {
    "HA": "Happy",
    "AN": "Angry",
    "DI": "Disgust",
    "NE": "Neutral",
    "SU": "Surprise",
    # "FE" (Fear) and "SA" (Sad) intentionally omitted: not covered by classify_emotion()'s rule table.
}


def collect_jaffe_images(dataset_root="./data/jaffe"):
    '''
    Scan dataset_root for JAFFE images and return a list of (image_path, true_emotion) pairs,
    skipping any file whose emotion code isn't in JAFFE_LABEL_MAP (Fear/Sad, or anything unrecognized).
    '''
    pairs = []
    files = glob.glob(os.path.join(dataset_root, "*.*"))  # supports .tiff/.tif/.jpg/.png
    for path in sorted(files):
        fname = os.path.basename(path)
        parts = fname.split(".")
        if len(parts) >= 3:
            code = parts[1][:2].upper()
            if code in JAFFE_LABEL_MAP:
                pairs.append((path, JAFFE_LABEL_MAP[code]))
    return pairs


def main():
    pairs = collect_jaffe_images("./data/jaffe")
    print(f"Found {len(pairs)} images across {len(JAFFE_LABEL_MAP)} emotions.")

    rows = []
    for i, (path, true_label) in enumerate(pairs):
        result = process_image(path)
        if result is None:
            continue  # no face/landmarks detected = skip 
        rows.append({"filename": os.path.basename(path), "true_emotion": true_label, **result})

        if i % 50 == 0:
            print(f"  Processed {i}/{len(pairs)}...")

    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)
    print(f"Saved {len(df)} results to results.csv")


if __name__ == "__main__":
    main()