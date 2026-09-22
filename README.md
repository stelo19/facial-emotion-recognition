# Facial Emotion Recognition — Rule-Based vs. Machine Learning

A critical reimplementation and evaluation of the feature-extraction stage described in:

> Aslam, A. & Hussain, B. (2021). *Emotion recognition techniques with rule based and
> machine learning approaches*. arXiv:2103.00658. https://arxiv.org/abs/2103.00658

This project extracts five geometric facial features (mouth opening, eyebrow shape,
lip-corner distance, forehead wrinkles) from an image and classifies the expression into
one of five emotions (Angry, Disgust, Happy, Neutral, Surprise). It also benchmarks the
paper's rule-based approach against several trained ML classifiers on the same features —
directly testing the paper's own framing: *"rule based **and** machine learning approaches."*

## What this is — and isn't

This reproduces the **feature-extraction and classification stage** of the paper, not the
full system: face detection uses Viola-Jones + MediaPipe FaceMesh landmarks (the paper uses
its own eye-detection method), and the reported accuracy is measured independently on a
public dataset, not copied from the paper. See **Results** and **Discussion** below for an
honest account of where this implementation diverges from the original and why.

## Repository Structure

```
.
├── feature_extraction.py     # Core pipeline: face detection, landmarks, feature extraction,
│                              # rule-based classification (see module docstring for details)
├── evaluate_dataset.py       # Batch-runs the pipeline over JAFFE, saves results.csv
├── calibrate_thresholds.py   # Grid-searches rule-based thresholds on a train split,
│                              # reports honest accuracy on a held-out test split
├── compare_classifiers.py    # 5-fold CV benchmark: rule-based vs. 5 ML classifiers
├── app.py                    # Gradio demo — see Demo section below
├── results.csv                # Extracted features + predictions (JAFFE)
├── requirements.txt
└── README.md
```

`results.csv` **is** committed — it contains only the five numeric features and labels per
image, no image data, so there's no licensing concern, and both the demo and the
calibration/comparison scripts need it to run without requiring a full JAFFE download just
to try things out. The raw dataset images themselves are **not** included.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows; use `source venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

**Dataset** (only needed if you want to regenerate `results.csv` yourself): this project
uses [JAFFE](https://zenodo.org/record/3451524) (DOI 10.5281/zenodo.3451524), freely
available, ~213 grayscale images across 7 emotions. Download it and place the images under
`data/jaffe/`.

```bash
python evaluate_dataset.py       # (optional) regenerates results.csv
python calibrate_thresholds.py   # calibrates and validates the rule-based thresholds
python compare_classifiers.py    # runs the rule-based vs. ML comparison
python app.py                    # launches the interactive demo
```

**Note on the workflow**: `evaluate_dataset.py` is the only script that processes actual
images (face detection + landmark extraction via `feature_extraction.py`) — it's the slow
step. `calibrate_thresholds.py` and `compare_classifiers.py` only read the numeric features
already saved in `results.csv`; they never touch the images or `feature_extraction.py`
directly, so they run in seconds and can be re-run freely (e.g. to try different
classifiers or threshold ranges) without reprocessing the dataset. Only re-run
`evaluate_dataset.py` if you change something inside `feature_extraction.py`'s feature
extraction logic itself — otherwise the existing `results.csv` is still valid.

## Pipeline

1. **Face detection** — Viola-Jones (Haar Cascade) + an elliptical mask to isolate the face
   and remove background, then resized to 281×381 (as specified in the paper).
2. **Landmark localization** — MediaPipe FaceMesh (468 points), used to precisely locate the
   eyebrows, forehead and mouth regions on each individual face, rather than assuming fixed
   proportions of the image.
3. **Feature extraction** — five features per the paper's formulas:
   - **MO** (Mouth Opening): distance between the two intensity peaks in the mouth map
   - **EBC** (Eyebrow Constriction) / **EBM** (Eyebrow Mean height): shape and position of
     the eyebrow, via a morphological gradient
   - **LC** (Lip Corners): horizontal mouth-corner distance, via inverted-luminance mapping
   - **W** (Wrinkles): Canny edge count on the forehead
4. **Classification** — a rule-based Majority Voting system (paper section 3.3), following
   the HES (Human Emotion Sensitivity) threshold table.

Color images use the paper's original YCbCr chrominance-based formulas for the Eye Map and
Mouth Map. **Grayscale images are detected automatically** and handled with a
luminance-based fallback instead — the paper's Cr/Cb-based formulas are meaningless on
grayscale input, since grayscale photos carry no real chrominance information (this matters
in practice: JAFFE, the dataset used for evaluation, is grayscale).

## Results

Evaluated on 150 JAFFE images (5 emotions, Fear/Sad excluded — not covered by the rule
table), with a stratified train/test split for the rule-based system and 5-fold
cross-validation for the ML classifiers:

| Method | Accuracy |
|---|---|
| Rule-based (paper's thresholds, uncalibrated) | 14.0% |
| Rule-based (thresholds calibrated via grid search) | 17.8% (held-out test) |
| Decision Tree | 38.7% ± 7.8% (5-fold CV) |
| Logistic Regression | 41.3% ± 6.9% (5-fold CV) |
| Random Forest | 46.0% ± 8.3% (5-fold CV) |
| SVM (RBF kernel) | 50.7% ± 4.9% (5-fold CV) |
| **k-NN (k=5)** | **52.0% ± 4.5% (5-fold CV) — best accuracy/stability trade-off** |
| Ensemble (SVM + k-NN + Random Forest) | 53.3% ± 4.7% (5-fold CV) |

**Feature importance** (Decision Tree): EBM and EBC (eyebrow shape/position) are by far the
most informative features; LC (lip corners) carries close to no discriminative signal in
this implementation (confirmed independently via one-way ANOVA, p=0.09 — the only feature of
the five with no statistically significant difference across emotion classes).

The ~3x gap between the rule-based system and the best ML classifier, on the exact same five
numeric features, is the central empirical finding of this project.

## Demo

`app.py` is a Gradio app: upload a face photo and see the four extracted feature maps, the
five numeric features, and the emotion predicted by both the rule-based system and a k-NN
classifier trained on `results.csv`. Run locally with `python app.py`.

**Known limitation**: the k-NN model and rule-based thresholds are currently calibrated
exclusively on grayscale JAFFE data. Color input (most real-world photos) is processed
through a different code path internally (see Pipeline above) and produces feature values
on a different scale — predictions on color photos should be treated as unreliable for now.
Extending calibration to color data is planned future work.

## Discussion & Limitations

- **The paper never reports JAFFE evaluated alone.** Its own comparison table lists results
  only for JAFFE+SFEW, JAFFE+SFEW, RaFD, and RaFD+JAFFE+SFEW combinations — never JAFFE in
  isolation. This is consistent with what this reimplementation found directly: the paper's
  chrominance-based formulas have little to work with on a grayscale-only dataset, so a fair
  like-for-like comparison against the paper's reported 94.18% isn't really possible using
  JAFFE alone.
- **Several implementation details are qualitative, not numeric**, in the paper (e.g. "high-
  intensity pixels", morphological kernel sizes, exact Canny thresholds) — this
  reimplementation had to choose reasonable values for these, which will differ from the
  authors' exact numbers and compounds across five chained features. The exact training
  methodology (repeated train/test iteration vs. a single held-out check) also isn't fully
  specified in the paper, which likely contributes further to the gap.
- **Eye localization** underpins the paper's entire ROI geometry; this project substitutes
  MediaPipe FaceMesh landmarks for the paper's own eye-detection method — conceptually
  similar, not identical.

## Citation

If referencing this work, please also cite the original paper:
```
Aslam, A. & Hussain, B. (2021). Emotion recognition techniques with rule based and
machine learning approaches. arXiv:2103.00658.
```

JAFFE dataset: Lyons, M. et al. — https://zenodo.org/record/3451524 (DOI 10.5281/zenodo.3451524)
