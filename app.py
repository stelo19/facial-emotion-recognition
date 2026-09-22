"""
app.py

Gradio demo for the Facial Emotion Recognition project. Upload a photo,
see the four extracted feature maps (Eye Map, Mouth Map, Eyebrow,
Wrinkles), the five numeric features, and two side-by-side predictions:

  - the rule-based classifier (feature_extraction.classify_emotion),
    replicating the paper's HES Majority Voting system
  - a k-NN classifier trained on results.csv, the best-performing model
    found in compare_classifiers.py (52.0% CV accuracy vs. 17.8% for the
    calibrated rule-based system)

Showing both together is the point of this demo: it makes the project's
central finding (rule-based vs. trained ML on the exact same features)
visible and interactive, not just a number in a table.

Run locally with: python app.py
"""

import os
import cv2
import numpy as np
import pandas as pd
import gradio as gr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from feature_extraction import process_image, classify_emotion

FEATURES = ["MO", "EBC", "EBM", "LC", "W"]

# --- Train the k-NN comparison model once at startup, from results.csv ---
# results.csv holds only the 5 numeric features + labels per image, no
# image data itself, so it's safe and lightweight to ship alongside the
# demo (unlike the JAFFE images themselves).
df = pd.read_csv("results.csv")
knn_model = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
knn_model.fit(df[FEATURES], df["true_emotion"])


def predict(image):
    """
    image: numpy array (RGB) from Gradio's image input.
    Returns: composite feature-map image, a feature table, and the two
    predicted-emotion strings (rule-based and k-NN).
    """
    if image is None:
        return None, None, "No image provided.", "No image provided."

    # Gradio gives RGB; feature_extraction.process_image expects a file
    # path (it calls cv2.imread internally), so save a temp BGR copy.
    temp_path = "temp_input.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    result = process_image(temp_path, save_output=True)
    os.remove(temp_path)

    if result is None:
        return None, None, "No face detected.", "No face detected."

    composite = cv2.cvtColor(cv2.imread("output.jpg"), cv2.COLOR_BGR2RGB)

    feature_table = pd.DataFrame([{
        "MO": result["MO"], "EBC": round(result["EBC"], 3),
        "EBM": round(result["EBM"], 3), "LC": result["LC"], "W": result["W"],
    }])

    rule_based_pred = result["predicted"]
    knn_pred = knn_model.predict([[result[f] for f in FEATURES]])[0]

    return composite, feature_table, rule_based_pred, knn_pred


with gr.Blocks(title="Facial Emotion Recognition — Rule-Based vs. ML") as demo:
    gr.Markdown(
        "# Facial Emotion Recognition\n"
        "Reimplementation of *Aslam & Hussain (2021)* — upload a face photo to see the "
        "extracted features and compare the paper's rule-based classifier against a "
        "trained k-NN model on the same features. "
        "[Paper](https://arxiv.org/abs/2103.00658) · "
        "[GitHub](https://github.com/YOUR_USERNAME/YOUR_REPO)"
    )

    with gr.Row():
        image_input = gr.Image(label="Upload a face photo", type="numpy")
        composite_output = gr.Image(label="Extracted feature maps")

    submit_btn = gr.Button("Analyze", variant="primary")

    with gr.Row():
        feature_output = gr.Dataframe(label="Extracted features (MO, EBC, EBM, LC, W)")

    with gr.Row():
        rule_output = gr.Textbox(label="Rule-based prediction (paper's HES system)")
        knn_output = gr.Textbox(label="k-NN prediction (best ML model, 52% CV accuracy)")

    submit_btn.click(
        fn=predict,
        inputs=image_input,
        outputs=[composite_output, feature_output, rule_output, knn_output],
    )

if __name__ == "__main__":
    demo.launch()
