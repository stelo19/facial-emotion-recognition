'''
compare_classifiers.py

Compares the rule-based classifier (feature_extraction.classify_emotion) against several trained ML classifiers
on the same 5 numeric features (MO, EBC, EBM, LC, W) from results.csv.
A 5-fold stratified cross-validation benchmark across 5 classifiers (+ a soft-voting ensemble of the top 3), 
reporting mean accuracy and standard deviation.
Cross-validation is used instead of a single train/test split because the dataset is small (150 images) and 
averaging over 5 folds gives a much more stable, defensible number.

StandardScaler is applied (via sklearn Pipelines) for Logistic Regression, SVM and k-NN, since these algorithms 
are sensitive to feature scale (e.g. W ranges ~0-250, EBM ranges ~0-0.6); without scaling, W would dominate
just because its numbers are bigger, not because it's more informative.
Tree-based models (Decision Tree, Random Forest) split on independent per-feature thresholds, so scaling doesn't matter for them.
'''

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV

df = pd.read_csv("results.csv")
FEATURES = ["MO", "EBC", "EBM", "LC", "W"]
X = df[FEATURES]
y = df["true_emotion"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Decision Tree":       DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "SVM (RBF)":           make_pipeline(StandardScaler(), SVC(kernel="rbf")),
    "k-NN (k=5)":          make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
    "Ensemble (SVM+kNN+RF)": make_pipeline(StandardScaler(), VotingClassifier(estimators=[
        ("svm", CalibratedClassifierCV(SVC(kernel="rbf"), ensemble=False)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ], voting="soft")),
}

print("=== 5-fold cross-validation accuracy ===")
print(f"{'Rule-based (calibrated thresholds)':22s}  ~17.8% (single held-out test split -- see calibrate_thresholds.py)")
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name:22s}  {scores.mean():.1%} \u00b1 {scores.std():.1%}")

# Feature importance (diagnostic only, not part of the CV benchmark) 
'''
Decision Tree fit on the full dataset, used only to read which of the 5 features the tree found most useful 
for splitting emotions apart.
'''
print("\n=== Feature importance (Decision Tree, fit on full dataset) ===")
tree = DecisionTreeClassifier(max_depth=4, random_state=42)
tree.fit(X, y)
importances = pd.Series(tree.feature_importances_, index=FEATURES).sort_values(ascending=False)
print(importances)