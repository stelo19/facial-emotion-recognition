'''
calibrate_thresholds.py

Finds threshold values for the rule-based classify_emotion() function (feature_extraction.py) 
by grid-searching over candidate cutoffs computed from a TRAINING split of results.csv, 
then reports accuracy on a held-out TEST split (an honest, non-overfit number).

Why this script exists: the paper's own thresholds (MO>25, LC>50, W>200, EBC>0.5, EBM>0.7) 
were calibrated on the authors' exact implementation and their own 600-image training set. 
Earlier attempts in this project to guess thresholds "by eye" from feature distributions
kept failing for the same reason. This script replicates what the paper actually did: 
pick thresholds by evaluating them against labeled data, not by intuition.

Candidate values per feature are the 25th/40th/50th/60th/75th percentiles
observed in the training split; reasonable, data-grounded cutoff points,
rather than arbitrary guesses.
'''

import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split

from feature_extraction import classify_emotion

df = pd.read_csv("results.csv")
train, test = train_test_split(df, test_size=0.3, stratify=df["true_emotion"], random_state=42)


def candidates(col):
    return np.percentile(train[col], [25, 40, 50, 60, 75])


best_acc, best_params = 0, None
for mo_t, lc_t, w_t, ebc_t, ebm_t in product(
        candidates("MO"), candidates("LC"), candidates("W"),
        candidates("EBC"), candidates("EBM")):
    preds = train.apply(lambda r: classify_emotion(
        r.MO, r.LC, r.W, r.EBC, r.EBM,
        mo_thresh=mo_t, lc_thresh=lc_t, w_thresh=w_t,
        ebc_thresh=ebc_t, ebm_thresh=ebm_t), axis=1)
    acc = (preds == train["true_emotion"]).mean()
    if acc > best_acc:
        best_acc, best_params = acc, (mo_t, lc_t, w_t, ebc_t, ebm_t)

print("Best thresholds found (MO, LC, W, EBC, EBM):", best_params)
print(f"Training accuracy: {best_acc:.1%}")

# Honest validation: evaluate the chosen thresholds on data that played no part in choosing them.
mo_t, lc_t, w_t, ebc_t, ebm_t = best_params
test_preds = test.apply(lambda r: classify_emotion(
    r.MO, r.LC, r.W, ebc_t, ebm_t,
    mo_thresh=mo_t, lc_thresh=lc_t, w_thresh=w_t), axis=1)
test_acc = (test_preds == test["true_emotion"]).mean()
print(f"Test accuracy (honest, held-out): {test_acc:.1%}")

# Refresh the 'predicted' column across the FULL dataset (not just train/test)
# using the newly calibrated thresholds, so results.csv stays consistent with
# whatever thresholds this run just found (no manual copy-paste needed here)
df["predicted"] = df.apply(lambda r: classify_emotion(
    r.MO, r.LC, r.W, ebc_t, ebm_t,
    mo_thresh=mo_t, lc_thresh=lc_t, w_thresh=w_t), axis=1)
df.to_csv("results.csv", index=False)
print("results.csv updated with predictions from the calibrated thresholds.")

'''
Reminder: results.csv is now consistent with the thresholds above, 
but feature_extraction.py's classify_emotion() defaults are NOT updated automatically: 
copy the "Best thresholds found" values printed above into that function's signature by hand.
'''