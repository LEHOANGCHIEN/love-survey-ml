# --- Compatibility class for FrequencyEncoder ---
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator

class FrequencyEncoder(TransformerMixin, BaseEstimator):
    """
    Simple frequency encoder compatible with saved pipeline.
    Replaces categorical values by their frequency (proportion) computed at fit().
    """
    def __init__(self, cols: Optional[List[str]] = None, normalize: bool = True, fill_value: float = 0.0):
        self.cols = cols
        self.normalize = normalize
        self.fill_value = fill_value
        self.freq_maps_ = {}

    def fit(self, X, y=None):
        if isinstance(X, (np.ndarray, list)):
            X = pd.DataFrame(X)
        df = X.copy()
        cols = self.cols if self.cols is not None else list(df.columns)
        for c in cols:
            ser = df[c].astype("object")
            counts = ser.value_counts(dropna=False)
            if self.normalize:
                freqs = (counts / counts.sum()).to_dict()
            else:
                freqs = counts.to_dict()
            self.freq_maps_[c] = freqs
        return self

    def transform(self, X):
        if isinstance(X, (np.ndarray, list)):
            X = pd.DataFrame(X)
        df = X.copy()
        cols = self.cols if self.cols is not None else list(df.columns)
        for c in cols:
            fmap = self.freq_maps_.get(c, {})
            df[c] = df[c].map(lambda v: fmap.get(v, self.fill_value)).astype(float)
        return df

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
# --- End FrequencyEncoder ---

# ----- predict_example.py (auto-detect transformed/raw) -----
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# load bundle (path relative to where you run script)
BUNDLE_PATH = Path("notebooks/models/model_bundle.pkl")
if not BUNDLE_PATH.exists():
    BUNDLE_PATH = Path("models/model_bundle.pkl")  # fallback

bundle = joblib.load(str(BUNDLE_PATH))
model = bundle["model"]
pre = bundle["preprocessor"]

# Try to load expected transformed-feature names (X_features.csv) to compare
X_feat_path = Path("data/X_features.csv")
transformed_cols = None
if X_feat_path.exists():
    try:
        transformed_cols = list(pd.read_csv(X_feat_path, encoding="utf-8-sig", nrows=0).columns)
    except:
        transformed_cols = None

def predict_from_df(df: pd.DataFrame):
    # If df columns match transformed feature names, assume already transformed
    if transformed_cols is not None and list(df.columns) == transformed_cols:
        print("Input looks LIKE transformed feature matrix (X_features). Bypassing preprocessor.")
        X_in = df.copy()
    else:
        # Otherwise assume raw => apply preprocessor (transform)
        print("Input treated as RAW survey data. Applying preprocessor.transform(...)")
        # pre.transform may require columns in certain order; assume df has raw columns expected
        X_in = pre.transform(df)
        # pre.transform returns np.array or DataFrame; ensure DataFrame with correct columns for model
        if isinstance(X_in, np.ndarray):
            # model expects numeric array -> OK
            pass
        else:
            # if DataFrame, convert to numpy
            try:
                X_in = X_in.values
            except:
                X_in = np.asarray(X_in)

    # Predict
    pred = model.predict(X_in)
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(X_in)[:, 1]
        except Exception:
            prob = None
    return pred, prob

if __name__ == "__main__":
    # adjust path to your example file: either notebooks/example_input.csv or ./example_input.csv
    # If running from project root and file is in root: use "../example_input.csv" if needed.
    possible_paths = [
        Path("notebooks/example_input.csv"),
        Path("example_input.csv"),
        Path("data/example_input.csv"),
        Path("notebooks/example_input_raw.csv"),
    ]
    input_path = None
    for p in possible_paths:
        if p.exists():
            input_path = p
            break
    if input_path is None:
        raise FileNotFoundError("No example_input.csv found. Create one in project root or notebooks/.")
    print("Loading input from:", input_path)
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    p, prob = predict_from_df(df)
    print("pred:", p)
    print("prob:", prob)
