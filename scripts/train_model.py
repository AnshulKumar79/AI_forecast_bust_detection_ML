import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, precision_recall_curve, auc
import pickle
import os

def train_and_calibrate(train_df: pd.DataFrame, val_df: pd.DataFrame, target_col: str = 'is_bust'):
    """
    Trains a LightGBM model and calibrates it using Isotonic Regression.
    """
    # Exclude metadata columns from features
    exclude_cols = ['region_id', 'variable', 'nwp_forecast', 'ground_truth_actual', 'absolute_error', target_col]
    features = [c for c in train_df.columns if c not in exclude_cols]
    
    X_train, y_train = train_df[features], train_df[target_col]
    X_val, y_val = val_df[features], val_df[target_col]
    
    # 1. Train base classifier
    print("Training LightGBM Classifier...")
    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=100,
        random_state=42,
        class_weight='balanced' # Busts are often rare events
    )
    model.fit(X_train, y_train)
    
    # 2. Predict raw probabilities on validation split
    raw_val_probs = model.predict_proba(X_val)[:, 1]
    
    # 3. Fit Isotonic Regression calibrator
    print("Fitting Isotonic Regression Calibrator...")
    calibrator = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    calibrator.fit(raw_val_probs, y_val)
    
    # 4. Evaluate calibration
    p_cal = calibrator.predict(raw_val_probs)
    brier = brier_score_loss(y_val, p_cal)
    
    # Precision-Recall AUC
    precision, recall, _ = precision_recall_curve(y_val, p_cal)
    pr_auc = auc(recall, precision)
    
    print(f"Validation Brier Score: {brier:.4f}")
    print(f"Validation PR-AUC: {pr_auc:.4f}")
    
    return model, calibrator, features

def save_models(model, calibrator, output_dir="../bustlib/weights"):
    """
    Saves the trained models to the bustlib weights directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save booster
    model.booster_.save_model(os.path.join(output_dir, "lgbm_model.booster"))
    
    # Save calibrator
    with open(os.path.join(output_dir, "calibrator.pkl"), "wb") as f:
        pickle.dump(calibrator, f)
        
    print(f"Models saved successfully to {output_dir}/")

if __name__ == "__main__":
    print("Training script initialized. Ready to train on engineered dataset.")
    # Example usage:
    # train_df = pd.read_csv("data/train.csv")
    # val_df = pd.read_csv("data/val.csv")
    # model, cal, feats = train_and_calibrate(train_df, val_df)
    # save_models(model, cal)
