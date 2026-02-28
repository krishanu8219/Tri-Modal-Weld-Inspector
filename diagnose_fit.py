"""Overfitting / Underfitting Diagnostic — Train vs Validation loss comparison."""
import os, logging, numpy as np, pandas as pd
from sklearn.metrics import log_loss, f1_score, accuracy_score
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

from src.train_binary import prepare_data
from src.train_multiclass import prepare_multiclass_data
from sklearn.preprocessing import LabelEncoder

def main():
    df_train = pd.read_csv("train_split.csv")
    df_val = pd.read_csv("val_split.csv")

    # ── Binary Model ──
    logging.info("=== BINARY MODEL DIAGNOSTICS ===")
    bin_model = joblib.load("artifacts/binary_model.joblib")
    
    X_train, y_train, _ = prepare_data(df_train)
    X_val, y_val, _ = prepare_data(df_val)
    
    train_probs = bin_model.predict_proba(X_train)
    val_probs = bin_model.predict_proba(X_val)
    
    if train_probs.shape[1] > 1:
        train_p = train_probs[:, 1]
        val_p = val_probs[:, 1]
    else:
        train_p = train_probs[:, 0]
        val_p = val_probs[:, 0]
    
    train_loss = log_loss(y_train, train_p)
    val_loss = log_loss(y_val, val_p)
    
    train_preds = (train_p >= 0.58).astype(int)
    val_preds = (val_p >= 0.58).astype(int)
    
    train_f1 = f1_score(y_train, train_preds, zero_division=0)
    val_f1 = f1_score(y_val, val_preds, zero_division=0)
    train_acc = accuracy_score(y_train, train_preds)
    val_acc = accuracy_score(y_val, val_preds)
    
    gap = train_loss - val_loss
    
    print("\n" + "="*55)
    print("  BINARY MODEL — Overfitting / Underfitting Report")
    print("="*55)
    print(f"  {'Metric':<25} {'Train':>10} {'Val':>10} {'Gap':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Log Loss':<25} {train_loss:>10.4f} {val_loss:>10.4f} {gap:>+10.4f}")
    print(f"  {'F1 Score':<25} {train_f1:>10.4f} {val_f1:>10.4f} {train_f1-val_f1:>+10.4f}")
    print(f"  {'Accuracy':<25} {train_acc:>10.4f} {val_acc:>10.4f} {train_acc-val_acc:>+10.4f}")
    
    if abs(train_f1 - val_f1) < 0.03 and val_f1 > 0.90:
        verdict_bin = "✅ GOOD FIT — Small train-val gap, high val performance"
    elif train_f1 - val_f1 > 0.05:
        verdict_bin = "⚠️ OVERFITTING — Train F1 much higher than Val F1"
    elif val_f1 < 0.80:
        verdict_bin = "⚠️ UNDERFITTING — Val F1 too low"
    else:
        verdict_bin = "✅ SLIGHT OVERFIT but acceptable — Val F1 still strong"
    
    print(f"\n  Verdict: {verdict_bin}\n")
    
    # ── Multiclass Model ──
    logging.info("=== MULTICLASS MODEL DIAGNOSTICS ===")
    multi_model = joblib.load("artifacts/multiclass_model.joblib")
    le = joblib.load("artifacts/label_encoder.joblib")
    
    X_train_m, y_train_m, _ = prepare_multiclass_data(df_train)
    X_val_m, y_val_m, _ = prepare_multiclass_data(df_val)
    
    y_train_enc = le.transform(y_train_m)
    y_val_enc = le.transform(y_val_m)
    
    train_probs_m = multi_model.predict_proba(X_train_m)
    val_probs_m = multi_model.predict_proba(X_val_m)
    
    train_loss_m = log_loss(y_train_enc, train_probs_m, labels=list(range(len(le.classes_))))
    val_loss_m = log_loss(y_val_enc, val_probs_m, labels=list(range(len(le.classes_))))
    
    train_preds_m = multi_model.predict(X_train_m)
    val_preds_m = multi_model.predict(X_val_m)
    
    train_f1_m = f1_score(y_train_enc, train_preds_m, average='macro', zero_division=0)
    val_f1_m = f1_score(y_val_enc, val_preds_m, average='macro', zero_division=0)
    train_acc_m = accuracy_score(y_train_enc, train_preds_m)
    val_acc_m = accuracy_score(y_val_enc, val_preds_m)
    
    gap_m = train_loss_m - val_loss_m
    
    print("="*55)
    print("  MULTICLASS MODEL — Overfitting / Underfitting Report")
    print("="*55)
    print(f"  {'Metric':<25} {'Train':>10} {'Val':>10} {'Gap':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Log Loss':<25} {train_loss_m:>10.4f} {val_loss_m:>10.4f} {gap_m:>+10.4f}")
    print(f"  {'Macro F1':<25} {train_f1_m:>10.4f} {val_f1_m:>10.4f} {train_f1_m-val_f1_m:>+10.4f}")
    print(f"  {'Accuracy':<25} {train_acc_m:>10.4f} {val_acc_m:>10.4f} {train_acc_m-val_acc_m:>+10.4f}")
    
    if abs(train_f1_m - val_f1_m) < 0.05 and val_f1_m > 0.85:
        verdict_multi = "✅ GOOD FIT — Small train-val gap, high val performance"
    elif train_f1_m - val_f1_m > 0.10:
        verdict_multi = "⚠️ OVERFITTING — Train F1 much higher than Val F1"
    elif val_f1_m < 0.70:
        verdict_multi = "⚠️ UNDERFITTING — Val Macro F1 too low"
    else:
        verdict_multi = "✅ SLIGHT OVERFIT but acceptable — Val Macro F1 still strong"
    
    print(f"\n  Verdict: {verdict_multi}\n")

if __name__ == "__main__":
    main()
