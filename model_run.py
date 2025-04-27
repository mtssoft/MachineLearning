import os
import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# === Ayarlar ===
data_root = r"C:\\Users\\tugru\\Desktop\\Machine Learning Datasets"
output_dir = os.path.join(data_root, "Result")
os.makedirs(output_dir, exist_ok=True)

datasets = {
    "CSE": os.path.join(data_root, "CSE-CIC-IDS2018", "Cleaned"),
    "BETH": os.path.join(data_root, "BETH", "Cleaned")
}

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(
        tree_method='gpu_hist', 
        n_jobs=-1,
        random_state=42,
        verbosity=0
    ),
    "LightGBM": LGBMClassifier(
        device='gpu',
        random_state=42,
        n_jobs=-1,
        min_data_in_leaf=50,
        min_sum_hessian_in_leaf=1e-3,
        force_col_wise=True
    )
}

results = []
detailed_logs = []

for dataset_name, path in datasets.items():
    print(f"\n📂 Data Set: {dataset_name}")
    X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(path, "y_train.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(path, "y_test.csv")).squeeze()

    for df in [X_train, X_test]:
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        df.fillna(0, inplace=True)

    for model_name, model in models.items():
        print(f"\n🚀 Model: {model_name} — Data Set: {dataset_name}")
        start_time = time.time()

        try:
            if model_name == "XGBoost":
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                evals_result = model.evals_result()
                for epoch, loss in enumerate(evals_result["validation_0"]["logloss"]):
                    detailed_logs.append({
                        "DataSet": dataset_name,
                        "Model": model_name,
                        "Epoch": epoch + 1,
                        "LogLoss": loss
                    })
            elif model_name == "LightGBM":
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="binary_logloss")
                if hasattr(model, "_evals_result"):
                    for epoch, loss in enumerate(model._evals_result["valid_0"]["binary_logloss"]):
                        detailed_logs.append({
                            "DataSet": dataset_name,
                            "Model": model_name,
                            "Epoch": epoch + 1,
                            "LogLoss": loss
                        })
            else:
                model.fit(X_train, y_train)

            end_time = time.time()
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_pred)

            print(f"✅ Accuracy: {acc:.4f} — Precision: {prec:.4f} — Recall: {rec:.4f} — F1: {f1:.4f} — ROC-AUC: {roc:.4f}")
            print("Confusion Matrix:")
            print(confusion_matrix(y_test, y_pred))

            results.append({
                "DataSet": dataset_name,
                "Model": model_name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
                "ROC_AUC": roc,
                "TrainingTimeSec": round(end_time - start_time, 2)
            })

        except Exception as e:
            print(f"❌ HATA: {model_name} - {dataset_name} çalıştırılamadı. Sebep: {e}")

# === Sonuçları kaydet ===
results_csv = os.path.join(output_dir, "machine_learning_results.csv")
pd.DataFrame(results).to_csv(results_csv, index=False)
print(f"\n📁 Sonuçlar kaydedildi: {results_csv}")

if detailed_logs:
    log_csv = os.path.join(output_dir, "machine_learning_training_log.csv")
    pd.DataFrame(detailed_logs).to_csv(log_csv, index=False)
    print(f"📁 Epoch logları kaydedildi: {log_csv}")

print("\n📊 Toplu Sonuçlar:")
print(pd.DataFrame(results).to_string(index=False))
