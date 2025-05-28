import os
import time
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from lime.lime_tabular import LimeTabularExplainer

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
    print(f"\nData Set: {dataset_name}")
    X_train = pd.read_csv(os.path.join(path, "X_train.csv"))
    y_train = pd.read_csv(os.path.join(path, "y_train.csv")).squeeze()
    X_test = pd.read_csv(os.path.join(path, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(path, "y_test.csv")).squeeze()

    for df in [X_train, X_test]:
        df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
        df.fillna(0, inplace=True)

    for model_name, model in models.items():
        print(f"\nModel: {model_name} — Data Set: {dataset_name}")
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
                model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric="binary_logloss", callbacks=[])
                evals_result = model.evals_result_
                for epoch, loss in enumerate(evals_result["valid_0"]["binary_logloss"]):
                    detailed_logs.append({
                        "DataSet": dataset_name,
                        "Model": model_name,
                        "Epoch": epoch + 1,
                        "LogLoss": float(loss)
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

            print(f"Accuracy: {acc:.4f} — Precision: {prec:.4f} — Recall: {rec:.4f} — F1: {f1:.4f} — ROC-AUC: {roc:.4f}")
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

            # === SHAP Açıklamaları ===
            print("SHAP açıklamaları hesaplanıyor...")
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test.iloc[:100])

                if isinstance(shap_values, list):
                    shap_arr = shap_values[1] if len(shap_values) > 1 else shap_values[0]
                else:
                    shap_arr = shap_values

                if shap_arr.ndim == 3 and shap_arr.shape[2] == 2:
                    shap_arr = shap_arr[:, :, 1]

                shap_df = pd.DataFrame(shap_arr, columns=X_train.columns)
                shap_df.insert(0, "SampleIndex", shap_df.index)

                shap_output_path = os.path.join(output_dir, f"shap_values_{dataset_name}_{model_name}.csv")
                shap_df.to_csv(shap_output_path, index=False)
                print(f"SHAP açıklamaları kaydedildi: {shap_output_path}")
            except Exception as shap_e:
                print(f"SHAP açıklamaları hesaplanamadı: {shap_e}")

            # === LIME Açıklamaları ===
            print("LIME açıklamaları hesaplanıyor...")
            try:
                lime_samples = X_test.iloc[:100].copy()
                lime_explainer = LimeTabularExplainer(
                    training_data=np.array(X_train),
                    feature_names=X_train.columns.tolist(),
                    class_names=["Class 0", "Class 1"],
                    mode="classification",
                    discretize_continuous=True
                )

                lime_records = []
                for i in range(100):
                    instance = lime_samples.iloc[i]
                    predict_fn = lambda x: model.predict_proba(pd.DataFrame(x, columns=X_train.columns))
                    exp = lime_explainer.explain_instance(
                        data_row=instance.values,
                        predict_fn=predict_fn,
                        num_features=min(10, X_train.shape[1])
                    )
                    for feature, weight in exp.as_list():
                        lime_records.append({
                            "SampleIndex": i,
                            "Feature": feature,
                            "Weight": weight
                        })

                lime_df = pd.DataFrame(lime_records)
                lime_output_path = os.path.join(output_dir, f"lime_values_{dataset_name}_{model_name}.csv")
                lime_df.to_csv(lime_output_path, index=False)
                print(f"LIME açıklamaları kaydedildi: {lime_output_path}")

            except Exception as lime_e:
                print(f"LIME açıklamaları hesaplanamadı: {lime_e}")

        except Exception as e:
            print(f"HATA: {model_name} - {dataset_name} çalıştırılamadı. Sebep: {e}")

# === Sonuçları kaydet ===
results_csv = os.path.join(output_dir, "machine_learning_results.csv")
pd.DataFrame(results).to_csv(results_csv, index=False)
print(f"\nSonuçlar kaydedildi: {results_csv}")

if detailed_logs:
    log_csv = os.path.join(output_dir, "machine_learning_training_log.csv")
    pd.DataFrame(detailed_logs).to_csv(log_csv, index=False)
    print(f"Epoch logları kaydedildi: {log_csv}")

print("\nToplu Sonuçlar:")
print(pd.DataFrame(results).to_string(index=False))
