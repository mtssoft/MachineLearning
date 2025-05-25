import os
import gc
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, Conv1D, MaxPooling1D, Flatten, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from lime.lime_tabular import LimeTabularExplainer
from shap.explainers._permutation import PermutationExplainer

# === GPU Ayarı ===
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU kullanılabilir:", gpus[0].name)
else:
    print("GPU kullanılabilir değil!")

# === Dosya Ayarları ===
data_root = r"C:\\Users\\tugru\\Desktop\\Machine Learning Datasets"
datasets = {
    "CSE": os.path.join(data_root, "CSE-CIC-IDS2018", "Cleaned"),
    "BETH": os.path.join(data_root, "BETH", "Cleaned")
}
output_dir = os.path.join(data_root, "Result")
os.makedirs(output_dir, exist_ok=True)
output_csv = os.path.join(output_dir, "deep_results.csv")
log_csv = os.path.join(output_dir, "deep_training_log.csv")

results, training_logs = [], []

# === XAI Fonksiyonları ===
def explain_with_shap(model, X_sample, dataset, model_name, feature_names, reshape_fn=None):
    try:
        print("XAI: SHAP açıklamaları hesaplanıyor...")
        X_sample = np.array(X_sample, dtype=np.float32)
        def pred_fn(x):
            x_pred = reshape_fn(x) if reshape_fn else x
            preds = model.predict(x_pred, verbose=0).flatten()
            return preds
        explainer = PermutationExplainer(pred_fn, X_sample)
        shap_values = explainer(X_sample)
        shap_df = pd.DataFrame(shap_values.values, columns=feature_names)
        shap_df.insert(0, "SampleIndex", shap_df.index)
        shap_df.to_csv(os.path.join(output_dir, f"shap_{dataset}_{model_name}.csv"), index=False)
        print(f"XAI: SHAP çıktısı kaydedildi → shap_{dataset}_{model_name}.csv")
    except Exception as e:
        print(f"XAI hatası (SHAP): {e}")


def explain_with_lime(model, X_train, X_sample, dataset, model_name, feature_names, reshape_fn=None):
    try:
        print("XAI: LIME açıklamaları hesaplanıyor...")
        explainer = LimeTabularExplainer(
            training_data=np.array(X_train),
            feature_names=feature_names,
            class_names=["Class 0", "Class 1"],
            mode="classification"
        )
        lime_records = []
        for i in range(min(100, X_sample.shape[0])):
            row = X_sample[i]
            def pred_fn(x):
                x_r = reshape_fn(x) if reshape_fn else x
                preds = model.predict(x_r, verbose=0).flatten()
                return np.column_stack([1 - preds, preds])
            exp = explainer.explain_instance(data_row=row, predict_fn=pred_fn, num_features=10)
            for feature, weight in exp.as_list():
                lime_records.append({"SampleIndex": i, "Feature": feature, "Weight": weight})
        pd.DataFrame(lime_records).to_csv(os.path.join(output_dir, f"lime_{dataset}_{model_name}.csv"), index=False)
        print(f"XAI: LIME çıktısı kaydedildi → lime_{dataset}_{model_name}.csv")
    except Exception as e:
        print(f"XAI hatası (LIME): {e}")

# === Değerlendirme ===
def evaluate_model(model, X_test, y_test, name, dataset, history, duration, X_train=None, feature_names=None, reshape_fn=None):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)
    results.append({
        "Dataset": dataset, "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_pred_prob),
        "Duration_sec": round(duration, 2)
    })
    for i, loss in enumerate(history.history['loss']):
        training_logs.append({
            "Dataset": dataset, "Model": name,
            "Epoch": i + 1, "Loss": loss,
            "Accuracy": history.history['accuracy'][i]
        })
    if X_train is not None and feature_names is not None:
        explain_with_shap(model, X_test[:100], dataset, name, feature_names, reshape_fn)
        explain_with_lime(model, X_train, X_test[:100], dataset, name, feature_names, reshape_fn)
    del model, y_pred, y_pred_prob
    gc.collect()
    tf.keras.backend.clear_session()

# === Ana Eğitim Döngüsü ===
for dataset_name, path in datasets.items():
    print(f"\nVeri seti başlıyor: {dataset_name}")
    X_train_df = pd.read_csv(os.path.join(path, "X_train.csv"))
    X_test_df = pd.read_csv(os.path.join(path, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(path, "y_train.csv")).squeeze().astype(np.float32)
    y_test = pd.read_csv(os.path.join(path, "y_test.csv")).squeeze().astype(np.float32)
    feature_names = X_train_df.columns.tolist()
    for df in [X_train_df, X_test_df]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_df.astype(np.float32))
    X_test = scaler.transform(X_test_df.astype(np.float32))
    input_dim = X_train.shape[1]

    # === CNN ===
    X_train_cnn = np.expand_dims(X_train, axis=2)
    X_test_cnn = np.expand_dims(X_test, axis=2)
    cnn = Sequential([
        Conv1D(32, kernel_size=3, activation='relu', input_shape=(input_dim, 1)),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    cnn.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    start = time.time()
    hist = cnn.fit(X_train_cnn, y_train, epochs=1, batch_size=64, verbose=1)
    evaluate_model(cnn, X_test_cnn, y_test, "CNN", dataset_name, hist, time.time() - start, X_train, feature_names, lambda x: np.expand_dims(x, axis=2))

    # === LSTM ===
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, input_dim))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, input_dim))
    lstm = Sequential([
        LSTM(64, input_shape=(1, input_dim)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    lstm.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    start = time.time()
    hist = lstm.fit(X_train_lstm, y_train, epochs=1, batch_size=64, verbose=1)
    evaluate_model(lstm, X_test_lstm, y_test, "LSTM", dataset_name, hist, time.time() - start, X_train, feature_names, lambda x: x.reshape((x.shape[0], 1, input_dim)))

    # === Autoencoder + Dense ===
    with tf.device('/CPU:0'):
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(32, activation='relu')(input_layer)
        encoded = Dense(16, activation='relu')(encoded)
        decoded = Dense(32, activation='relu')(encoded)
        decoded = Dense(input_dim, activation='linear')(decoded)
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=Adam(), loss='mse')
        autoencoder.fit(X_train, X_train, epochs=1, batch_size=32, verbose=1)
        encoded_train = autoencoder.predict(X_train)
        encoded_test = autoencoder.predict(X_test)

    ae_model = Sequential([
        Dense(32, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    ae_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    start = time.time()
    hist = ae_model.fit(encoded_train, y_train, epochs=1, batch_size=32, verbose=1)
    evaluate_model(ae_model, encoded_test, y_test, "Autoencoder+Dense", dataset_name, hist, time.time() - start, encoded_train, feature_names)

    # === MLP ===
    mlp = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    mlp.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    start = time.time()
    hist = mlp.fit(X_train, y_train, epochs=1, batch_size=64, verbose=1)
    evaluate_model(mlp, X_test, y_test, "MLP", dataset_name, hist, time.time() - start, X_train, feature_names)

# === Sonuçları Kaydet ===
pd.DataFrame(results).to_csv(output_csv, index=False)
pd.DataFrame(training_logs).to_csv(log_csv, index=False)
print(f"\nToplam sonuçlar kaydedildi: {output_csv}")
print(f"Eğitim logları kaydedildi: {log_csv}")
