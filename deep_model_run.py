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

# === GPU ayarı ===
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
gpus = tf.config.list_physical_devices('GPU')
print(f"{'✅ GPU kullanılabilir: ' + gpus[0].name if gpus else '❌ GPU kullanılabilir değil!'}")

# === Yol ayarları ===
data_root = r"C:\\Users\\tugru\\Desktop\\Machine Learning Datasets"
datasets = {
    "CSE": os.path.join(data_root, "CSE-CIC-IDS2018", "Cleaned"),
    "BETH": os.path.join(data_root, "BETH", "Cleaned")
}
output_dir = os.path.join(data_root, "Result")
os.makedirs(output_dir, exist_ok=True)

output_csv = os.path.join(output_dir, "deep_results.csv")
log_csv = os.path.join(output_dir, "deep_training_log.csv")

results = []
training_logs = []

def evaluate_model(model, X_test, y_test, name, dataset, history, duration):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred_prob)

    print(f"✅ {dataset} - {name} → Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {roc:.4f}")

    results.append({
        "Dataset": dataset,
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC_AUC": roc,
        "Duration_sec": round(duration, 2)
    })

    if history:
        for i in range(len(history.history['loss'])):
            training_logs.append({
                "Dataset": dataset,
                "Model": name,
                "Epoch": i + 1,
                "Loss": history.history['loss'][i],
                "Accuracy": history.history['accuracy'][i]
            })

    del model, y_pred, y_pred_prob
    gc.collect()
    tf.keras.backend.clear_session()

for dataset_name, path in datasets.items():
    print(f"\n📂 Veri seti: {dataset_name}")
    X_train = pd.read_csv(os.path.join(path, "X_train.csv")).astype(np.float32)
    y_train = pd.read_csv(os.path.join(path, "y_train.csv")).squeeze().astype(np.float32)
    X_test = pd.read_csv(os.path.join(path, "X_test.csv")).astype(np.float32)
    y_test = pd.read_csv(os.path.join(path, "y_test.csv")).squeeze().astype(np.float32)

    for df in [X_train, X_test]:
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    input_dim = X_train.shape[1]

    # === MLP ===
    try:
        mlp = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        mlp.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        start = time.time()
        hist = mlp.fit(X_train, y_train, epochs=5, batch_size=64, verbose=1)
        evaluate_model(mlp, X_test, y_test, "MLP", dataset_name, hist, time.time() - start)
    except Exception as e:
        print(f"❌ MLP hatası: {e}")

    # === CNN ===
    try:
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
        hist = cnn.fit(X_train_cnn, y_train, epochs=5, batch_size=64, verbose=1)
        evaluate_model(cnn, X_test_cnn, y_test, "CNN", dataset_name, hist, time.time() - start)
    except Exception as e:
        print(f"❌ CNN hatası: {e}")

    # === LSTM ===
    try:
        X_train_lstm = X_train.reshape((X_train.shape[0], 1, input_dim))
        X_test_lstm = X_test.reshape((X_test.shape[0], 1, input_dim))
        lstm = Sequential([
            LSTM(64, input_shape=(1, input_dim)),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        lstm.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        start = time.time()
        hist = lstm.fit(X_train_lstm, y_train, epochs=5, batch_size=64, verbose=1)
        evaluate_model(lstm, X_test_lstm, y_test, "LSTM", dataset_name, hist, time.time() - start)
    except Exception as e:
        print(f"❌ LSTM hatası: {e}")

    # === AutoEncoder + Dense ===
    try:
        with tf.device('/CPU:0'):
            input_layer = Input(shape=(input_dim,))
            encoded = Dense(32, activation='relu')(input_layer)
            encoded = Dense(16, activation='relu')(encoded)
            decoded = Dense(32, activation='relu')(encoded)
            decoded = Dense(input_dim, activation='linear')(decoded)
            autoencoder = Model(inputs=input_layer, outputs=decoded)
            autoencoder.compile(optimizer=Adam(), loss='mse')
            autoencoder.fit(X_train, X_train, epochs=5, batch_size=32, verbose=1)

            encoded_train = autoencoder.predict(X_train)
            encoded_test = autoencoder.predict(X_test)

        ae_model = Sequential([
            Dense(32, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])
        ae_model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
        start = time.time()
        hist = ae_model.fit(encoded_train, y_train, epochs=5, batch_size=32, verbose=1)
        evaluate_model(ae_model, encoded_test, y_test, "Autoencoder+Dense", dataset_name, hist, time.time() - start)
    except Exception as e:
        print(f"❌ Autoencoder hatası: {e}")

# === CSV çıktısı
pd.DataFrame(results).to_csv(output_csv, index=False)
pd.DataFrame(training_logs).to_csv(log_csv, index=False)
print(f"\n📁 Toplam sonuçlar: {output_csv}")
print(f"📁 Eğitim logları: {log_csv}")
