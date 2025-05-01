import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from ctgan import CTGAN
from sklearn.utils import shuffle

# === YOL AYARLARI ===
data_root = r"C:\Users\tugru\Desktop\Machine Learning Datasets"
datasets = {
    "CSE": os.path.join(data_root, "CSE-CIC-IDS2018", "Cleaned"),
    "BETH": os.path.join(data_root, "BETH", "Cleaned")
}
balanced_root = os.path.join(data_root, "Balanced")
os.makedirs(balanced_root, exist_ok=True)

# === HER VERİ SETİ İÇİN DENGELEME ===
for name, path in datasets.items():
    print(f"\n📁 Veri Seti: {name}")
    X = pd.read_csv(os.path.join(path, "X_train.csv"))
    y = pd.read_csv(os.path.join(path, "y_train.csv")).squeeze()

    # === Hatalı değerleri temizle ===
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    y = y.loc[X.notna().all(axis=1)]
    X = X.dropna().reset_index(drop=True)
    y = y.reset_index(drop=True)

    # Oranlar
    majority_count = (y == 0).sum()
    minority_count = (y == 1).sum()
    target = majority_count

    smote_target = minority_count + int((target - minority_count) / 2)
    n_ctgan_samples = target - smote_target

    print(f"🔧 Orijinal Azınlık: {minority_count} — SMOTE Hedefi: {smote_target} — CTGAN Hedefi: {n_ctgan_samples}")

    # === SMOTE ile yarıya kadar artır ===
    smote = SMOTE(sampling_strategy={1: smote_target}, random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    # === CTGAN için azınlık sınıfını hazırla ===
    df_smote = X_smote.copy()
    df_smote["Label"] = y_smote
    minority_df = df_smote[df_smote["Label"] == 1].copy().reset_index(drop=True)

    # CTGAN eğitimine yalnızca sayısal değerler verilmeli
    ctgan = CTGAN(epochs=50)  # daha az epoch ile daha hızlı
    ctgan.fit(minority_df)

    synth_df = ctgan.sample(n_ctgan_samples)
    synth_df["Label"] = 1

    # === Son veri kümesini birleştir ===
    df_majority = df_smote[df_smote["Label"] == 0]
    df_minority = df_smote[df_smote["Label"] == 1]
    final_df = pd.concat([df_majority, df_minority, synth_df], ignore_index=True)
    final_df = shuffle(final_df, random_state=42)

    # === Ayır ve yaz ===
    X_balanced = final_df.drop(columns=["Label"])
    y_balanced = final_df["Label"]

    out_dir = os.path.join(balanced_root, name)
    os.makedirs(out_dir, exist_ok=True)
    X_balanced.to_csv(os.path.join(out_dir, "X_train.csv"), index=False)
    y_balanced.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)

    print(f"✅ {name} için dengeli veri oluşturuldu ve şuraya kaydedildi: {out_dir}")
