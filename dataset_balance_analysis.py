import pandas as pd

# === Dosya yolları ===
cse_y_train = r"C:\Users\tugru\Desktop\Machine Learning Datasets\CSE-CIC-IDS2018\Cleaned\y_train.csv"
beth_y_train = r"C:\Users\tugru\Desktop\Machine Learning Datasets\BETH\Cleaned\y_train.csv"

# === CSE veri seti etiket dağılımı ===
cse_labels = pd.read_csv(cse_y_train).squeeze()
print("📊 CSE y_train dağılımı:")
print(cse_labels.value_counts(normalize=True) * 100)

# === BETH veri seti etiket dağılımı ===
beth_labels = pd.read_csv(beth_y_train).squeeze()
print("\n📊 BETH y_train dağılımı:")
print(beth_labels.value_counts(normalize=True) * 100)
