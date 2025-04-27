import os
import pandas as pd

# === CSE-CIC-IDS2018 (ilk 100k satır) ===
df_cse = pd.read_csv(
    r"C:\Users\tugru\Desktop\Machine Learning Datasets\CSE-CIC-IDS2018\CSE_CIC_IDS2018_Merged.csv", 
    low_memory=False, 
    on_bad_lines='skip',
    nrows=100000
)
print("\n✅ CSE Sütunları:")
print(df_cse.columns)
print("\n📊 CSE Label Sınıf Dağılımı:")
print(df_cse['Label'].value_counts())

# === BETH (ilk 100k satır) ===
df_beth = pd.read_csv(
    r"C:\Users\tugru\Desktop\Machine Learning Datasets\BETH\BETH_Merged.csv", 
    low_memory=False, 
    on_bad_lines='skip',
    nrows=100000
)
print("\n✅ BETH Sütunları:")
print(df_beth.columns)

# === Tahmini hedef sütunları deniyoruz ===
print("\n📊 BETH 'sus' sütunu değerleri:")
print(df_beth['sus'].value_counts(dropna=False))

print("\n📊 BETH 'evil' sütunu değerleri:")
print(df_beth['evil'].value_counts(dropna=False))

# Bonus: Her iki kombinasyonu göster
print("\n🧪 'sus' + 'evil' kombinasyonları:")
print(df_beth[['sus', 'evil']].drop_duplicates())
