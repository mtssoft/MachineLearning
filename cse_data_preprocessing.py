import pandas as pd
import os
from sklearn.model_selection import train_test_split

# === Ayarlar ===
input_path = r"C:\\Users\\tugru\\Desktop\\Machine Learning Datasets\\CSE-CIC-IDS2018\\CSE_CIC_IDS2018_Merged.csv"
chunk_size = 100000  # RAM dostu okuma
output_dir = r"C:\\Users\\tugru\\Desktop\\Machine Learning Datasets\\CSE-CIC-IDS2018\\Cleaned"
os.makedirs(output_dir, exist_ok=True)
temp_cleaned_csv = os.path.join(output_dir, "Cleaned_CSE_Full.csv")

# Eğer varsa eski dosyayı sil
if os.path.exists(temp_cleaned_csv):
    os.remove(temp_cleaned_csv)

header_written = False

# === Chunk chunk temizle ve doğrudan diske yaz ===
for chunk in pd.read_csv(input_path, chunksize=chunk_size, sep=',', engine='python', on_bad_lines='warn'):
    if chunk.empty:
        print("⚠️ Boş chunk atlandı")
        continue

    if 'Label' not in chunk.columns:
        print("⚠️ 'Label' sütunu bulunamadı, chunk atlandı")
        continue

    # Label'ı binary hale getir
    chunk['Label'] = chunk['Label'].apply(lambda x: 0 if str(x).strip().lower() == 'benign' else 1)

    # Gereksiz sütunları çıkar
    cols_to_drop = ['Timestamp', 'source_file'] if 'source_file' in chunk.columns else ['Timestamp']
    chunk = chunk.drop(columns=[col for col in cols_to_drop if col in chunk.columns], errors='ignore')

    # Sadece sayısal sütunları al (Label hariç)
    numeric_cols = chunk.select_dtypes(include=['number']).columns.tolist()
    if 'Label' in numeric_cols:
        numeric_cols.remove('Label')

    if len(numeric_cols) == 0:
        print("⚠️ Sayısal sütun bulunamadı, chunk atlandı")
        continue

    chunk = chunk[numeric_cols + ['Label']]

    # Eksik verileri at
    chunk = chunk.dropna()

    if not chunk.empty:
        chunk.to_csv(temp_cleaned_csv, mode='a', index=False, header=not header_written)
        header_written = True
        print(f"✅ Chunk diske yazıldı: {len(chunk)} satır")
    else:
        print("⚠️ Tüm satırlar dropna ile silindi, chunk atlandı")

# === Tüm veriyi oku ve eğitim/test olarak ayır ===
df_clean = pd.read_csv(temp_cleaned_csv)
print(f"\n✅ Tüm veri diskten yüklendi. Toplam kayıt: {len(df_clean)}")

X = df_clean.drop(columns=['Label'])
y = df_clean['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Eğitim/test verilerini kaydet
X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

print("\n💾 Tüm eğitim/test verileri başarıyla diske kaydedildi.")
print(f"\n🧪 Eğitim kümesi boyutu: {len(X_train)}")
print(f"🧪 Test kümesi boyutu: {len(X_test)}")
