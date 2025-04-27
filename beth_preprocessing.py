import pandas as pd
import os
from sklearn.model_selection import train_test_split

# === Ayarlar ===
beth_dir = r"C:\\Users\\tugru\\Desktop\\Machine Learning Datasets\\BETH"
beth_files = [os.path.join(beth_dir, f) for f in os.listdir(beth_dir) if f.endswith(".csv")]
chunk_size = 50000
output_dir = os.path.join(beth_dir, "Cleaned")
os.makedirs(output_dir, exist_ok=True)
cleaned_path = os.path.join(output_dir, "Cleaned_BETH_Full.csv")

# Eski dosya varsa sil
if os.path.exists(cleaned_path):
    os.remove(cleaned_path)

header_written = False

# === Her dosyayı chunk bazlı işle ===
for file in beth_files:
    for chunk in pd.read_csv(file, chunksize=chunk_size, sep=',', engine='python', on_bad_lines='warn'):
        if chunk.empty:
            print("⚠️ Boş chunk atlandı")
            continue

        if 'evil' not in chunk.columns:
            print("⚠️ 'evil' sütunu bulunamadı, chunk atlandı")
            continue

        # Label olarak sadece evil sütununu kullan
        chunk['Label'] = chunk['evil'].astype(int)

        # Eğitimde kullanılmaması gereken sütunları çıkar
        drop_cols = ['timestamp', 'processName', 'eventName', 'args', 'hostName',
                     'stackAddresses', 'source_file', 'sus', 'evil']
        chunk = chunk.drop(columns=[col for col in drop_cols if col in chunk.columns], errors='ignore')

        # Sadece sayısal sütunları al
        numeric_cols = chunk.select_dtypes(include=['number']).columns.tolist()
        if 'Label' in numeric_cols:
            numeric_cols.remove('Label')
        if len(numeric_cols) == 0:
            print("⚠️ Sayısal sütun bulunamadı, chunk atlandı")
            continue

        chunk = chunk[numeric_cols + ['Label']]
        chunk = chunk.dropna()

        if not chunk.empty:
            chunk.to_csv(cleaned_path, mode='a', index=False, header=not header_written)
            header_written = True
            print(f"✅ Chunk işlendi ve yazıldı: {len(chunk)} satır")
        else:
            print("⚠️ dropna sonrası chunk boş, atlandı")

# === Tüm veri temizlendi, oku ve eğitim/test verisine böl ===
df_clean = pd.read_csv(cleaned_path)
print(f"\n✅ BETH tam veri yüklendi. Toplam kayıt: {len(df_clean)}")

X = df_clean.drop(columns=['Label'])
y = df_clean['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

print("\n💾 BETH eğitim/test verileri başarıyla diske kaydedildi.")
print(f"🧪 Eğitim kümesi boyutu: {len(X_train)}")
print(f"🧪 Test kümesi boyutu: {len(X_test)}")
