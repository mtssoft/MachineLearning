import os
import pandas as pd

# === 1. CSE-CIC-IDS2018 (RAM dostu) ===
cse_dir = r"C:\Users\tugru\Desktop\Machine Learning Datasets\CSE-CIC-IDS2018"
output_cse = os.path.join(cse_dir, "CSE_CIC_IDS2018_Merged.csv")
chunk_size = 50000
header_written = False

with open(output_cse, "w", encoding="utf-8") as f_out:
    for file in os.listdir(cse_dir):
        if not file.endswith(".csv") or "Merged" in file:
            continue  # Kendi çıktı dosyamızı atla
        if file.endswith(".csv"):
            file_path = os.path.join(cse_dir, file)
            print(f"➡️ CSE İşleniyor: {file}")
            for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
                chunk["source_file"] = file
                chunk.to_csv(f_out, index=False, header=not header_written, mode="a")
                header_written = True

print(f"\n CSE-CIC-IDS2018 birleştirildi ve şuraya kaydedildi: {output_cse}")

# === 2. BETH (RAM küçük, doğrudan yüklenebilir) ===
beth_dir = r"C:\Users\tugru\Desktop\Machine Learning Datasets\BETH"
beth_files = [os.path.join(beth_dir, f) for f in os.listdir(beth_dir) if f.endswith(".csv")]

beth_dfs = []
for file in beth_files:
    df = pd.read_csv(file, low_memory=False)
    df["source_file"] = os.path.basename(file)
    beth_dfs.append(df)

df_beth = pd.concat(beth_dfs, ignore_index=True)
print(f"\n BETH veri seti birleştirildi. Toplam kayıt: {len(df_beth)}")

# Kaydet
output_beth = os.path.join(beth_dir, "BETH_Merged.csv")
df_beth.to_csv(output_beth, index=False)
print(f" BETH birleştirildi ve şuraya kaydedildi: {output_beth}")
