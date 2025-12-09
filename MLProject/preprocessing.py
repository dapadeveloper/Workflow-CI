import pandas as pd
import os

# **********************************************
# PERBAIKAN PATH: 
# Sesuaikan path agar relatif terhadap direktori MLProject/
# **********************************************

# Path dataset MENTAH (asumsi file csv ada di dalam folder 'namadataset_preprocessing')
RAW_DATA_DIR = "namadataset_preprocessing"
RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "banknote_authentication.csv")

# Path output (file hasil preprocessing)
OUTPUT_PATH = os.path.join(RAW_DATA_DIR, "banknote_preprocessed.csv")


def load_data():
    # **********************************************
    # FIX: Pastikan folder output/data mentah ada
    # **********************************************
    if not os.path.exists(RAW_DATA_DIR):
        # Membuat folder jika belum ada (berguna jika skrip membuat folder ini)
        os.makedirs(RAW_DATA_DIR)
        print(f"Directory created: {RAW_DATA_DIR}")

    if not os.path.exists(RAW_DATA_PATH):
        # Seharusnya file mentah sudah ada di repositori
        raise FileNotFoundError(f"Dataset mentah tidak ditemukan: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Dataset loaded: {RAW_DATA_PATH}")
    print(df.head())
    return df

def preprocessing(df):
    df = df.drop_duplicates()

    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    # Pastikan file disimpan ke path yang benar
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessing completed. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    df = load_data()
    preprocessing(df)