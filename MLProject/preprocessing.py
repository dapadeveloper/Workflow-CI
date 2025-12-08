import pandas as pd
import os

# Path dataset (fix sesuai nama file kamu)
RAW_DATA_PATH = "namadataset_preprocessing/banknote_authentication.csv"

# Path output
OUTPUT_PATH = "namadataset_preprocessing/banknote_preprocessed.csv"

def load_data():
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    print(f"Dataset loaded: {RAW_DATA_PATH}")
    print(df.head())
    return df

def preprocessing(df):
    df = df.drop_duplicates()

    if df.isnull().sum().sum() > 0:
        df = df.dropna()

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Preprocessing completed. Saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    df = load_data()
    preprocessing(df)