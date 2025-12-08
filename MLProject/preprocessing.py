import pandas as pd

raw_path = "namadataset_preprocessing/banknote_raw.csv"
save_path = "namadataset_preprocessing/banknote_preprocessed.csv"

df = pd.read_csv(raw_path)
df = df.dropna()

df.to_csv(save_path, index=False)

print(" Preprocessing selesai. File disimpan di:", save_path)
