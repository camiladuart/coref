from coref import CorefDataset

DATA_DIR = r"C:\Users\PC\coref_data\ontonotes"

ds = CorefDataset(DATA_DIR, split="dev")
print("n_docs:", len(ds))
print("primeiro doc_key:", ds[0]["doc_key"])
print("primeira sentença:", ds[0]["sentences"][0][:20])
