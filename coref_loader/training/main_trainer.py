#treina e imprime instancias

from pathlib import Path
from coref import CorefDataset

def main():
    # ajuste o caminho para onde você salvou os jsonlines
    processed_dir = r"C:\Users\PC\coref_data\ontonotes"
    split = "dev"  # ou "train"/"test"

    ds = CorefDataset(processed_dir, split)
    print(f"[OK] carregado: {len(ds)} documentos ({split})\n")

    # imprime cada instância (doc)
    for i, doc in enumerate(ds.samples):
        doc_key = doc["doc_key"]
        n_sent  = len(doc["sentences"])
        n_toks  = sum(len(s) for s in doc["sentences"])
        n_clust = len(doc.get("clusters", []))
        print(f"[{i:05d}] {doc_key} | sentenças: {n_sent} | tokens: {n_toks} | clusters: {n_clust}")

if __name__ == "__main__":
    main()
