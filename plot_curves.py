import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# lista de (pasta_checkpoints, nome_lingua)
experimentos = [
    ("romance_mono_pt", "PT"),
    ("romance_mono_pt_mdb", "PT_mdb"),
    ("romance_mono_it", "IT"),
    ("romance_mono_it_mdb", "IT_mdb"),
    ("romance_mono_es", "ES"),
    ("romance_mono_es_mdb", "ES_mdb"),
    ("romance_mono_cat", "CAT"),
    ("romance_mono_cat_mdb", "CAT_mdb"),
    ("romance_mono_francor", "FRancor"),
    ("romance_mono_francor_mdb", "FRancor_mdb"),
    ("romance_mono_frdemocrat", "FRdemocrat"),
    ("romance_mono_frdemocrat_mdb", "FRdemocrat_mdb"),
]

base = Path("/projects/F202600026AIVLABDEUCALION/up202000683")

for pasta, lingua in experimentos:
    csv_path = base / pasta / "learning_curves.csv"
    if not csv_path.exists():
        print(f"[SKIP] {csv_path} não encontrado")
        continue

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Train Loss", marker="o")
    dev_df = df.dropna(subset=["dev_loss"])
    if len(dev_df) > 0:
        plt.plot(dev_df["epoch"], dev_df["dev_loss"], label="Dev Loss", marker="s")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Learning Curves — multilingue {lingua}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = base / f"learning_curve_{lingua.lower()}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[OK] Guardado: {out_path}")
