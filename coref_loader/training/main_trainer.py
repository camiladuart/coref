#imprime instâncias do dataset
    #dificuldades: tentei usar o dataset original do OntoNotes, com o codigo ontonotes_to_jsonlines.py, mas ele vem em formato .onf e o ontonotes-db-tool oficial é em *Python 2*, então não consegui importar.
    #mudei de plano e passei a testar com arquivos .jsonlines criados manualmente.
    #setup inicial: tive muitos erros de ambiente (versões de python, tensorflow, etc) -> resolvi criando um ambiente limpo (python 3.7) e instalando manualmente o que precisava.
    #imports e estrutura: o python não encontrava o pacote coref_loader -> resolvi rodando o trainer como módulo (python -m coref_loader.training.main_trainer).
    # roda 100% em Python 3; consigo ler e imprimir instâncias do dataset no formato .jsonlines; o código está sem dependências antigas (tensorflow, pytorch, etc.)

import argparse
from pathlib import Path
from coref_loader.data import CorefDataset

def count_tokens(sentences):
    return sum(len(s) for s in sentences)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Pasta que contém *.english.jsonlines")
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"],
                    help="Qual arquivo abrir (train/dev/test)")
    ap.add_argument("--limit", type=int, default=20, help="Quantos docs imprimir (0 = todos)")
    ap.add_argument("--preview", type=int, default=12, help="Qtde de tokens para prévia da 1ª sentença")
    args = ap.parse_args()

    ds = CorefDataset(args.data_dir, args.split)
    print(f"[OK] carregado: {len(ds)} documentos ({args.split})\n")

    n = len(ds) if args.limit == 0 else min(args.limit, len(ds))
    for i in range(n):
        doc = ds[i]
        doc_key = doc["doc_key"]
        nsents = len(doc["sentences"])
        ntoks = count_tokens(doc["sentences"])
        nclus = len(doc.get("clusters", []))
        preview = " ".join(doc["sentences"][0][:args.preview]) if nsents else ""
        print(f"[{i:05d}] {doc_key} | sentenças: {nsents} | tokens_total: {ntoks} | clusters: {nclus}")
        if preview:
            print(f"       1ª sentença: {preview}")
    print("\n[done]")

if __name__ == "__main__":
    main()
