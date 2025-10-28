# onf_to_jsonlines_from_on.py
# Lê OntoNotes .onf usando o pacote "on" (do ontonotes-db-tool) e gera JSONLines com sentenças.
import sys, json
from pathlib import Path

def add_tool_to_path(db_tool_src: Path):
    # db_tool_src deve ser ...\ontonotes-db-tool-v0.999b\src
    if db_tool_src.is_dir():
        sys.path.insert(0, str(db_tool_src))
    else:
        raise FileNotFoundError(f"Pasta não encontrada: {db_tool_src}")

def extract_sentences_from_onf(onf_path: Path):
    """
    Abre um .onf e usa o parser de árvore do pacote 'on' para extrair tokens por sentença.
    Retorna: (doc_key, [ [tok1, tok2, ...], ... ])
    """
    # imports tardios (já com sys.path modificado):
    from on.corpora import tree as on_tree

    text = onf_path.read_text(encoding="utf-8", errors="ignore")
    # O parser do tool normalmente espera árvores marcadas; usa um split por linhas
    # e pede para o 'on.corpora.tree' reconstruir árvores completas.
    # Estratégia: coletar todas as sub-árvores PTB presentes no arquivo e extrair folhas.
    sentences = []
    # o módulo tem utilitários que aceitam strings com uma árvore por vez; então
    # divide o arquivo em sequências de parênteses balanceados.
    buf, balance = [], 0
    for ch in text:
        if ch == '(':
            balance += 1
        if balance > 0:
            buf.append(ch)
        if ch == ')':
            balance -= 1
            if balance == 0 and buf:
                tree_str = ''.join(buf)
                buf = []
                try:
                    t = on_tree.Tree.from_string(tree_str)
                    # extrai folhas (tokens)
                    tokens = [leaf.word for leaf in t.leaves() if getattr(leaf, "word", None)]
                    if tokens:
                        sentences.append(tokens)
                except Exception:
                    # se não conseguir parsear esse trecho, ignora e segue
                    pass

    doc_key = onf_path.stem
    return doc_key, sentences

def convert(annotations_dir: Path, db_tool_src: Path, output_dir: Path):
    add_tool_to_path(db_tool_src)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / "ontonotes_all.from_on.jsonlines"
    n_docs = 0

    with out_path.open("w", encoding="utf-8") as out:
        for onf_path in annotations_dir.rglob("*.onf"):
            doc_key, sentences = extract_sentences_from_onf(onf_path)
            if sentences:
                obj = {
                    "doc_key": doc_key,
                    "sentences": sentences,
                    "speakers": [["-"] * len(s) for s in sentences],
                    "clusters": []
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_docs += 1

    print(f"[OK] {n_docs} docs → {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations_dir", required=True,
                    help="Pasta ...\\data\\files\\data\\english\\annotations (com .onf)")
    ap.add_argument("--db_tool_src", required=True,
                    help="Pasta ...\\ontonotes-db-tool-v0.999b\\src (contém o pacote 'on')")
    ap.add_argument("--output_dir", required=True, help="Pasta de saída")
    args = ap.parse_args()

    convert(
        annotations_dir=Path(args.annotations_dir),
        db_tool_src=Path(args.db_tool_src),
        output_dir=Path(args.output_dir),
    )


