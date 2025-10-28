# onf_to_jsonlines_balanced.py
# Converte OntoNotes .onf -> JSONLines (doc_key + sentences). Sem clusters.
# Não usa o pacote "on". Funciona em Python 3 (erros anteriores!!!)

import json
import re
from pathlib import Path
from typing import List, Tuple

# Padrão para detectar algo que "parece" uma árvore PTB:
# exige pelo menos um nó com etiqueta iniciando por letra maiúscula/dólar
HAS_PTB_NODE = re.compile(r"\([A-Z$][A-Za-z0-9$-]*\s")

# Extrai folhas (tokens) de uma árvore PTB simples: (TAG token)
LEAF_TOKEN = re.compile(r"\([^\s()]+\s+([^\s()]+)\)")

def extract_trees(text: str) -> List[str]:
    """
    Percorre o arquivo todo e retorna strings de subárvores com parênteses balanceados
    que "parecem" árvores PTB (validadas por HAS_PTB_NODE).
    """
    trees = []
    buf = []
    bal = 0
    for ch in text:
        if ch == '(':
            bal += 1
        if bal > 0:
            buf.append(ch)
        if ch == ')':
            bal -= 1
            if bal == 0 and buf:
                chunk = ''.join(buf)
                buf.clear()
                # heurística mínima: tem nó PTB?
                if HAS_PTB_NODE.search(chunk):
                    trees.append(chunk)
    return trees

def tokens_from_tree(tree_str: str) -> List[str]:
    """Extrai lista de tokens (folhas) de uma árvore PTB com padrão (TAG token)."""
    toks = []
    for m in LEAF_TOKEN.finditer(tree_str):
        tok = m.group(1)
        toks.append(tok)
    return toks

def parse_onf_file(path: Path) -> Tuple[str, List[List[str]]]:
    """
    Lê um .onf, encontra árvores PTB e devolve (doc_key, [sent1, sent2, ...]),
    onde cada sentença é lista de tokens.
    """
    text = path.read_text(encoding="utf-8", errors="ignore")
    trees = extract_trees(text)
    sentences = []
    for t in trees:
        toks = tokens_from_tree(t)
        # descarta "sentenças" com 0-1 token (ruído)
        if len(toks) >= 2:
            sentences.append(toks)
    doc_key = path.stem
    return doc_key, sentences

def convert(annotations_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ontonotes_all.onf_balanced.jsonlines"

    n_docs = 0
    with out_path.open("w", encoding="utf-8") as out:
        for onf in annotations_dir.rglob("*.onf"):
            doc_key, sentences = parse_onf_file(onf)
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
    ap.add_argument("--output_dir", required=True, help="Pasta de saída")
    args = ap.parse_args()
    convert(Path(args.annotations_dir), Path(args.output_dir))



