# onf_to_jsonlines.py
# Converte OntoNotes .onf -> JSONLines com "doc_key", "sentences" (sem clusters).
# Útil p/ imprimir cada instância já com sentenças/tokenização básica.

import os, json, re
from pathlib import Path
from typing import List

PTB_LINE = re.compile(r'^\(\s*[A-Z$\-,:`\'"]')  # linha que parece uma árvore PTB

def _extract_tokens_from_ptb_line(line: str) -> List[str]:
    # Extrai palavras folhas de uma árvore PTB (ex.: (DT The) -> "The").
    # Simples/robusto o bastante para começar.
    tokens = []
    # procura padrões como "(TAG token)" e captura o "token"
    for m in re.finditer(r'\([^\s()]+\s+([^\s()]+)\)', line):
        tok = m.group(1)
        tokens.append(tok)
    return tokens

def parse_onf_file(path: Path):
    docs = []
    doc_key = path.stem  # usa nome do arquivo sem extensão como doc_key
    sentences = []
    current_sent = []

    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for raw in f:
            line = raw.rstrip('\n')
            if PTB_LINE.match(line):
                # Concatena linhas de uma mesma árvore até fechar todos parênteses
                tree = line
                open_parens = line.count('(') - line.count(')')
                while open_parens > 0:
                    nxt = f.readline()
                    if not nxt:
                        break
                    tree += nxt
                    open_parens += nxt.count('(') - nxt.count(')')
                # agora tree deve conter uma árvore completa -> extrai tokens
                toks = _extract_tokens_from_ptb_line(tree)
                if toks:
                    sentences.append(toks)

    if sentences:
        docs.append({
            "doc_key": str(doc_key),
            "sentences": sentences,
            "speakers": [["-"] * len(s) for s in sentences],
            "clusters": []  # sem coref neste caminho
        })
    return docs

def find_onf_files(root: Path):
    return list(root.rglob('*.onf'))

def convert_onf(input_dir: str, output_dir: str):
    in_root = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    docs_bucket = []
    for p in find_onf_files(in_root):
        print(f"Lendo {p}")
        docs_bucket.extend(parse_onf_file(p))

    # Sem splits oficiais aqui (train/dev/test). Salva tudo em um só arquivo:
    out_path = out_root / "ontonotes_all.onf.jsonlines"
    with out_path.open('w', encoding='utf-8') as f:
        for d in docs_bucket:
            f.write(json.dumps(d, ensure_ascii=False) + '\n')
    print(f"[OK] {len(docs_bucket)} docs → {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Pasta raiz do OntoNotes (onde há data/english/annotations/.../*.onf)")
    ap.add_argument("--output_dir", required=True, help="Pasta de saída")
    args = ap.parse_args()
    convert_onf(args.input_dir, args.output_dir)
