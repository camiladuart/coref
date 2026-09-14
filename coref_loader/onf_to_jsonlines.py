# onf_to_jsonlines.py
# converte OntoNotes .onf -> JSONLines (doc_key + sentences)
#“PTB” = Penn Treebank: Árvore PTB = representação em parênteses (notação S-expr) da árvore de constituintes de uma sentence
    #porque os arquivos .onf do OntoNotes contêm anotações ricas -> aproveitar as folhas das árvores é uma forma confiável de recuperar a tokenização do corpus.
    #extrai sentenças sem ter que implementar um parser do formato ONF

import json
import re #regex -> reconhecer padroes de texto - arvores
from pathlib import Path
from typing import List, Tuple

#detecta inicio da arvore PTB "($":
HAS_PTB_NODE = re.compile(r"\([A-Z$][A-Za-z0-9$-]*\s")

#extrai folhas (tokens):
LEAF_TOKEN = re.compile(r"\([^\s()]+\s+([^\s()]+)\)")

def extract_trees(text: str) -> List[str]: #lê o texto inteiro e retorna strings de subárvores com parênteses balanceadosque parecem árvores PTB (validadas por HAS_PTB_NODE).
    trees = [] #lista final de subárvores
    buf = [] #bucket de caracteres para ir montando a subárvore atual
    bal = 0 #contador de balanceamento de parênteses: ( soma 1, ) subtrai 1. 0 → fechou uma subárvore completa.
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
                #heurística mínima: tem nó PTB?
                if HAS_PTB_NODE.search(chunk): #se a subárvore passar na heurística HAS_PTB_NODE (tem cara de árvore PTB), guarda em trees
                    trees.append(chunk)
    return trees

def tokens_from_tree(tree_str: str) -> List[str]: #extrai lista de tokens de uma árvore PTB com padrão (TAG token)
    toks = []
    for m in LEAF_TOKEN.finditer(tree_str):
        tok = m.group(1)
        toks.append(tok)
    return toks #sentence tokenizada

def parse_onf_file(path: Path) -> Tuple[str, List[List[str]]]:
    """
    Lê um .onf, encontra árvores PTB e devolve (doc_key, [sent1, sent2, ...]),
    """
    text = path.read_text(encoding="utf-8", errors="ignore") #le arq onf como texto, ignorando erros de encoding
    trees = extract_trees(text) #extrai subarvores
    sentences = []
    for t in trees: #para cada subarvore, extrai os tokens:
        toks = tokens_from_tree(t)
        # descarta sentences com 0-1 tokens (mt curto -> ruído)
        if len(toks) >= 2:
            sentences.append(toks) #guarda como sentence
    doc_key = path.stem
    return doc_key, sentences

#função principal de conversao:
def convert(annotations_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True) #garantir que a pasta de saída exista
    out_path = output_dir / "ontonotes_all.onf_balanced.jsonlines" #define arq de saida (jsonlines) com este nome fixo

    n_docs = 0
    with out_path.open("w", encoding="utf-8") as out:
        for onf in annotations_dir.rglob("*.onf"):
            doc_key, sentences = parse_onf_file(onf) #ara cada .onf, gera doc_key e as sentences
            if sentences: #se tiver pelo menos uma sentence, monta objeto no formato que criei no loader:
                obj = {
                    "doc_key": doc_key,
                    "sentences": sentences,
                    "speakers": [["-"] * len(s) for s in sentences],
                    "clusters": []
                }
                out.write(json.dumps(obj, ensure_ascii=False) + "\n") #escreve uma linha json
                n_docs += 1

    #ensagem final com quantos docs foram exportados e o caminho do arquivo de saída
    print(f"[OK] {n_docs} docs → {out_path}")

if __name__ == "__main__":
    import argparse #usando argparse para ler: --annotations_dir, onde estão os .onf; --output_dir,  onde salvar o .jsonlines
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations_dir", required=True,
                    help="Pasta ...\\data\\files\\data\\english\\annotations (com .onf)")
    ap.add_argument("--output_dir", required=True, help="Pasta de saída")
    args = ap.parse_args()
    convert(Path(args.annotations_dir), Path(args.output_dir))
