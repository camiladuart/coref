"""
converte ontonotes para jsonlines (1 documento por linha). saída por doc:
{
  "doc_key": "...",
  "sentences": [["Token", "..."], ["Outra", "frase", "."]],
  "speakers":  [["-","-"], ["-","-","-"]], 
  "clusters":  [[[i1, j1], [i2, j2]], ...]  
}
"""

import os
import json
from pathlib import Path
from typing import List, Tuple, Dict

# definir inicio e fim do doc "como aparece no original"
BEGIN = "#begin document"
END   = "#end document"

def _flush_sentence(tokens: List[str], sentences: List[List[str]]): # para fechar a frase atual
    if tokens:
        sentences.append(tokens[:])  # copia os tokens atuais para a lista de frases
        tokens.clear()               # limpa a lista de tokens

def _map_global_to_sent_positions(sentences: List[List[str]]): # ontonotes conta do documento (global) -> criar mapa pra saber qual sentence e qual posição dentro dela
    mapping = {} #dicionario vazio pra guardar os resultados
    g = 0  #contador de indices globais
    for s_idx, sent in enumerate(sentences): #s_idx = numero da frase; sent = lista de palavras daquela frase
        for t_idx, _ in enumerate(sent): #t_idx = indice de cada palavra dentro da frase; _ pq nao vou usar o valor
            mapping[g] = (s_idx, t_idx) #o token global número g está na frase s_idx, na posição t_idx
            g += 1
    return mapping

def _to_sentence_level_clusters(sentences: List[List[str]],
                                clusters_global: Dict[str, List[Tuple[int,int]]]): #converter de indices globais para indices por sentença
    mapping = _map_global_to_sent_positions(sentences)
    out = []
    for chain_id, spans in clusters_global.items():
        group = []
        for g_start, g_end in spans:
            s_start, t_start = mapping[g_start]
            s_end,   t_end   = mapping[g_end]
            if s_start != s_end:
                # se começar numa frase e terminar em outra, ignorar (é mais simples)
                continue
            group.append([t_start, t_end])  # span dentro da mesma frase
        if group:
            out.append(group)
    return out

def parse_conll_file(path: Path): #ler um arquivo e devolve uma lista de documentos (dicionários)
    docs = []

    with path.open(encoding="utf-8") as f: #começar com listas vazias
        doc_key = None
        sentences: List[List[str]] = []   # lista de frases
        tokens: List[str] = []            # tokens da frase atual
        token_index_global = 0            # posição global do token no documento

        # para montar os clusters de coref
        open_spans: Dict[str, List[int]] = {}       # id_da_cadeia -> quando acha (, abre; quando acha ), fecha e cria um span
        clusters_tmp: Dict[str, List[Tuple[int,int]]] = {}  # id_da_cadeia -> lista de spans (ini, fim) globais

        for raw in f:
            line = raw.rstrip("\n")

            # começo de documento
            if line.startswith(BEGIN):
                #reseta tudo para começar um novo doc
                left = line.find("(")
                right = line.find(")")
                doc_key = line[left+1:right] if left != -1 and right != -1 else "UNKNOWN"
                sentences, tokens = [], []
                token_index_global = 0
                open_spans.clear()
                clusters_tmp.clear()
                continue

            # fim do doc
            if line.startswith(END):
                _flush_sentence(tokens, sentences)  # fecha a última frase
                if doc_key and sentences: #monta o dicionário final com doc_key, sentences, speakers (“-”), clusters ->guarda em docs
                    doc = {
                        "doc_key": doc_key,
                        "sentences": sentences,
                        # speakers não é usado; preenche com "-" para cada token
                        "speakers": [["-"] * len(s) for s in sentences],
                        # clusters convertidos para índices por sentença
                        "clusters": _to_sentence_level_clusters(sentences, clusters_tmp),
                    }
                    docs.append(doc) 
                # limpa variáveis para o próximo documento
                doc_key = None
                sentences, tokens = [], []
                token_index_global = 0
                open_spans.clear()
                clusters_tmp.clear()
                continue

            # linha em branco = fim de uma sentença
            if line == "":
                _flush_sentence(tokens, sentences)
                continue

            # linha normal tem varias columas: 4ª coluna (parts[3]) é o token
            parts = line.split()
            if len(parts) < 4:
                # se a linha não tem colunas suficientes, pula
                continue

            token = parts[3]  
            coref = parts[-1] # a última coluna é a anotação de coreferência
            tokens.append(token)

            # tratamento da coluna de coreferência:
            if coref != "-":
                for chunk in coref.split("|"):
                    if chunk.startswith("(") and chunk.endswith(")"):
                        # menção de um token só: abre e fecha no mesmo token
                        chain = chunk[1:-1]
                        clusters_tmp.setdefault(chain, []).append((token_index_global, token_index_global))
                    elif chunk.startswith("("):
                        # encontrou o começo de uma menção; guarda o índice de início numa pilha
                        chain = chunk[1:]
                        open_spans.setdefault(chain, []).append(token_index_global)
                    elif chunk.endswith(")"):
                        # encontrou o fim de uma menção; pega o último início e forma o par (início, fim)
                        chain = chunk[:-1]
                        stack = open_spans.get(chain, [])
                        if stack:
                            start = stack.pop()
                            clusters_tmp.setdefault(chain, []).append((start, token_index_global))
                        # se não houver início correspondente, ignora

            token_index_global += 1  # add ao índice global de tokens

    return docs

def find_conll_files(root: Path): #anda pela pasta inteira, acha todos os arqs .v4_gold_conll e tenta adivinhar se o arquivo é de train, dev ou test pelo caminho
    found = []
    for p in root.rglob("*.v4_gold_conll"):
        low = str(p).lower()
        if "train" in low:
            split = "train"
        elif "development" in low or "dev" in low:
            split = "dev"
        elif "test" in low:
            split = "test"
        else:
            split = "unknown"
        found.append((split, p))
    return found

def convert_ontonotes(input_dir: str, output_dir: str): #converte todos os arquivos do input_dir e salva jsonlines no output_dir.
    root_in  = Path(input_dir)
    root_out = Path(output_dir)
    root_out.mkdir(parents=True, exist_ok=True)
# para cada arquivo encontrado, lê e acumula os documentos dentro do “balde” certo (train/dev/test).
    buckets: Dict[str, List[dict]] = {"train": [], "dev": [], "test": []} 

    # lê todos os arquivos .v4_gold_conll encontrados
    for split, path in find_conll_files(root_in):
        print(f"Lendo {path}  ({split})")
        docs = parse_conll_file(path)
        buckets.setdefault(split, []).extend(docs)

    # escreve um arquivo .jsonlines para cada split
    for split in ("train", "dev", "test"):
        docs = buckets.get(split, [])
        if not docs:
            continue
        out = root_out / f"{split}.english.jsonlines"
        with out.open("w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        print(f"[OK] {split}: {len(docs)} docs → {out}")

if __name__ == "__main__":
    # para poder rodar pelo terminal
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Pasta raiz do OntoNotes (onde ficam os *.v4_gold_conll)")
    ap.add_argument("--output_dir", required=True, help="Pasta de saída para salvar *.jsonlines")
    args = ap.parse_args()
    convert_ontonotes(args.input_dir, args.output_dir)
