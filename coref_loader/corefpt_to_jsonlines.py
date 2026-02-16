#CorefPT (XML) -> JSONLines (doc_key + sentences + clusters)
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional

#helpers de normalização de spans
SPAN_RE = re.compile(r"(\d+)\s*(?:\.\.|-)\s*(\d+)")
INT_RE = re.compile(r"\d+")

#aceitar spans tipo "12..15", "12-15", "12 13 14 15", "w12..w15" e retornar lista de (start,end):
def parse_span_text(span_text: str) -> List[Tuple[int, int]]:
    if not span_text:
        return []
    t = span_text.strip()

    #caso 12..15 ou 12-15
    m = SPAN_RE.search(t)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        if a <= b:
            return [(a, b)]
        return [(b, a)]

    #caso lista de números (pega min/max)
    nums = [int(x) for x in INT_RE.findall(t)]
    if len(nums) >= 2:
        return [(min(nums), max(nums))]
    if len(nums) == 1:
        return [(nums[0], nums[0])]
    return []

def safe_text(x: Optional[str]) -> str:
    return (x or "").strip()

#extração de tokens/sentenças
def extract_sentences_tokens(root: ET.Element) -> Tuple[List[List[str]], Dict[str, int]]:
    sentences: List[List[str]] = []
    token_id_to_idx: Dict[str, int] = {}

    global_idx = 0
    auto_id = 1

    #tenta achar sentenças
    sent_nodes = root.findall(".//s")
    if not sent_nodes:
        sent_nodes = root.findall(".//sentence")
    #se não achar, trata documento como uma sentença
    if not sent_nodes:
        sent_nodes = [root]

    for s in sent_nodes:
        toks: List[str] = []
        #tokens: <w> ou <token>
        tok_nodes = s.findall(".//w")
        if not tok_nodes:
            tok_nodes = s.findall(".//token")

        for w in tok_nodes:
            tok = safe_text(w.text)
            if tok == "":
                #pq token pode estar guardado em atributo
                tok = safe_text(w.get("form") or w.get("tok") or w.get("word"))
            if tok == "":
                continue

            tid = w.get("id")
            if not tid:
                #garantia: caso vier como "w12"
                tid = w.get("xml:id") or f"t{auto_id}"
                auto_id += 1

            toks.append(tok)
            token_id_to_idx[tid] = global_idx
            global_idx += 1

        if toks:
            sentences.append(toks)

    return sentences, token_id_to_idx

#extração de clusters: (a partir de diferentes estruturas) e retornando clusters_map
def try_extract_mentions_from_chains(root: ET.Element, token_id_to_idx: Dict[str, int]) -> Dict[str, List[List[int]]]:
    clusters: Dict[str, List[List[int]]] = {}

    # candidatos de nós de "cadeia"
    chain_nodes = []
    for tag in ["entity", "chain", "corefChain", "coref_chain", "markable_set"]:
        chain_nodes.extend(root.findall(f".//{tag}"))

    def add_span(cid: str, start: int, end: int):
        clusters.setdefault(cid, []).append([start, end])

    for ch in chain_nodes:
        cid = ch.get("id") or ch.get("cid") or ch.get("entity") or ch.get("chain_id")
        if not cid:
            continue
        # menções
        mention_nodes = []
        for mtag in ["mention", "markable", "span", "m"]:
            mention_nodes.extend(ch.findall(f".//{mtag}"))

        for m in mention_nodes:
            # casos comuns: start/end, from/to
            start_attr = m.get("start") or m.get("from")
            end_attr = m.get("end") or m.get("to")

            if start_attr and end_attr:
                # podem ser ids de tokens ("w12") ou números
                if start_attr in token_id_to_idx:
                    start = token_id_to_idx[start_attr]
                else:
                    start = int(INT_RE.findall(start_attr)[-1])

                if end_attr in token_id_to_idx:
                    end = token_id_to_idx[end_attr]
                else:
                    end = int(INT_RE.findall(end_attr)[-1])

                if start <= end:
                    add_span(cid, start, end)
                else:
                    add_span(cid, end, start)
                continue

            # span em texto/atributo
            span_attr = m.get("span") or m.get("target") or m.get("tokens")
            if span_attr:
                spans = parse_span_text(span_attr)
                for a,b in spans:
                    add_span(cid, a, b)
                continue

            #se o span vier no texto do nó
            spans = parse_span_text(safe_text(m.text))
            for a,b in spans:
                add_span(cid, a, b)

    return clusters

#para converter 1 xml em 1 objeto no formato {doc_key, sentences, speakers, clusters}:
def convert_one_xml(xml_path: Path) -> Optional[dict]:
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"!!Falha parse XML {xml_path}: {e}")
        return None

    doc_key = xml_path.stem

    sentences, token_id_to_idx = extract_sentences_tokens(root)
    
    clusters_map = try_extract_mentions_from_chains(root, token_id_to_idx)

    # se não achou nada, devolve doc
    obj = {
        "doc_key": doc_key,
        "sentences": sentences,
        "speakers": [["-"] * len(s) for s in sentences],
        "clusters": list(clusters_map.values()),
    }
    return obj

def convert(input_dir: Path, output_dir: Path, split_name: str = "all"):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split_name}.jsonlines"

    n_docs = 0
    n_with_clusters = 0

    with out_path.open("w", encoding="utf-8") as out:
        for xml_path in sorted(input_dir.rglob("*.xml")):
            obj = convert_one_xml(xml_path)
            if obj is None:
                continue

            # filtra docs vazios total
            if not obj["sentences"]:
                continue

            out.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n_docs += 1
            if obj["clusters"]:
                n_with_clusters += 1

    print(f"{n_docs} docs → {out_path}")
    print(f"docs com clusters: {n_with_clusters}/{n_docs}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Pasta com XMLs do CorefPT")
    ap.add_argument("--output_dir", required=True, help="Pasta de saída")
    ap.add_argument("--split_name", default="all", help="Nome do arquivo de saída (ex: train/dev/test/all)")
    args = ap.parse_args()
    convert(Path(args.input_dir), Path(args.output_dir), split_name=args.split_name)