#CorefUD/CRAC26 (.conllu) -> JSONLines (doc_key + sentences + clusters).
#Em vez de clusters: [], preencher clusters lendo Entity= da coluna MISC, conforme o formato descrito no PDF
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

NEW_DOC_RE = re.compile(r"^#\s*newdoc(?:\s+id)?\s*=\s*(.+)\s*$", re.IGNORECASE)

#extrai atributo KEY= da coluna MISC
def misc_get_attr(misc: str, key: str) -> Optional[str]:
    if not misc or misc == "_":
        return None
    for part in misc.split("|"):
        if part.startswith(key + "="):
            return part[len(key) + 1 :]
    return None

#parse p transformar em eventos simples:
def parse_entity_events(entity_val: str) -> List[Tuple[str, str, Optional[Tuple[int,int]]]]:
    if not entity_val:
        return []

    s = entity_val.strip()

    events: List[Tuple[str, str, Optional[Tuple[int,int]]]] = []
    i = 0
    L = len(s)

    def read_id(j: int) -> Tuple[Optional[str], Optional[Tuple[int,int]], int]:
        #lê base até encontrar delimitador
        start = j
        while j < L and s[j] not in "-) \t\r\n":
            # aceita colchetes como parte do token do id
            if s[j] == "|":
                break
            j += 1
        token = s[start:j]
        if not token:
            return None, None, j

        # separa base
        m = re.match(r"^(e\d+)(?:\[(\d+)\/(\d+)\])?$", token)
        if m:
            base = m.group(1)
            if m.group(2) and m.group(3):
                return base, (int(m.group(2)), int(m.group(3))), j
            return base, None, j

        #se vier algo estranho, pega até o primeiro '-' 
        base = token.split("-", 1)[0]
        m2 = re.match(r"^(e\d+)$", base)
        if m2:
            return base, None, j
        return base, None, j

    while i < L:
        ch = s[i]

        if ch == "(":
            #abertura: "(e5-..."; ID começa logo após '('
            i += 1
            base, part, i2 = read_id(i)
            if base:
                events.append(("open", base, part))
            i = i2
            continue

        #fechamento: "e5)" (pode aparecer várias vezes)
        if ch == "e":
            base, part, i2 = read_id(i)
            if base and i2 < L and s[i2] == ")":
                events.append(("close", base, part))
                i = i2 + 1
                continue

        i += 1

    return events

#escrever um doc json - 1 linha: 
def flush_doc(
    out_fh,
    doc_key: str,
    sentences: List[List[str]],
    clusters_map: Dict[str, List[List[int]]],
):
    # converte dict -> lista
    clusters = list(clusters_map.values())
    obj = {
        "doc_key": doc_key,
        "sentences": sentences,
        "speakers": [["-"] * len(s) for s in sentences],
        "clusters": clusters,
    }
    out_fh.write(json.dumps(obj, ensure_ascii=False) + "\n")

#le .conllu do input_dir e gera jsonlines output
def convert(input_dir: Path, output_dir: Path, split_name: str = "all"):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{split_name}.jsonlines"

    n_docs = 0

    with out_path.open("w", encoding="utf-8") as out:
        for conllu_path in sorted(input_dir.rglob("*.conllu")):
            #estado do arquivo 
            current_doc_key = conllu_path.stem
            sentences: List[List[str]] = []
            current_sent: List[str] = []
            #Coref parsing (índices globais no doc)
            token_index = 0
            # stacks por cluster p/ spans contíguos
            open_stack: Dict[str, List[int]] = {}
            # Para menções descontínuas: 
            disc_state: Dict[str, Dict] = {}

            clusters_map: Dict[str, List[List[int]]] = {}

            def ensure_cluster(cid: str):
                clusters_map.setdefault(cid, [])

            def start_disc_if_needed(cid: str, part: Tuple[int,int]):
                st = disc_state.setdefault(cid, {"next_gid": 1, "active_gid": None, "n": None, "pieces": {}})
                k, n = part
                #quando tiver [1/n] abrindo, inicia novo grupo.
                if k == 1 or st["active_gid"] is None:
                    gid = st["next_gid"]
                    st["next_gid"] += 1
                    st["active_gid"] = gid
                    st["n"] = n
                    st["pieces"][gid] = {}
                return st

            def try_finalize_disc(cid: str):
                st = disc_state.get(cid)
                if not st or st["active_gid"] is None or st["n"] is None:
                    return
                gid = st["active_gid"]
                n = st["n"]
                pieces = st["pieces"].get(gid, {})
                if len(pieces) == n:
                    # colocar cada parte como um span separado no mesmo cluster
                    ensure_cluster(cid)
                    for k in sorted(pieces.keys()):
                        s,e = pieces[k]
                        clusters_map[cid].append([s,e])
                    # fechar grupo
                    st["active_gid"] = None
                    st["n"] = None

            def close_disc_piece(cid: str, part: Tuple[int,int], start_idx: int, end_idx: int):
                st = start_disc_if_needed(cid, part)
                k, n = part
                gid = st["active_gid"]
                st["pieces"][gid][k] = (start_idx, end_idx)
                try_finalize_disc(cid)
                
            #fechar mention:
            def close_span(cid: str, start_idx: int, end_idx: int):
                ensure_cluster(cid)
                clusters_map[cid].append([start_idx, end_idx])

            #processar Entity no token atual
            def process_entity(entity_val: str, tok_idx: int):
                events = parse_entity_events(entity_val)
                for kind, cid, part in events:
                    if kind == "open":
                        if part is None:
                            open_stack.setdefault(cid, []).append(tok_idx)
                        else:
                            #guardar abertura por (cid,part_k)
                            key = f"{cid}[{part[0]}/{part[1]}]"
                            open_stack.setdefault(key, []).append(tok_idx)

                    elif kind == "close":
                        if part is None:
                            st = open_stack.get(cid)
                            if st:
                                start = st.pop()
                                close_span(cid, start, tok_idx)
                        else:
                            key = f"{cid}[{part[0]}/{part[1]}]"
                            st = open_stack.get(key)
                            if st:
                                start = st.pop()
                                close_disc_piece(cid, part, start, tok_idx)

            #ler arquivo
            with conllu_path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.rstrip("\n")

                    #newdoc
                    m = NEW_DOC_RE.match(line)
                    if m:
                        #flush doc anterior (se já tem conteúdo)
                        if sentences or current_sent:
                            if current_sent:
                                sentences.append(current_sent)
                                current_sent = []
                            flush_doc(out, current_doc_key, sentences, clusters_map)
                            n_docs += 1

                        current_doc_key = m.group(1).strip()
                        sentences = []
                        current_sent = []
                        token_index = 0
                        open_stack = {}
                        disc_state = {}
                        clusters_map = {}
                        continue

                    #comentário
                    if line.startswith("#"):
                        continue

                    #fim de sentença
                    if line.strip() == "":
                        if current_sent:
                            sentences.append(current_sent)
                            current_sent = []
                        continue

                    cols = line.split("\t")
                    if len(cols) < 10:
                        continue

                    tid = cols[0]
                    form = cols[1]
                    misc = cols[9]

                    #ignora multiword token ("1-2")
                    if "-" in tid:
                        continue

                    # ignora empty nodes (3.1)
                    if "." in tid:
                        continue
      
                    current_sent.append(form)

                    ent = misc_get_attr(misc, "Entity")
                    if ent:
                        process_entity(ent, token_index)

                    token_index += 1

            # flush último doc do arquivo
            if current_sent:
                sentences.append(current_sent)

            #só escreve se tem algo
            if sentences:
                flush_doc(out, current_doc_key, sentences, clusters_map)
                n_docs += 1

    print(f"{n_docs} docs → {out_path}")

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True, help="Pasta com .conllu (CorefUD/CRAC26)")
    ap.add_argument("--output_dir", required=True, help="Pasta de saída")
    ap.add_argument("--split_name", default="all", help="Nome do arquivo de saída (ex: train/dev/test/all)")
    args = ap.parse_args()
    convert(Path(args.input_dir), Path(args.output_dir), split_name=args.split_name)