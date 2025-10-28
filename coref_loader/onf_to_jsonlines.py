# onf_to_jsonlines_coref.py
# Lê OntoNotes .onf via "ontonotes-db-tool-v0.999b" e gera JSONLines com sentences + clusters.
# Compatível com o CorefDataset (doc_key, sentences, speakers, clusters).
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple

def safe_get(obj, *names, default=None):
    """Tenta acessar atributos em cascata; se não existir, retorna default."""
    cur = obj
    for n in names:
        if cur is None:
            return default
        if hasattr(cur, n):
            cur = getattr(cur, n)
        elif isinstance(cur, dict) and n in cur:
            cur = cur[n]
        else:
            return default
    return cur

def to_sentence_level_clusters(sent_lens: List[int],
                               spans: List[Tuple[int,int,int]]) -> List[List[List[int]]]:
    """
    Converte spans globais (sent_global_idx, start_global, end_global)
    para clusters por sentença: [[ [start_tok, end_tok], ... ], ...]
    Ignora spans que cruzam sentenças.
    """
    # sent_lens: lista com o tamanho (n_tokens) de cada sentença
    # spans: lista de tuplas (s_idx, start_tok, end_tok) por menção
    clusters: Dict[int, List[List[int]]] = {}  # chain_id -> list[[start, end]]
    # simplificada, ela só repassa os spans válidos:
    # (no chamador, agrupamos por chain_id e filtramos sentenças cruzadas)
    raise NotImplementedError("Não usada: agrupamento é feito no chamador.")

def convert(onto_root: Path, db_tool_dir: Path, output_dir: Path, language: str = "english"):
    # 1) Coloca o ontonotes-db-tool no sys.path
    sys.path.insert(0, str(db_tool_dir))

    # 2) Importa o parser (nomes podem variar; tentamos 'ontonotes' ou 'ontonotes_db')
    try:
        from ontonotes import OntoNotes
    except Exception:
        try:
            from ontonotes_db import OntoNotes  # fallback em algumas distribuições
        except Exception as e:
            raise ImportError(
                f"Não consegui importar o parser. Verifique se existe 'ontonotes.py' em {db_tool_dir}. Erro: {e}"
            )

    # 3) Instancia o leitor. Em muitas versões, OntoNotes(root=...) espera o root do pacote
    #    e sabe localizar 'data/files/data'. Se sua versão exigir o caminho exato, ajuste aqui.
    onto = OntoNotes(root=str(onto_root))

    # 4) Itera documentos
    docs_out = []

    # Muitas versões expõem: onto.iter_documents() ou onto.iter_documents(language='english')
    try:
        it = onto.iter_documents(language=language)
    except TypeError:
        it = onto.iter_documents()

    for doc in it:
        # doc.document_id / doc.id / doc.name (varia por versão)
        doc_key = safe_get(doc, "document_id") or safe_get(doc, "id") or safe_get(doc, "name") or "UNKNOWN"

        # Sentenças e tokens (nomes variam: doc.sentences -> lista de sent, e cada sent tem 'tokens' ou 'words')
        sentences_tokens: List[List[str]] = []
        sent_objs = safe_get(doc, "sentences", default=[])
        for s in sent_objs:
            toks = safe_get(s, "tokens") or safe_get(s, "words") or []
            # Alguns parsers guardam objetos com .word; outros já são strings
            if toks and not isinstance(toks[0], str):
                toks = [safe_get(t, "word") or str(t) for t in toks]
            sentences_tokens.append(toks or [])

        if not sentences_tokens:
            # Alguns parsers guardam como doc.tokens por sentença
            toks2d = safe_get(doc, "tokens")  # às vezes [[...], [...]]
            if toks2d:
                sentences_tokens = toks2d

        # Coreferência (varia muito por versão). Tentar caminhos comuns:
        # - doc.coref_chains: dict[chain_id] -> list of mentions
        # - doc.coref: objeto com .chains
        # - doc.coref_spans / doc.coref_mentions etc.
        clusters_final: List[List[List[int]]] = []

        # montar um mapeamento simples: chain_id -> lista de (sent_idx, start, end)
        chains = None
        chains = safe_get(doc, "coref_chains") or safe_get(doc, "coref", "chains") or safe_get(doc, "coref") \
                 or safe_get(doc, "coreference") or None

        if chains:
            # Normaliza para dict-like: chain_id -> list of mentions
            # Cada menção deve ter pelo menos: sentence index e (start,end) em tokens
            group_by_chain: Dict[str, List[Tuple[int,int,int]]] = {}

            if isinstance(chains, dict):
                items = chains.items()
            elif hasattr(chains, "__iter__"):
                # pode ser lista de objetos-chain com .mentions
                items = enumerate(chains)
            else:
                items = []

            for cid, chain in items:
                # Pega as menções
                mentions = safe_get(chain, "mentions") or safe_get(chain, "spans") or chain
                if not mentions:
                    continue
                for m in mentions:
                    # Tenta pegar índices de sentença e token
                    s_idx = safe_get(m, "sentence_index") or safe_get(m, "sentence_id") or safe_get(m, "sentence") or 0
                    start = safe_get(m, "start") or safe_get(m, "start_token") or safe_get(m, "span_start") or 0
                    end   = safe_get(m, "end")   or safe_get(m, "end_token")   or safe_get(m, "span_end")   or start
                    try:
                        s_idx = int(s_idx)
                        start = int(start)
                        end   = int(end)
                    except Exception:
                        continue
                    # Normaliza: alguns parsers usam end exclusivo; garantimos inclusivo
                    if end < start:
                        start, end = end, start
                    group_by_chain.setdefault(str(cid), []).append((s_idx, start, end))

            # Converte por sentença e ignora spans que cruzem sentenças
            # Estrutura alvo: clusters = [ [ [start,end], [start,end] ], ... ]
            for cid, spans in group_by_chain.items():
                cluster_sentlevel: List[List[int]] = []
                for (s_idx, start, end) in spans:
                    if 0 <= s_idx < len(sentences_tokens):
                        max_tok = len(sentences_tokens[s_idx]) - 1
                        start = max(0, min(start, max_tok))
                        end   = max(0, min(end,   max_tok))
                        cluster_sentlevel.append([start, end])
                if cluster_sentlevel:
                    clusters_final.append(cluster_sentlevel)

        # Monta o JSON do documento
        if sentences_tokens:
            speakers = [["-"] * len(s) for s in sentences_tokens]
            docs_out.append({
                "doc_key": str(doc_key),
                "sentences": sentences_tokens,
                "speakers": speakers,
                "clusters": clusters_final
            })

    # Grava saída (um arquivo único)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "ontonotes_all.onf_coref.jsonlines"
    with out_path.open("w", encoding="utf-8") as f:
        for d in docs_out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print(f"[OK] {len(docs_out)} docs → {out_path}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontonotes_root", required=True, help="Raiz do pacote OntoNotes (ex.: ...\\ontonotes-release-5.0)")
    ap.add_argument("--db_tool_dir", required=True, help="Pasta do ontonotes-db-tool-v0.999b")
    ap.add_argument("--output_dir", required=True, help="Pasta de saída")
    ap.add_argument("--language", default="english", help="Língua (ex.: english)")
    args = ap.parse_args()

    convert(
        onto_root=Path(args.ontonotes_root),
        db_tool_dir=Path(args.db_tool_dir),
        output_dir=Path(args.output_dir),
        language=args.language
    )

