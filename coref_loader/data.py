# loader de JSONLines
import json
from pathlib import Path
from typing import List, Dict, Any
import torch
from typing import List, Tuple

def extract_genre(doc_key: str):
    if not isinstance(doc_key, str) or not doc_key:
        return None

    doc_key = doc_key.strip().lower()

    #prioriza os separadores mais comuns
    for sep in ["/", "_", "."]:
        if sep in doc_key:
            return doc_key.split(sep)[0]

    return doc_key


class CorefDataset: ## lê arqs jsonlines do ontonotes. cada linha: {"doc_key", "sentences", "speakers"(opcional), "clusters"(opcional)}
    def __init__(self, data_dir: str, split: str, config: Dict[str, Any]):
        self.config = config
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples: List[Dict[str, Any]] = self._load_jsonlines() #chama o load_jsonlines e guarda o resultado em self.samples

    def _load_jsonlines(self) -> List[Dict[str, Any]]: #abrir e ler o arquivo
        path = self.data_dir / f"{self.split}.jsonlines" #montando o caminho
        if not path.exists():
            raise FileNotFoundError(f"Não encontrei: {path}")
        docs = [] #preparando lista vazia para juntar os docs
        with path.open("r", encoding="utf-8") as f:
            for line in f: #lê linha por linha
                line = line.strip() #strip pra tirar espaços/quebras de linha
                if not line:
                    continue
                d = json.loads(line) #converte texto json da linha p um dicionario python d
                d["genre"] = extract_genre(d.get("doc_key", "")) #chama a função def acima
                #cada doc precisa ter doc_key e sentences. senao, mensagem de erro:
                if "doc_key" not in d or "sentences" not in d:
                    raise ValueError(f"JSON inválido (faltam campos): {d}")
                # campos opcionais: valor padrao [], -
                d.setdefault("clusters", [])
                d.setdefault("speakers", [["-"] * len(s) for s in d["sentences"]])
                docs.append(d) #acrescenta cada doc em docs e retorna a lista completa
        return docs

    def __len__(self) -> int:
        return len(self.samples) #quantos docs há

    def __getitem__(self, idx: int) -> Dict[str, Any]: #já pago o genero no load
        return self.samples[idx]



#juntar todas as sentenças num único vetor de tokens + criar um sentence_map (lista-> a qual sentença pertence cada token):
def flatten_sentences(sentences: List[List[str]]) -> Tuple[List[str], List[int]]:
    tokens = []
    sentence_map = []
    for s_idx, sent in enumerate(sentences):
        tokens.extend(sent)
        sentence_map.extend([s_idx] * len(sent))
    return tokens, sentence_map

#gerar todos os spans (start,end) que nao cruzam sentença e tem largura <= max_span_width:
def build_candidates(sentence_map: List[int], max_span_width: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
    N = len(sentence_map)
    starts = []
    ends = []
    for i in range(N): #for que vai aumentando o tamanho do span até max_span_width
        for w in range(max_span_width):
            j = i + w
            if j >= N:
                break #passou do ultimo token, para
            if sentence_map[i] == sentence_map[j]: #compara os dois sentence_maps:
                starts.append(i)                    #iguais = inicio e fim na mesma sentença -> span válido
                ends.append(j)                        #diferentes = j já está numa sentença diferente -> span cruzou sentenças -> parar
            else:
                break  
    if not starts: 
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long) ##se não encontrou nenhum span, retorna tensores vazios
    return torch.tensor(starts, dtype=torch.long), torch.tensor(ends, dtype=torch.long) #se sim, transforma as listas starts e ends em tensores torch.LongTensor (pra usar no modelo)

def extract_gold_spans(example): #pega clusters e retorna (gold_starts, gold_ends) únicos
    clusters = example.get("clusters", [])
    gold = sorted({(m[0], m[1]) for c in clusters for m in c})
    if not gold:
        return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)
    gold_starts = torch.tensor([s for s, _ in gold], dtype=torch.long)
    gold_ends   = torch.tensor([e for _, e in gold], dtype=torch.long)
    return gold_starts, gold_ends

#para pegar os gold spans com o id do cluster de cada um:
def extract_gold_spans_with_clusters(example):
    clusters = example.get("clusters", [])
    gold_spans = []
    gold_cluster_ids = []

    for cid, cluster in enumerate(clusters, start=1):
        for start, end in cluster:   #cada par [start, end]
            gold_spans.append((start, end))
            gold_cluster_ids.append(cid)
    if not gold_spans:
        #nada anotado:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
    starts, ends = zip(*gold_spans) #dividindo a lista em starts e ends
    return (
        torch.tensor(starts, dtype=torch.long),
        torch.tensor(ends, dtype=torch.long),
        torch.tensor(gold_cluster_ids, dtype=torch.long),
    ) #convertendo tudo para tensores

def sentence_chunks(sentences, max_segment_len):
    for i in range(0, len(sentences), max_segment_len):
        yield i, sentences[i : i + max_segment_len]
