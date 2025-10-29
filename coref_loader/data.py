# loader de JSONLines:
# sem classes que herdam de torch.utils.data.Dataset -> como Seq2Seq; sem transformações, tokenização, ...
# não herda nada -> classe pequena -> só lê, valida e entrega os documentos

import json
from pathlib import Path
from typing import List, Dict, Any

class CorefDataset: ## lê arqs jsonlines do ontonotes. cada linha: {"doc_key", "sentences", "speakers"(opcional), "clusters"(opcional)}

    def __init__(self, data_dir: str, split: str): #data_dir: pasta com arq.english.jsonlines; split: "train", "dev" ou "test"
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples: List[Dict[str, Any]] = self._load_jsonlines() #chama o load_jsonlines e guarda o resultado em self.samples

    def _load_jsonlines(self) -> List[Dict[str, Any]]: #abrir e ler o arquivo
        path = self.data_dir / f"{self.split}.english.jsonlines" #montando o caminho
        if not path.exists():
            raise FileNotFoundError(f"Não encontrei: {path}")
        docs = [] #preparando lista vazia para juntar os docs
        with path.open("r", encoding="utf-8") as f:
            for line in f: #lê linha por linha
                line = line.strip() #strip pra tirar espaços/quebras de linha
                if not line:
                    continue
                d = json.loads(line) #converte texto json da linha p um dicionario python d
                #cada doc precisa ter doc_key e sentences. senao, mensagem de erro:
                if "doc_key" not in d or "sentences" not in d:
                    raise ValueError(f"JSON inválido (faltam campos): {d}")
                # campos opcionais: valor padrao [], -
                d.setdefault("clusters", [])
                d.setdefault("speakers", [["-"] * len(s) for s in d["sentences"]])
                docs.append(d) #acrescenta cada doc em docs e retorna a lista completa
        return docs

    def __len__(self) -> int:
        return len(self.samples) #len p saber quantos docs há

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx] #retorna o primeiro documento
