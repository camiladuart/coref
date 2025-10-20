#code to load jsonlines from coref datasets

import json
from pathlib import Path

class CorefDataset: #class to read and load coref data in jsonlines format. each line is a document with: doc_key (name), sentences (tokens list), clusters (groups of mentions).

    def __init__(self, data_dir, split): #initialize dataset. param data_dir = folder with jsonlines files; param split = train/dev/test.
        self.data_dir = Path(data_dir)
        self.split = split
        self.samples = self._load_jsonlines()

    def _load_jsonlines(self): #reads jsonlines file and returns a list of dictionnaires (one by document)
        file_path = self.data_dir / f"{self.split}.english.jsonlines"
        if not file_path.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        docs = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                docs.append(data)
        return docs

    def __len__(self): #returns number of documents
        return len(self.samples)

    def __getitem__(self, idx): #allows to access a document by its index (ex: dataset[0])
        return self.samples[idx]
