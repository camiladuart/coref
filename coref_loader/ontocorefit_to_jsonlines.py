import json
from pathlib import Path

def convert_file(input_path: str, output_path: str, drop_singletons: bool = False):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = data["docs"]

    with open(output_path, "w", encoding="utf-8") as out:
        for doc_id, doc in docs.items():
            for part_id, part in doc["parts"].items():
                utts = part["utts"]

                sentences = []
                speakers = []
                clusters_by_label = {}

                token_offset = 0

                for utt in utts:
                    tokens = utt["tokens"]
                    speaker = utt["speakers"]

                    sentences.append(tokens)
                    speakers.append([speaker] * len(tokens))

                    for mention in utt["corefs"]:
                        start = token_offset + mention["start"]
                        end = token_offset + mention["end"]   # assumindo end inclusivo
                        label = str(mention["label"])

                        clusters_by_label.setdefault(label, []).append([start, end])

                    token_offset += len(tokens)

                clusters = []
                for label, spans in clusters_by_label.items():
                    # ordena spans pelo início
                    spans = sorted(spans, key=lambda x: (x[0], x[1]))

                    if drop_singletons and len(spans) < 2:
                        continue

                    clusters.append(spans)

                example = {
                    "doc_key": f"{doc_id}_{part_id}",
                    "sentences": sentences,
                    "speakers": speakers,
                    "clusters": clusters,
                }

                out.write(json.dumps(example, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    base = Path("/projects/F202600026AIVLABDEUCALION/up202000683/coref_data/OntoCorefIT")

    convert_file(
        str(base / "OntoCorefIT_train.json"),
        str(base / "train.jsonlines"),
        drop_singletons=False,
    )
    convert_file(
        str(base / "OntoCorefIT_dev.json"),
        str(base / "dev.jsonlines"),
        drop_singletons=False,
    )
    convert_file(
        str(base / "OntoCorefIT_test.json"),
        str(base / "test.jsonlines"),
        drop_singletons=False,
    )

    print("Conversão concluída.")
