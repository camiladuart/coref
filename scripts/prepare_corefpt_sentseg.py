import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer

END_PUNCT = {".", "!", "?", "...", "…", "?!", "!?", ";", ":"}
CLOSE_QUOTES = {'"', "'", "”", "’", "»", ")", "]", "}"}

def is_end_token(tok: str) -> bool:
    tok = str(tok)
    if tok in END_PUNCT:
        return True
    if len(tok) > 1 and tok[-1] in ".!?":
        # evita separar abreviações muito óbvias tipo "Dr.", "Sr.", "Sra."
        low = tok.lower()
        if low in {"dr.", "dra.", "sr.", "sra.", "prof.", "profa.", "ex.", "etc."}:
            return False
        return True
    return False

def normalize_sentences(sentences):
    """
    Retorna:
      old_sentences: list[list[str]]
      flat_tokens: list[str]

    Aceita formatos comuns:
      - [[tok, tok], [tok, tok]]
      - [tok, tok, tok]
      - ["texto inteiro ..."]
      - "texto inteiro ..."
    """
    if isinstance(sentences, str):
        old_sentences = [sentences.split()]
    elif isinstance(sentences, list):
        if not sentences:
            old_sentences = []
        elif all(isinstance(x, list) for x in sentences):
            old_sentences = [[str(t) for t in sent] for sent in sentences]
        elif all(isinstance(x, str) for x in sentences):
            # Se for ["O gato dorme ."] trata como frase textual.
            # Se for ["O", "gato", "dorme", "."] trata como uma sentença tokenizada.
            if len(sentences) == 1 and (" " in sentences[0]):
                old_sentences = [sentences[0].split()]
            elif any(" " in x for x in sentences):
                old_sentences = [x.split() for x in sentences]
            else:
                old_sentences = [[str(t) for t in sentences]]
        else:
            raise ValueError(f"Formato inesperado em sentences: {type(sentences)}")
    else:
        raise ValueError(f"Formato inesperado em sentences: {type(sentences)}")

    flat_tokens = [tok for sent in old_sentences for tok in sent]
    return old_sentences, flat_tokens

def flatten_speakers(obj, old_sentences, n_tokens):
    speakers = obj.get("speakers", None)
    if speakers is None:
        return None

    # speakers como list[list], mesmo shape das sentenças antigas
    if (
        isinstance(speakers, list)
        and len(speakers) == len(old_sentences)
        and all(isinstance(x, list) for x in speakers)
    ):
        flat = []
        ok = True
        for sent, spk_sent in zip(old_sentences, speakers):
            if len(spk_sent) == len(sent):
                flat.extend([str(s) if s is not None else "UNK" for s in spk_sent])
            elif len(spk_sent) == 1:
                flat.extend([str(spk_sent[0]) if spk_sent[0] is not None else "UNK"] * len(sent))
            else:
                ok = False
                break
        if ok and len(flat) == n_tokens:
            return flat

    # speakers como lista plana por token
    if isinstance(speakers, list) and all(not isinstance(x, list) for x in speakers):
        if len(speakers) == n_tokens:
            return [str(s) if s is not None else "UNK" for s in speakers]

        # speakers como 1 speaker por sentença antiga
        if len(speakers) == len(old_sentences):
            flat = []
            for sent, spk in zip(old_sentences, speakers):
                flat.extend([str(spk) if spk is not None else "UNK"] * len(sent))
            if len(flat) == n_tokens:
                return flat

    # Se não der para alinhar, devolve UNK para não quebrar shape.
    return ["UNK"] * n_tokens

def wp_len(tokenizer, tokens):
    if not tokens:
        return 0
    enc = tokenizer(
        tokens,
        is_split_into_words=True,
        add_special_tokens=False,
        truncation=False,
        return_tensors="pt",
    )
    return enc["input_ids"].size(1)

def split_tokens_into_sentences(tokens, tokenizer, max_wp):
    """
    Divide por pontuação.
    Se alguma frase ainda passar de max_wp, quebra em chunks artificiais.
    Nunca muda a ordem nem a quantidade de tokens.
    """
    rough = []
    cur = []

    for tok in tokens:
        cur.append(tok)

        if is_end_token(tok):
            rough.append(cur)
            cur = []
        elif tok in CLOSE_QUOTES and len(cur) >= 2 and is_end_token(cur[-2]):
            rough.append(cur)
            cur = []

    if cur:
        rough.append(cur)

    final = []
    for sent in rough:
        if wp_len(tokenizer, sent) <= max_wp:
            final.append(sent)
            continue

        chunk = []
        for tok in sent:
            candidate = chunk + [tok]
            if chunk and wp_len(tokenizer, candidate) > max_wp:
                final.append(chunk)
                chunk = [tok]
            else:
                chunk = candidate

        if chunk:
            final.append(chunk)

    return final

def split_speakers_like_sentences(flat_speakers, new_sentences):
    out = []
    pos = 0
    for sent in new_sentences:
        n = len(sent)
        out.append(flat_speakers[pos:pos+n])
        pos += n
    return out

def max_sentence_wp(tokenizer, sentences):
    if not sentences:
        return 0
    return max(wp_len(tokenizer, sent) for sent in sentences)

def process_file(input_path, output_path, tokenizer, max_wp):
    docs = 0
    changed = 0
    max_before = 0
    max_after = 0
    bad_after = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, 1):
            if not line.strip():
                continue

            obj = json.loads(line)
            old_sentences, old_flat = normalize_sentences(obj["sentences"])
            n_tokens = len(old_flat)

            before_wp = max_sentence_wp(tokenizer, old_sentences)
            max_before = max(max_before, before_wp)

            new_sentences = split_tokens_into_sentences(old_flat, tokenizer, max_wp=max_wp)
            new_flat = [tok for sent in new_sentences for tok in sent]

            if old_flat != new_flat:
                raise RuntimeError(
                    f"{input_path.name}:{line_no}: ERRO: tokens mudaram depois da divisão."
                )

            after_wp = max_sentence_wp(tokenizer, new_sentences)
            max_after = max(max_after, after_wp)

            if after_wp > 512:
                bad_after += 1

            if len(new_sentences) != len(old_sentences):
                changed += 1

            obj["sentences"] = new_sentences

            if "speakers" in obj:
                flat_speakers = flatten_speakers(obj, old_sentences, n_tokens)
                obj["speakers"] = split_speakers_like_sentences(flat_speakers, new_sentences)

                speaker_flat = [s for sent in obj["speakers"] for s in sent]
                if len(speaker_flat) != len(new_flat):
                    raise RuntimeError(
                        f"{input_path.name}:{line_no}: ERRO: speakers desalinhados."
                    )

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            docs += 1

    return {
        "file": input_path.name,
        "docs": docs,
        "changed_docs": changed,
        "max_sentence_wp_before": max_before,
        "max_sentence_wp_after": max_after,
        "docs_still_over_512": bad_after,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--encoder_name", default="neuralmind/bert-base-portuguese-cased")
    ap.add_argument("--max_wp", type=int, default=480)
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(args.encoder_name, local_files_only=True)

    print(f"[INPUT]  {input_dir}")
    print(f"[OUTPUT] {output_dir}")
    print(f"[ENCODER] {args.encoder_name}")
    print(f"[MAX WP por frase/chunk] {args.max_wp}")

    all_stats = []
    for split in ["train", "dev", "test"]:
        inp = input_dir / f"{split}.jsonlines"
        out = output_dir / f"{split}.jsonlines"

        if not inp.exists():
            raise FileNotFoundError(inp)

        stats = process_file(inp, out, tokenizer, max_wp=args.max_wp)
        all_stats.append(stats)

    print("\n===== SUMMARY =====")
    for s in all_stats:
        print(
            f"{s['file']}: docs={s['docs']} | changed={s['changed_docs']} | "
            f"max_wp_before={s['max_sentence_wp_before']} | "
            f"max_wp_after={s['max_sentence_wp_after']} | "
            f"still_over_512={s['docs_still_over_512']}"
        )

    if any(s["docs_still_over_512"] > 0 for s in all_stats):
        raise RuntimeError("Ainda existe sentença/chunk com mais de 512 wordpieces.")

    print("\nOK: arquivos corrigidos sem mudar tokens nem offsets.")
    print("Use esta pasta como DATA_DIR no treino:")
    print(output_dir)

if __name__ == "__main__":
    main()
