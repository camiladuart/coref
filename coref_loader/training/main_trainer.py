import argparse
import torch
import json
from pathlib import Path
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from coref_loader.data import CorefDataset, extract_gold_spans_with_clusters
from coref_loader.training.model import CorefModel
from tqdm import tqdm
import numpy as np

def sentence_chunks(sentences, max_segment_len):
    for i in range(0, len(sentences), max_segment_len):
        yield i, sentences[i : i + max_segment_len]

def evaluate(model, dataset, tokenizer, config, device, limit=None):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_spans = 0
    doc_count = 0
    
    with torch.no_grad():
        for idx, ex in enumerate(dataset):
            if limit and idx >= limit:
                break
                
            sentences = ex["sentences"]
            gold_starts_all, gold_ends_all, gold_cluster_ids_all = extract_gold_spans_with_clusters(ex)
            genre = ex.get("genre", None)
            
            doc_loss = 0.0
            doc_spans = 0
            doc_correct = 0
            
            for seg_start, seg_sents in sentence_chunks(sentences, config["max_segment_len"]):
                logits, mention_labels, loss = model.forward(
                    sentences=sentences,
                    seg_start=seg_start,
                    seg_sents=seg_sents,
                    tokenizer=tokenizer,
                    gold_starts_all=gold_starts_all,
                    gold_ends_all=gold_ends_all,
                    gold_cluster_ids_all=gold_cluster_ids_all,
                    max_span_width=config["max_span_width"],
                    genre=genre,
                )
                
                if logits.numel() == 0:
                    continue
                    
                if loss is not None and torch.is_tensor(loss):
                    doc_loss += loss.item()
                    
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                doc_correct += (preds == mention_labels).sum().item()
                doc_spans += mention_labels.numel()
            
            if doc_spans > 0:
                total_loss += doc_loss
                total_acc += doc_correct
                total_spans += doc_spans
                doc_count += 1
    
    avg_loss = total_loss / max(doc_count, 1)
    avg_acc = total_acc / max(total_spans, 1)
    
    return avg_loss, avg_acc

#train for one epoch
def train_epoch(model, dataset, tokenizer, config, optimizer, device, epoch, limit=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_spans = 0
    doc_count = 0

    pbar = tqdm(enumerate(dataset),
                total=(len(dataset) if not limit else limit),
                desc=f"Epoch {epoch}")

    for idx, ex in pbar:
        if limit and idx >= limit:
            break

        sentences = ex["sentences"]
        gold_starts_all, gold_ends_all, gold_cluster_ids_all = extract_gold_spans_with_clusters(ex)
        genre = ex.get("genre", None)

        doc_spans = 0
        doc_correct = 0

        optimizer.zero_grad(set_to_none=True)

        #loss acumulada do documento (tensor)
        doc_total_loss = torch.tensor(0.0, device=device)

        for seg_start, seg_sents in sentence_chunks(sentences, config["max_segment_len"]):
            logits, mention_labels, loss = model.forward(
                sentences=sentences,
                seg_start=seg_start,
                seg_sents=seg_sents,
                tokenizer=tokenizer,
                gold_starts_all=gold_starts_all,
                gold_ends_all=gold_ends_all,
                gold_cluster_ids_all=gold_cluster_ids_all,
                max_span_width=config["max_span_width"],
                genre=genre,
            )

            if logits.numel() == 0:
                continue

            #acumula loss 
            if loss is not None and torch.is_tensor(loss) and loss.requires_grad:
                doc_total_loss = doc_total_loss + loss

            #métricas 
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).long()
                doc_correct += (preds == mention_labels).sum().item()
                doc_spans += mention_labels.numel()

        #update uma vez no final do documento
        if doc_spans > 0:
            doc_total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += float(doc_total_loss.item())
            total_acc += doc_correct
            total_spans += doc_spans
            doc_count += 1

            pbar.set_postfix({
                "loss": f"{total_loss / doc_count:.4f}",
                "acc": f"{total_acc / total_spans:.3f}",
            })

    avg_loss = total_loss / max(doc_count, 1)
    avg_acc = total_acc / max(total_spans, 1)
    return avg_loss, avg_acc

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="Pasta com *.english.jsonlines")
    ap.add_argument("--split", default="train", choices=["train", "dev", "test"])
    ap.add_argument("--limit", type=int, default=0, help="Quantos docs usar (0 = todos)")

    # Training arguments
    ap.add_argument("--epochs", type=int, default=20, help="Número de épocas")
    ap.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    ap.add_argument("--eval_every", type=int, default=1, help="Avaliar a cada N épocas")
    ap.add_argument("--save_dir", type=str, default="checkpoints", help="Diretório para salvar modelos")
    ap.add_argument("--resume_from", type=str, default=None, help="Checkpoint para continuar treinamento")
    
    ap.add_argument("--inspect", action="store_true", help="Inspeciona 1 doc do split.")
    ap.add_argument("--inspect_idx", type=int, default=0, help="Índice do documento a inspecionar no split.")
    ap.add_argument("--thresh", type=float, default=0.2, help="Threshold para considerar mention.")
    
    args = ap.parse_args()

    # Config
    config = {
        "encoder_name": "bert-base-cased",
        "max_span_width": 30,
        "max_segment_len": 3,
        "top_span_ratio": 0.4,
        "max_top_antecedents": 50,
        "use_genre": False, #por agora - outros datasets
        "genres": [],
        "genre_emb_size": 20,
    }

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)

    # Load datasets
    print(f"[Loading datasets from: {args.data_dir}]")

    train_ds = None
    dev_ds = None
    test_ds = None

    if args.split == "train":
        train_ds = CorefDataset(args.data_dir, "train", config)
        dev_ds   = CorefDataset(args.data_dir, "dev", config)
        print(f"[OK] Train: {len(train_ds)} docs")
        print(f"[OK] Dev: {len(dev_ds)} docs")

    elif args.split == "dev":
        dev_ds = CorefDataset(args.data_dir, "dev", config)
        print(f"[OK] Dev: {len(dev_ds)} docs")

    elif args.split == "test":
        test_ds = CorefDataset(args.data_dir, "test", config)
        print(f"[OK] Test: {len(test_ds)} docs")
    
    # Auto-detect genres
    if config.get("use_genre", False):
        #detectar a partir do train
        source_ds = train_ds if train_ds is not None else (dev_ds if dev_ds is not None else test_ds)

        all_genres = sorted({ex.get("genre") for ex in source_ds.samples if ex.get("genre") is not None})
        config["genres"] = all_genres
        print(f"[OK] Genres detectados: {len(all_genres)} -> {all_genres[:10]}...")

        if len(all_genres) == 0:
            raise ValueError("Auto-detect de gêneros vazio (nenhum 'genre' encontrado no split).")

    # Model, tokenizer, device
    tokenizer = AutoTokenizer.from_pretrained(config["encoder_name"])
    model = CorefModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device: {device}]")
    model.to(device)
    
    # Resume from checkpoint 
    start_epoch = 0
    if args.resume_from and args.split in ["train", "test"]:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[Loaded checkpoint: {args.resume_from}]")

    def union_find_clusters(n, links):
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a,b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for a,b in links:
            union(a,b)
        groups = {}
        for i in range(n):
            r = find(i)
            groups.setdefault(r, []).append(i)
        return list(groups.values())

    def inspect_one_doc(model, ex, tokenizer, config, device, thresh=args.thresh):
        model.eval()
        print(f"\nDOC_KEY: {ex.get('doc_key','(sem doc_key)')}")
        print(f"GENRE: {ex.get('genre', None)}")
        sentences = ex["sentences"]
        print(f"Num sentences: {len(sentences)}")

        gold_starts_all, gold_ends_all, gold_cluster_ids_all = extract_gold_spans_with_clusters(ex)
        genre = ex.get("genre", None)

        with torch.no_grad():
            for seg_start, seg_sents in sentence_chunks(sentences, config["max_segment_len"]):
                logits, mention_labels, loss = model.forward(
                    sentences=sentences,
                    seg_start=seg_start,
                    seg_sents=seg_sents,
                    tokenizer=tokenizer,
                    gold_starts_all=gold_starts_all,
                    gold_ends_all=gold_ends_all,
                    gold_cluster_ids_all=gold_cluster_ids_all,
                    max_span_width=config["max_span_width"],
                    genre=genre,
                    return_debug=True,
                )

                dbg = getattr(model, "last_debug", None)
                if dbg is None or dbg["span_starts_tok"] is None:
                    print(f"\nSegment {seg_start}: (sem spans no beam)")
                    continue

                probs = torch.sigmoid(logits).detach().cpu()
                starts = dbg["span_starts_tok"].numpy().tolist()
                ends   = dbg["span_ends_tok"].numpy().tolist()
                pair   = dbg["pair_scores"].numpy()
                
                #debug extra. mostrar top spans mesmo se threshold n for passado
                topk = min(10, len(probs))
                vals, idxs = torch.topk(probs, k=topk)
                print("Top probs:", [float(v) for v in vals])


                # tokens do segmento
                seg_tokens = [w for sent in seg_sents for w in sent]

                # spans acima do threshold
                kept = [i for i,p in enumerate(probs) if float(p) >= thresh]
                print(f"\nSEGMENT start={seg_start} | spans_no_beam={len(probs)} | kept>={thresh} = {len(kept)}")
                if len(kept) == 0:
                    # mostra os top-10 spans mesmo sem passar o threshold
                    for i in idxs.tolist():
                        s,e = starts[i], ends[i]
                        txt = " ".join(seg_tokens[s:e+1]) if 0 <= s <= e < len(seg_tokens) else "(fora do range)"
                        print(f"  TOP span[{i:3d}] prob={float(probs[i]):.3f} tok=({s},{e}) text='{txt}'")
                    continue

                # imprimir spans
                for i in kept[:20]:
                    s,e = starts[i], ends[i]
                    if 0 <= s <= e < len(seg_tokens):
                        txt = " ".join(seg_tokens[s:e+1])
                    else:
                        txt = "(fora do range)"
                    print(f"  span[{i:3d}] prob={float(probs[i]):.3f} tok=({s},{e}) text='{txt}'")

                # links: melhor antecedente com score > 0
                links = []
                for i in kept:
                    if i == 0:
                        continue
                    row = pair[i][:i]
                    j = int(row.argmax()) if len(row) > 0 else -1
                    if j >= 0 and row[j] > 0:
                        links.append((i, j))

                print("\nLinks (span_i -> antecedente_j) com score>0:")
                for i,j in links[:30]:
                    print(f"  {i} -> {j}  score={pair[i][j]:.3f}")

                clusters = union_find_clusters(len(probs), links)
                # só clusters que têm pelo menos 2 spans e estão no kept
                clusters = [
                    [x for x in c if x in kept]
                    for c in clusters
                ]
                clusters = [c for c in clusters if len(c) >= 2]

                print("\nClusters (apenas size>=2):")
                if not clusters:
                    print("  (nenhum cluster com size>=2)")
                for k,c in enumerate(clusters, 1):
                    pieces = []
                    for idx in c:
                        s,e = starts[idx], ends[idx]
                        txt = " ".join(seg_tokens[s:e+1]) if 0 <= s <= e < len(seg_tokens) else "(fora do range)"
                        pieces.append(f"{idx}:{txt}")
                    print(f"  Cluster {k}: " + " | ".join(pieces))

    if args.inspect:
        # escolhe o split certo
        if args.split == "train":
            ds = train_ds
        elif args.split == "dev":
            ds = dev_ds
        else:
            ds = test_ds

        ex = ds[args.inspect_idx]
        inspect_one_doc(model, ex, tokenizer, config, device, thresh=args.thresh)
        return

    # DEV: evaluate all epochs (no training)
    if args.split == "dev":
        save_dir = Path(args.save_dir)
        ckpts = sorted(save_dir.glob("checkpoint_epoch_*.pt"),
                    key=lambda p: int(p.stem.split("_")[-1]))

        if len(ckpts) == 0:
            raise FileNotFoundError(f"Nenhum checkpoint_epoch_*.pt em {save_dir}")

        best_epoch = None
        best_loss = float("inf")

        print(f"\n[DEV] Evaluating {len(ckpts)} checkpoints...")
        for p in ckpts:
            epoch_num = int(p.stem.split("_")[-1])
            ckpt = torch.load(p, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

            dev_loss, dev_acc = evaluate(
                model, dev_ds, tokenizer, config, device,
                limit=args.limit if args.limit > 0 else None
            )
            print(f"[Dev][Epoch {epoch_num}] Loss: {dev_loss:.4f} | Acc: {dev_acc:.3f}")

            if dev_loss < best_loss:
                best_loss = dev_loss
                best_epoch = epoch_num

        print(f"\n[DEV] Best epoch by dev loss: epoch={best_epoch} dev_loss={best_loss:.4f}")
        print(f"[DEV] Use this checkpoint for TEST:")
        print(f"      {save_dir}/checkpoint_epoch_{best_epoch}.pt")
        best_path = save_dir / "best_epoch.txt"
        best_path.write_text(str(best_epoch) + "\n")
        print(f"[DEV] Saved best epoch to {best_path}")
        return

    # TEST: evaluate best checkpoint 
    if args.split == "test":

        # usa automaticamente o melhor do DEV
        if not args.resume_from:
            best_path = Path(args.save_dir) / "best_epoch.txt"
            if not best_path.exists():
                raise FileNotFoundError(
                    f"Não encontrado {best_path}. "
                    f"Rode primeiro: --split dev"
                )

            best_epoch = int(best_path.read_text().strip())
            args.resume_from = str(
                Path(args.save_dir) / f"checkpoint_epoch_{best_epoch}.pt"
            )
            print(f"[TEST] Auto resume_from = {args.resume_from}")

        # carrega o checkpoint escolhido
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[TEST] Loaded checkpoint: {args.resume_from}")

        test_loss, test_acc = evaluate(
            model, test_ds, tokenizer, config, device,
            limit=args.limit if args.limit > 0 else None
        )
        print(f"[TEST] Loss: {test_loss:.4f} | Acc: {test_acc:.3f}")
        return


    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Training loop
    print("\n[Starting training...]")
    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch+1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_ds, tokenizer, config, optimizer, device, 
            epoch+1, limit=args.limit if args.limit > 0 else None
        )
        print(f"\n[Train] Loss: {train_loss:.4f} | Acc: {train_acc:.3f}")
        if (epoch + 1) % args.eval_every == 0:
            dev_loss, dev_acc = evaluate(
                model, dev_ds, tokenizer, config, device,
                limit=args.limit if args.limit > 0 else None
            )
            print(f"[Dev] Loss: {dev_loss:.4f} | Acc: {dev_acc:.3f}")

        # Save checkpoint every epoch
        checkpoint_path = save_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'config': config,
        }, checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")

    print("\n[Training completed!]")

if __name__ == "__main__":
    main()