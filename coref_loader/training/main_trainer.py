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
    
    args = ap.parse_args()

    # Config
    config = {
        "encoder_name": "bert-base-cased",
        "max_span_width": 30,
        "max_segment_len": 3,
        "top_span_ratio": 0.4,
        "max_top_antecedents": 50,
        "use_genre": True,
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