import os
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
import os
import csv
from datetime import datetime
import torch.nn.functional as F
import pickle
import pandas as pd
from datasets import load_dataset, disable_progress_bar

def load_cf_pairs(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [(item["s1"], item["s2"]) for item in data]

def load_task_data(data_name, pct=1.0, subset="test"):
    def sample_balanced(dataset, label_field="label", pct=1.0):
        if pct < 1.0:
            all_labels = np.array(dataset[label_field])
            n_per_class = int(len(dataset) * pct / 2)
            idx_0 = np.where(all_labels == 0)[0][:n_per_class].tolist()
            idx_1 = np.where(all_labels == 1)[0][:n_per_class].tolist()
            return dataset.select(idx_0 + idx_1)
        return dataset

    disable_progress_bar()
    if data_name == "imdb":
        dataset = load_dataset("stanfordnlp/imdb")[subset]
        sample = sample_balanced(dataset, pct=pct)
        return sample["text"], sample["label"]

    elif data_name == "yelp":
        dataset = load_dataset("fancyzhx/yelp_polarity")[subset]
        sample = sample_balanced(dataset, pct=pct)
        return sample["text"], sample["label"]
    else:
        raise ValueError(f"Unknown dataset: '{data_name}'. Choose from: 'imdb', 'yelp'.")

def evaluate_fairness(model, tokenizer, pairs, device, batch_size=32):
    model.eval()

    cfs = []
    flip = []
    numerator = []

    with torch.no_grad():
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]

            s1_list = [s1 for s1, _ in batch]
            s2_list = [s2 for _, s2 in batch]

            t1 = tokenizer(
                s1_list,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            t2 = tokenizer(
                s2_list,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(device)

            logits1 = model(**t1).logits
            logits2 = model(**t2).logits
            d = torch.abs(logits1 - logits2).mean(dim=-1)
            f = (logits1.argmax(dim=-1) != logits2.argmax(dim=-1)).float()

            cfs.extend(d.cpu().tolist())
            flip.extend(f.cpu().tolist())
            numerator.extend((d * f).cpu().tolist())

            del t1, t2, logits1, logits2

    if np.sum(numerator) == 0:
        return np.mean(cfs), np.mean(flip), 0, 0

    return np.mean(cfs), np.mean(flip), np.sum(numerator)/np.sum(flip), np.sum(numerator)/np.sum(cfs)

def evaluate_accuracy(model, tokenizer, test_texts, test_labels, device, batch_size=128, max_length=256, cache_path = "test_tokens_cache.pkl"):
    model.eval()
    
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            all_tokens = pickle.load(f)
    else:
        all_tokens = tokenizer(
            test_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        with open(cache_path, "wb") as f:
            pickle.dump(all_tokens, f)

    preds = []
    labels = []
    
    with torch.no_grad():
        for i in range(0, len(test_texts), batch_size):
            batch_tokens = {
                k: v[i:i + batch_size].to(device) 
                for k, v in all_tokens.items()
            }
            
            batch_labels = test_labels[i:i + batch_size]

            outputs = model(**batch_tokens)
            batch_preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()

            preds.extend(batch_preds)
            labels.extend(batch_labels)

    acc = accuracy_score(labels, preds)
    f1  = f1_score(labels, preds)
    return acc, f1

def evaluate_model(model, tokenizer, test_texts, test_labels, bias_pairs, batch_size=64, max_length=256, cache_path = "test_tokens_cache.pkl"):
    device = next(model.parameters()).device    

    print("Evaluating accuracy...")
    acc, f1 = evaluate_accuracy(
        model=model,
        tokenizer=tokenizer,
        test_texts=test_texts,
        test_labels=test_labels,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        cache_path = cache_path
    )
    
    print("Evaluating fairness...")
    cfs, flip, w_cfs, w_flip = evaluate_fairness(
        model=model,
        tokenizer=tokenizer,
        pairs=bias_pairs,
        device=device,
    )

    return {
        "accuracy": acc,
        "f1": f1,
        "cfs": cfs,
        "flip": flip,
        "w_cfs": w_cfs,
        "w_flip": w_flip
    }

