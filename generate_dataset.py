import argparse
import itertools
import json
import os
import random
from itertools import combinations
from typing import Dict, List, Tuple

IDENTITIES: Dict[str, List[str]] = {
    "gender":             ["female", "male", "nonbinary"],
    "age":                ["young", "middle-aged", "elderly"],
    "religion":           ["Muslim", "Christian", "Jewish", "Hindu", "Buddhist"],
    "disability":         ["disabled", "non-disabled"],
    "sexual_orientation": ["gay", "straight", "bisexual"],
    "ethnicity":          ["White", "Black", "Latino", "Asian"],
}

TEMPLATE_FILES: Dict[str, str] = {
    "imdb": "imdb_templates.txt",
    "yelp": "yelp_templates.txt",
}

def generate_all_identity_pairs(identities_dict: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    all_pairs = []
    for cat, ids in identities_dict.items():
        pairs = list(itertools.combinations(ids, 2))
        all_pairs.extend(pairs)
    return all_pairs

def split_templates(
    templates: List[str],
    calibration_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[List[str], List[str]]:
    random.seed(seed)
    shuffled = templates.copy()
    random.shuffle(shuffled)
    split_idx = int(len(shuffled) * calibration_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]

def generate_split_dataset(
    identities_dict: Dict[str, List[str]],
    templates: List[str],
    templates_per_pair: int,
    seed: int = 42
) -> List[Dict]:

    random.seed(seed)
    identity_pairs = generate_all_identity_pairs(identities_dict)
    dataset = []

    for pair in identity_pairs:
        identity_a, identity_b = pair

        selected_templates = random.sample(
            templates, min(templates_per_pair, len(templates))
        )

        for template in selected_templates:
            s1 = template.format(identity=identity_a)
            s2 = template.format(identity=identity_b)
            dataset.append({
                "identity_pair": pair,
                "template": template,
                "s1": s1,
                "s2": s2
            })

    return dataset

def build_dataset(
    identities_dict: Dict[str, List[str]],
    templates: List[str],
    templates_per_pair: int = 100,
    calibration_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:

    calibration_templates, test_templates = split_templates(
        templates,
        calibration_ratio,
        seed
    )

    calibration_set = generate_split_dataset(
        identities_dict,
        calibration_templates,
        templates_per_pair,
        seed
    )

    test_set = generate_split_dataset(
        identities_dict,
        test_templates,
        templates_per_pair,
        seed + 1
    )

    return calibration_set, test_set

def save_to_json(data: List[Dict], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def filter_by_bias_categories( dataset: List[Dict], identities_dict: Dict[str, List[str]], output_dir: str,  split_name: str,
) -> Dict[str, List[Dict]]:
    
    results: Dict[str, List[Dict]] = {}
 
    for category, values in identities_dict.items():
        target_pairs = {frozenset(pair) for pair in combinations(values, 2)}
 
        filtered = [
            entry for entry in dataset
            if frozenset(entry["identity_pair"]) in target_pairs
        ]
 
        output_path = os.path.join(output_dir, f"{category}_{split_name}_set.json")
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(filtered, f, indent=2, ensure_ascii=False)
 
        n_pairs = len(list(combinations(values, 2)))
        print(
            f"  [{category:<20}]  {n_pairs} pairs │ "
            f"{len(filtered):>5} / {len(dataset)} entries  →  {output_path}"
        )
        results[category] = filtered
 
    return results

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a bias-analysis dataset (calibration + test).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", choices=["imdb", "yelp"], required=True)
    parser.add_argument("--templates_per_pair", type=int, default=15)
    parser.add_argument("--calibration_ratio", type=float, default=0.5)
    return parser.parse_args()

def main() -> None:
    args = parse_args()

    dataset_name  = args.dataset
    template_file = TEMPLATE_FILES[dataset_name]
    output_dir    = dataset_name

    with open(template_file, "r", encoding="utf-8") as f:
        templates = [line.strip().strip('",') for line in f if line.strip()]

    calibration_set, test_set = build_dataset(
        IDENTITIES, templates,
        templates_per_pair=args.templates_per_pair,
        calibration_ratio=args.calibration_ratio,
        seed=42,
    )
    
    save_to_json(calibration_set, os.path.join(output_dir, "calibration_set.json"))
    save_to_json(test_set,        os.path.join(output_dir, "test_set.json"))

    print(f"[{dataset_name.upper()}] {len(calibration_set)} calibration | {len(test_set)} test entries saved to '{output_dir}/'")

    filter_by_bias_categories(test_set, IDENTITIES, output_dir, "test")

    print(f" Dataset '{dataset_name.upper()}' successfully generated. All files saved to '{output_dir}/'.")

if __name__ == "__main__":
    main()