# MC-QAD — Memory-Constrained Quantization-Aware Debiasing

MC-QAD (Memory-Constrained Quantization-Aware Debiasing) is a unified framework that jointly compresses and debiases language models by optimizing both model weights and a mixed-precision bit allocation policy under a strict memory budget.  

Currently, the framework supports BERT and RoBERTa models, evaluated on the movie review (IMDb) and restaurant review (Yelp) domains.

---

## Requirements

Install the required dependencies:
```bash
pip install torch transformers datasets scikit-learn numpy pandas
```

---

## Project Structure
```
.
├── mc-qad.py               # Main script: trains and evaluates MC-QAD
├── utils.py                # Utility functions (metrics, data loading)
├── generate_dataset.py     # Generates counterfactual bias datasets
├── imdb_templates.txt      # Sentence templates for IMDb
└── yelp_templates.txt      # Sentence templates for Yelp
```

---

## Step 1 — Generate the datasets

Before running MC-QAD, you need to generate the counterfactual datasets for the target domain.  
This script reads sentence templates and generates a calibration set and a test set, then filters the latter by bias category.
```bash
python generate_dataset.py --dataset imdb
python generate_dataset.py --dataset yelp
```

This creates a directory named `imdb/` (or `yelp/`) with the following files:
```
imdb/
├── calibration_set.json
├── test_set.json
├── age_test_set.json
├── disability_test_set.json
├── ethnicity_test_set.json
├── gender_test_set.json
├── religion_test_set.json
└── sexual_orientation_test_set.json
```

The `yelp/` directory mirrors the same structure.

### Parameters

| Parameter | Required | Default | Description |
|---|---|---|---|
| `--dataset` | yes | — | Dataset to use: `imdb` or `yelp` |
| `--templates_per_pair` | no | `15` | Number of templates sampled per identity pair |
| `--calibration_ratio` | no | `0.5` | Fraction of templates used for calibration set |

---

## Step 2 — Run MC-QAD

Once the datasets are generated, you can train and evaluate the MC-QAD method.
```bash
python mc-qad.py --dataset imdb --model bert --bias_category ethnicity
```

### Parameters

| Parameter                 | Required | Default      | Description                                                                                         |
|---------------------------|----------|--------------|-----------------------------------------------------------------------------------------------------|
| `--dataset`               | yes      | —            | Dataset to use: `imdb`, `yelp`                                                                      |
| `--model`                 | yes      | —            | Base model architecture: `bert`, `roberta`                                                          |
| `--bias_category`         | yes      | —            | Bias category to evaluate: `age`, `disability`, `ethnicity`, `gender`, `religion`, `sexual_orientation`, `all`. Use `all` for the full test set |
| `--bitwidth_choices`      | no       | `4 8 16 32`  | Candidate bitwidths for quantization                                                                |
| `--lr_model`              | no       | `1e-6`       | Learning rate for model parameters                                                                  |
| `--lr_alloc`              | no       | `1e-3`       | Learning rate for precision allocation                                                              |
| `--lr_lambda`             | no       | `1e-2`       | Learning rate for Lagrangian multiplier                                                             |
| `--max_epochs`            | no       | `100`        | Maximum number of training epochs                                                                   |
| `--cf_batch_size`         | no       | `16`         | Batch size for counterfactual pairs                                                                 |
| `--reduction_perc`        | no       | `0.7`        | Target bit-complexity reduction percentage                                                          |
| `--beta`                  | no       | `0.1`        | Fairness loss weight                                                                                |
| `--tolerance`             | no       | `1e-2`       | Early stopping tolerance                                                                            |

---

## Examples

**Minimal run:**
```bash
# Step 1 - Generate the datasets
python generate_dataset.py --dataset imdb

# Step 2 - Run MC-QAD
python mc-qad.py --dataset imdb --model bert --bias_category age
```

**Full run with custom parameters:**
```bash
# Step 1 - Generate the datasets
python generate_dataset.py --dataset imdb --templates_per_pair 20 --calibration_ratio 0.6

# Step 2 - Run MC-QAD
python mc-qad.py \
  --dataset imdb \
  --model roberta \
  --bias_category gender \
  --bitwidth_choices 4 8 16 \
  --lr_model 1e-5 \
  --lr_alloc 1e-3 \
  --lr_lambda 1e-2 \
  --max_epochs 50 \
  --cf_batch_size 16 \
  --reduction_perc 0.7 \
  --beta 0.2 \
  --tolerance 1e-2 \
```
