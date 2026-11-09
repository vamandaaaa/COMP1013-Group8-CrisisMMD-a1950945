# Multimodal Humanitarian Classification Using CrisisMMD

COMP1013 Multi-Modal Data Analysis — Group 8

## Project Overview

This project investigates whether combining information from social media text and images improves the classification of disaster-related posts.

We use the CrisisMMD dataset to classify text-image pairs into five humanitarian categories. Our original benchmark compares two unimodal baselines with two multimodal fusion approaches:

1. Text-only classification using BERT features.
2. Image-only classification using EfficientNet-B0 features.
3. Early fusion using concatenated text and image features.
4. Late fusion using weighted prediction probabilities.

The objective is to evaluate whether multimodal information improves classification performance, particularly when humanitarian categories are imbalanced.

## Dataset

We use the five-class humanitarian task with agreed text-image labels from CrisisMMD.

The classification categories are:

- Affected individuals
- Infrastructure and utility damage
- Not humanitarian
- Other relevant information
- Rescue, volunteering or donation effort

The original dataset contains 8,079 agreed-label samples.

After conservative data cleaning, 8,038 samples remain.

| Split | Original | Cleaned |
|---|---:|---:|
| Train | 6,126 | 6,093 |
| Development | 998 | 994 |
| Test | 955 | 951 |
| Total | 8,079 | 8,038 |

The cleaning process excluded 18 visually verified placeholder images and 23 additional records involved in cross-split exact-pixel duplication.

Potential near-duplicates that could not be confidently identified as exact duplicates were retained and documented as a limitation.

The original images, metadata and pretrained feature caches are not included in this repository.

## Methodology

### Feature Extraction

The original benchmark uses pretrained, frozen feature representations provided through the group cache:

| Modality | Encoder | Feature dimension |
|---|---|---:|
| Text | BERT | 768 |
| Image | EfficientNet-B0 | 1,280 |

We did not fine-tune the pretrained encoders.

Features are standardised using statistics fitted on the training data only.

### Classification Models

Each trained classifier uses a feed-forward neural network containing:

- An input layer matching the feature dimension.
- A hidden layer with 128 neurons.
- ReLU activation.
- Dropout regularisation.
- A final layer predicting five classes.

We use Adam optimisation, class-weighted cross-entropy loss and early stopping based on development macro-F1.

The two fusion approaches are:

**Early fusion:** The 768-dimensional BERT features and 1,280-dimensional EfficientNet-B0 features are concatenated into one 2,048-dimensional vector before classification.

**Late fusion:** Separate text-only and image-only classifiers generate class probabilities. The probabilities are combined using a weight selected on the development set.

## Hyperparameter Experiments

We evaluated six combinations of learning rate and dropout.

| Experiment | Dropout | Learning rate |
|---|---:|---:|
| E1 | 0.3 | 0.001 |
| E2 | 0.3 | 0.0003 |
| E3 | 0.3 | 0.003 |
| E4 | 0.5 | 0.001 |
| E5 | 0.5 | 0.0003 |
| E6 | 0.5 | 0.003 |

The other settings were held constant:

- Hidden units: 128
- Batch size: 128
- Maximum epochs: 35
- Early stopping patience: 6
- Random seed: 42
- Device: CPU

The best configuration for each approach was selected using development macro-F1.

## Original Baseline Test Results

The selected models were evaluated on 951 cleaned test samples.

| Model | Selected experiment | Test accuracy | Test macro-F1 |
|---|---|---:|---:|
| Text-only | E5 | 0.7929 | 0.7456 |
| Image-only | E3 | 0.7918 | 0.7119 |
| Early fusion | E6 | 0.8675 | 0.8003 |
| **Late fusion** | **E4** | **0.8759** | **0.8470** |

Late fusion achieved the highest observed test macro-F1.

Compared with the strongest unimodal baseline, text-only, late fusion improved macro-F1 by approximately 0.1015 (10.15 percentage points).

The 95% paired bootstrap confidence interval for this difference was [0.0445, 0.1546].

The observed test difference between late fusion and early fusion was not statistically conclusive.

The evaluation also identified class imbalance, overfitting and variation in performance across disaster events.

For complete experimental tables, learning curves, confusion matrices, confidence intervals and discussion, see:

**[Detailed Experimental Results](RESULTS.md)**

## Repository Structure

```text
.
├── README.md
├── RESULTS.md
├── requirements.txt
├── .gitignore
├── crisismmd_audit.py
├── visual_review.py
├── prepare_clean_splits.py
├── train_models.py
├── evaluate_models.py
├── results/
│   ├── data_quality/
│   ├── experiments/
│   └── evaluation/
└── figures/
    ├── d03_lr001/
    ├── d05_lr001/
    ├── d05_lr003/
    └── test_confusion_matrices.png
```

Dataset files, model checkpoints, cached embeddings and individual predictions are stored locally rather than committed to GitHub.

## Environment

The implementation uses Python 3.13 and the following main libraries:

- NumPy
- PyTorch
- scikit-learn
- SciPy
- Matplotlib
- Pillow
- torchvision
- Transformers
- safetensors

Install the required Python packages using:

```bash
python -m pip install -r requirements.txt
```

## Reproduction Instructions

The commands below assume that the required dataset and trusted group feature cache have been obtained separately.

Place the following archives in the project root:

```text
splits.zip
crisismmd_group_share.zip
```

Place the CrisisMMD image directory under:

```text
data/raw/data_image/
```

The group feature cache must contain the pretrained BERT and EfficientNet-B0 feature arrays. Additional scripts now extract frozen ResNet-50 and ViT image features and BERT token features. The original BERT and EfficientNet-B0 group cache is still needed for the baseline.

### 1. Audit the Dataset

```bash
python crisismmd_audit.py \
  --group-zip crisismmd_group_share.zip \
  --splits-zip splits.zip \
  --out audit_output
```

### 2. Generate Visual Review Sheets

```bash
python visual_review.py \
  --splits-zip splits.zip \
  --audit-dir audit_output \
  --images-root data/raw \
  --out audit_output/visual_review
```

Visual inspection is required to confirm the placeholder groups before applying the cleaning policy.

### 3. Prepare Cleaned Splits

```bash
python prepare_clean_splits.py \
  --splits-zip splits.zip \
  --audit-dir audit_output \
  --images-root data/raw \
  --out data/processed \
  --report-dir results/data_quality
```

### 4. Train the Models

For each experiment, run:

```bash
python train_models.py \
  --group-zip crisismmd_group_share.zip \
  --processed-dir data/processed \
  --results-dir results/experiments/d03_lr001 \
  --models-dir models/d03_lr001 \
  --figures-dir figures/d03_lr001 \
  --epochs 35 \
  --patience 6 \
  --batch-size 128 \
  --hidden 128 \
  --dropout 0.3 \
  --lr 0.001 \
  --seed 42 \
  --device cpu
```

This example runs E1. Repeat the training with the dropout, learning rate and folder name corresponding to E2–E6.

Each experiment saves:

- Model checkpoints.
- Development results.
- Training history.
- Training and validation curves.
- Feature standardisation statistics.

The test split is not accessed during training.

### 5. Evaluate the Selected Models

Once all six experiments have been completed:

```bash
python evaluate_models.py
```

The evaluation script loads:

- Text-only from E5.
- Image-only from E3.
- Early fusion from E6.
- Late fusion from E4.

Late fusion uses both unimodal checkpoints from E4 and their development-selected probability weights.

The script produces:

```text
results/evaluation/test_results.json
figures/test_confusion_matrices.png
data/evaluation/test_predictions.csv
```

Individual test predictions are retained locally.

## Extended Experiments (Train/Dev Only)

Following the original baseline, we conducted additional
experiments using 6,093 cleaned training and 994 development
pairs. The original official test was not accessed in these
additional experiments.

### Models and methods

- Image encoder comparison: EfficientNet-B0, ResNet-50,
  and ViT-B/16 with frozen pooled BERT representations.
- Cross-attention: BERT token representations attend to
  ViT image patches using PyTorch MultiheadAttention.
- Pooled fusion: averaged BERT tokens and ViT patches,
  without cross-attention.
- Modality dropout: randomly zeroed text or image features
  during training, with probability 0.15 per modality.
- Missing-modality evaluation: development stress tests
  using zeroed text or image features.
- Leave-one-event-out evaluation: disaster events held out
  from fold fitting using cleaned train/dev data.

### Development macro-F1

The following values are means across seeds 42, 43 and 44.

| Model | Mean macro-F1 |
|---|---:|
| ResNet-50 early fusion | 0.8102 |
| ViT-B/16 early fusion | 0.8053 |
| EfficientNet-B0 early fusion | 0.7888 |
| Token/patch pooled fusion | 0.7827 |
| Pooled fusion with modality dropout | 0.7536 |
| Cross-attention | 0.7487 |

These results are not official test scores. The encoder
comparison and token/patch models also use different text
representations and training settings, so this table is
not a fully controlled architectural comparison.

### Unseen-event experiment

Leave-one-event-out pooled macro-F1:

- Text-only: 0.6410
- Image-only: 0.6123
- Early fusion: 0.7499
- Late fusion: 0.7014

This experiment holds events out within the train/dev
portion. It is not a completely untouched external
benchmark because these data had been used during
earlier project development.

### Additional Python scripts

- extract_image_features.py
- extract_text_tokens.py
- compare_image_encoders.py
- train_cross_attention.py
- train_pooled_fusion.py
- train_modality_dropout.py
- evaluate_missing_modality.py
- evaluate_dropout_robustness.py
- evaluate_unseen_events.py

Additional feature extraction examples:

```bash
python extract_image_features.py --model resnet50 --device mps
python extract_image_features.py --model vit_b_16 --device mps
python extract_text_tokens.py --device mps
python compare_image_encoders.py --device mps --seed 42
```

These commands require the cleaned splits and local source
data. Use CPU instead of MPS on unsupported computers.

Training checkpoints, original posts, images and feature
caches remain local and are not distributed on GitHub.

The original baseline test results above were obtained
earlier. The test data had already been inspected, so they
should not be described as a pristine blind evaluation.

## Limitations

The results should be interpreted with several limitations in mind:

- The dataset contains substantial class imbalance.
- The affected-individuals class has only nine test examples.
- Some unresolved visually similar images may remain across splits.
- The models use frozen pretrained features rather than end-to-end encoder fine-tuning.
- Original baseline hyperparameter experiments used one seed. The later encoder and fusion experiments used seeds 42, 43 and 44.
- Original per-event breakdowns were descriptive. A separate leave-one-event-out experiment was subsequently conducted using train/dev data.
- The test split had been inspected in an earlier project run and is therefore not a completely untouched blind test.

Additional image encoders, cross-attention, and leave-one-event-out evaluation were subsequently implemented and evaluated on the cleaned train/dev data.

The findings support the usefulness of multimodal feature fusion on the evaluated CrisisMMD splits, but do not establish generalisation to all future disasters.

## Ethical Considerations

Disaster-related social media content may contain sensitive personal information or reflect unequal representation across locations, languages and communities.

The system should not be used as an autonomous humanitarian decision-making tool. False negatives may cause urgent information to be missed, while false positives can create unnecessary alerts.

Model outputs require appropriate human review before operational use.

The repository focuses on scripts and aggregate experimental results rather than redistributing original social media content and images.

## Further Information

See [RESULTS.md](RESULTS.md) for the full experimental results, statistical comparisons, visualisations and interpretation.
