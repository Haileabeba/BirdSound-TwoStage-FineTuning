# Progressive HuBERT

Multi-label species classification from bioacoustic audio using [HuBERT](https://huggingface.co/facebook/hubert-base-ls960). This project compares three training strategies for detecting species in **overlapping** (multi-species) recordings.

## Experiments

| Experiment | Training data | Evaluation data | Purpose |
|------------|---------------|-----------------|---------|
| **single** | Single-species (80% train, 20% val) | Full overlapping set | Can clean single-species pretraining generalize to mixed audio? |
| **overlap** | Overlapping (80% train, 20% test) | Held-out overlapping | Baseline when only mixed clips are available |
| **progressive** | Phase 1: single-species → Phase 2: fine-tune on overlapping | Held-out overlapping | Two-stage curriculum learning |

All experiments use multi-label classification (`BCEWithLogitsLoss`) and report **loss**, **micro/macro F1**, and **Hamming loss**.

## Dataset layout

Place your audio under two roots (paths are configurable):

**Single-species** — one species per file:

```
SingleSpecies/
├── SpeciesA/
│   ├── clip001.wav
│   └── clip002.flac
└── SpeciesB/
    └── clip003.wav
```

**Overlapping** — multiple species per clip; folder name lists species (underscore-separated):

```
OverlappingSpecies/
├── SpeciesA_SpeciesB/
│   ├── mix001.wav
│   └── mix002.wav
└── SpeciesA_SpeciesC_SpeciesD/
    └── mix003.flac
```

Species names in folder titles may include parenthetical notes; text before `(` is used as the label (e.g. `Robin (dawn)` → `Robin`).

## Installation

```bash
cd "progressive project"
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

GPU is recommended; CPU training is supported but slow.

## Configuration

1. Copy the example config:
   ```bash
   cp configs/local.example.yaml configs/local.yaml
   ```
2. Edit `configs/local.yaml` and set `single_species_path` and `overlapping_species_path`.

Or pass paths on the command line (overrides config):

```bash
python -m progressive_hubert.cli \
  --experiment progressive \
  --single-path /path/to/SingleSpecies \
  --overlap-path /path/to/OverlappingSpecies \
  --output-dir outputs
```

## Usage

### Command line

```bash
# Single-species training, test on overlapping
python -m progressive_hubert.cli --experiment single --config configs/local.yaml

# Overlap-only training
python -m progressive_hubert.cli --experiment overlap --config configs/local.yaml

# Progressive (recommended): pretrain then fine-tune
python -m progressive_hubert.cli --experiment progressive --config configs/local.yaml
```

Optional flags: `--epochs`, `--batch-size`, `--learning-rate`, `--seed`, `--output-dir`.

### Shell helpers

```bash
chmod +x scripts/*.sh
./scripts/run_progressive.sh
```

## Outputs

Checkpoints and label maps are written under `outputs/`:

```
outputs/
├── single_species/
│   ├── best_single_species.pt
│   └── species_labels.txt
├── overlap_only/
│   └── best_overlap_only.pt
└── progressive/
    ├── best_single_species.pt      # phase 1
    └── best_progressive.pt       # phase 2 (final)
```

`species_labels.txt` maps class indices to species names for inference.

## Project structure

```
progressive project/
├── progressive_hubert/   # Python package
│   ├── audio.py        # Loading & HuBERT features
│   ├── cli.py          # Entry point
│   ├── data.py         # Dataset indexing & labels
│   ├── dataset.py      # PyTorch Dataset
│   ├── experiments.py
│   └── train.py        # Training loop & metrics
├── configs/
├── scripts/
├── data/               # Optional local data (gitignored)
├── requirements.txt
└── README.md
```

## Publish to GitHub

From this folder:

```bash
cd "/media/rtr-system/Hubert/progressive project"
git init
git add .
git commit -m "Initial release: progressive HuBERT multi-label species classification"
# Create an empty repo on GitHub, then:
git remote add origin https://github.com/YOUR_USER/progressive-hubert.git
git branch -M main
git push -u origin main
```

Do not commit large audio datasets or `.pt` checkpoints; they are listed in `.gitignore`.

## Citation



## License

MIT — see [LICENSE](LICENSE).
