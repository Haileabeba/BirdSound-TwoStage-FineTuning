# procomment5foldcross

Extension of **procomment** (reviewer response: fair optimizer-step comparison) with **5-fold cross-validation** for all three HuBERT fine-tuning strategies.

## What this adds beyond procomment

| Feature | procomment | **procomment5foldcross** |
|---------|------------|---------------------------|
| Single train/val split | Yes | No — **5 folds** |
| Strategies | single, overlap, progressive | Same, **each fold** |
| Matched optimizer updates | Yes | Yes (per fold) |
| Plots vs updates | Yes | Per-fold + **mean ± std bar chart** |
| Robust metrics | One run | **mean ± std** over 5 folds |

## Cross-validation design

Each fold `k = 1…5`:

1. **Single-species** data → 5-fold split → train / val  
2. **Overlapping** data → 5-fold split → train / test  
3. Train all three strategies with **same total optimizer steps** (when `match_total_updates: true`)  
4. Evaluate on the **overlap test fold** (fair comparison on mixed audio)

```
Fold k:
  single-species  : train(single_train_k) → test(overlap_test_k)
  overlap-only    : train(overlap_train_k) → test(overlap_test_k)
  progressive     : phase1(single) → phase2(overlap_train_k) → test(overlap_test_k)
```

## Quick start

```bash
cd "/media/rtr-system/0924507943/Hubert/procomment5foldcross"
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

cp configs/local.example.yaml configs/local.yaml
# Edit single_species_path and overlapping_species_path

# Full 5-fold CV (long — GPU recommended)
python -m progressive_hubert.cli --experiment cv5fold --config configs/local.yaml

# Quick smoke test (2 folds, 2 epochs per phase)
python -m progressive_hubert.cli --experiment cv5fold-quick --config configs/local.yaml
```

Windows:

```bat
scripts\run_cv5fold.bat configs\local.yaml
```

## Outputs

```
outputs/cv_5fold/
  fold_0/ ... fold_4/
    single_species/history.json
    overlap_only/history.json
    progressive/history.json
    fold_summary.json
  cv_aggregate.json
  cv_results_5fold.csv
  figures/
    fold_0/ ... fold_4/     # loss & accuracy vs updates
    cv5_aggregate_metrics.png   # mean ± std across folds
```

## Config (`configs/cv5fold.yaml`)

| Key | Default | Meaning |
|-----|---------|---------|
| `n_folds` | 5 | Number of CV folds |
| `match_total_updates` | true | Fair step budget (procomment) |
| `phase1_epochs`, `phase2_epochs` | 50 | Define progressive total budget |
| `seed` | 42 | KFold shuffle seed |

## Relation to procomment

Use **procomment** for a single fair comparison and paper Figures 6–7.  
Use **procomment5foldcross** when reviewers or the paper need **cross-validated mean ± std** results.

## License

MIT
