# 5-fold CV methodology (for manuscript)

## Purpose

Report **mean ± standard deviation** of test metrics across 5 folds, with the same **matched optimizer-update budget** as procomment (reviewer comment on epoch plots).

## Suggested text

> We evaluated each fine-tuning strategy using **5-fold cross-validation**. In each fold, single-species and overlapping recordings were partitioned independently (70% train / 30% hold-out for single val; 80% train / 20% test for overlap). All strategies received the **same total number of AdamW updates** per fold. Progressive training used a two-phase schedule (single-species pre-training, then overlapping fine-tuning) with a continuous global step counter. Results are reported as **mean ± std** over folds. Learning curves are plotted against **optimizer updates**, not epoch index.

## Reporting table

Use `outputs/cv_5fold/cv_results_5fold.csv`:

| Strategy | test_micro_f1_mean | test_micro_f1_std | test_subset_accuracy_mean | ... |

## Figures

- Per-fold: `figures/fold_k/eval_loss_vs_updates.png`
- Aggregate: `figures/cv5_aggregate_metrics.png`
