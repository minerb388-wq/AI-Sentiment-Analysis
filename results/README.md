# Evaluation outputs

Run the following command from the repository root to generate reproducible
model-comparison outputs:

```powershell
python evaluate_models.py
```

The script creates:

- `model_comparison.csv` — a table suitable for an assignment report.
- `model_comparison.json` — the same results for reproducible downstream analysis.

Each category is evaluated with the same stratified 80/20 split and compares:

1. A majority-class baseline.
2. Multinomial Naive Bayes.
3. Logistic Regression.

Metrics include accuracy and weighted precision, recall, and F1 score.
