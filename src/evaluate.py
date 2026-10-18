import argparse

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

from src.inference import CodeAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the saved model on the held-out split of a PrimeVul CSV."
    )
    parser.add_argument(
        "--csv",
        default="data/primevul_sample.csv",
        help="CSV with func and target columns.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--threshold",
        type=float,
        help="Decision threshold (defaults to the analyzer's configured threshold).",
    )
    args = parser.parse_args()

    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")

    data = pd.read_csv(args.csv)
    missing_columns = {"func", "target"} - set(data.columns)
    if missing_columns:
        parser.error(f"CSV is missing required columns: {', '.join(sorted(missing_columns))}")
    data = data.dropna(subset=["func", "target"])
    labels = data["target"].astype(int).to_numpy()
    codes = data["func"].astype(str).tolist()
    _, temporary_codes, _, temporary_labels = train_test_split(
        codes,
        labels,
        test_size=0.3,
        stratify=labels,
        random_state=42,
    )
    _, test_codes, _, test_labels = train_test_split(
        temporary_codes,
        temporary_labels,
        test_size=0.5,
        stratify=temporary_labels,
        random_state=42,
    )

    analyzer = CodeAnalyzer(threshold=args.threshold)
    probabilities = analyzer.predict_probabilities(test_codes, batch_size=args.batch_size)
    predictions = [int(probability >= analyzer.threshold) for probability in probabilities]
    print(f"Evaluated {len(test_labels)} samples on {analyzer.device}.")
    print(f"Decision threshold: {analyzer.threshold:.2f}")
    print(classification_report(test_labels, predictions, target_names=["Safe", "Vulnerable"]))
    print(f"ROC AUC: {roc_auc_score(test_labels, probabilities):.4f}")


if __name__ == "__main__":
    main()