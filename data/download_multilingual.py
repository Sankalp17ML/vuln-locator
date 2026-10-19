"""
Multilingual Vulnerability Dataset Acquisition Script (CrossVul)
================================================================
Downloads and processes the multilingual CrossVul benchmark across
Java, Python, PHP, JavaScript, Ruby, Go, and C#.

Formats data to match the VulnLocator standard schema:
- 'func': Source code of the function / snippet
- 'target': Binary vulnerability label (1 = Vulnerable, 0 = Safe/Patched)
- 'language': Detected / labeled programming language
- 'cwe_id': CWE identifier (if available)

Outputs:
data/multilingual_sample.csv
"""

import os
import re
import argparse
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq


TARGET_LANGUAGES = {
    "java",
    "python",
    "php",
    "javascript",
    "js",
    "ruby",
    "go",
    "csharp",
}


def download_and_format_crossvul(
    output_path: str = "data/multilingual_sample.csv",
    sample_size: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    print("=" * 65)
    print("VulnLocator V2.0: Multilingual Dataset Pipeline (CrossVul)")
    print("=" * 65)

    print("\nFetching CrossVul dataset partitions from Hugging Face...")
    rows = []

    # Download CrossVul partition 1
    try:
        parquet_path = hf_hub_download(
            repo_id="hitoshura25/crossvul",
            filename="data/train-00000-of-00002.parquet",
            repo_type="dataset",
        )
        print(f"Loaded CrossVul partition from: {parquet_path}")
        table = pq.read_table(parquet_path)
        df_raw = table.to_pandas()

        for _, row in df_raw.iterrows():
            lang = str(row.get("language", "")).lower().strip()
            # Standardize js -> javascript
            if lang == "js":
                lang = "javascript"

            # Filter for non-C systems languages (Generalist target languages)
            if lang in TARGET_LANGUAGES:
                vuln_code = str(row.get("vulnerable_code", "")).strip()
                fixed_code = str(row.get("fixed_code", "")).strip()
                cwe = str(row.get("cwe_id", "CWE-Unknown"))

                if vuln_code and len(vuln_code) > 20:
                    rows.append({
                        "func": vuln_code,
                        "target": 1,
                        "language": lang,
                        "cwe": cwe,
                    })

                if fixed_code and len(fixed_code) > 20:
                    rows.append({
                        "func": fixed_code,
                        "target": 0,
                        "language": lang,
                        "cwe": cwe,
                    })

    except Exception as e:
        print(f"Notice during partition 1 fetch: {e}")

    # Also augment with xin1997 Java & Python CrossVul subsets if available
    for subset_lang, repo in [
        ("java", "xin1997/crossvul-java_all_only_input"),
        ("python", "xin1997/crossvul-python_all_only_input"),
    ]:
        try:
            print(f"Augmenting with {subset_lang.upper()} CrossVul subset from {repo}...")
            sub_path = hf_hub_download(
                repo_id=repo,
                filename="data/train-00000-of-00001.parquet",
                repo_type="dataset",
            )
            df_sub = pq.read_table(sub_path).to_pandas()
            for _, row in df_sub.iterrows():
                code = str(row.get("content", "")).strip()
                row_id = str(row.get("id", ""))
                # in CrossVul naming, bad = vulnerable (1), good = patched (0)
                is_vuln = 1 if "_bad_" in row_id else 0
                if code and len(code) > 20:
                    rows.append({
                        "func": code,
                        "target": is_vuln,
                        "language": subset_lang,
                        "cwe": "CWE-CrossVul",
                    })
        except Exception as e:
            print(f"Notice during {subset_lang} augmentation: {e}")

    df = pd.DataFrame(rows)
    print(f"\nExtracted total multilingual samples: {len(df):,}")

    if df.empty:
        raise RuntimeError("No multilingual samples could be extracted.")

    # Deduplicate and clean
    df = df.drop_duplicates(subset=["func"])
    df = df.dropna(subset=["func", "target"])
    df["target"] = df["target"].astype(int)

    # Sample if greater than requested sample_size
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=seed)

    print("\nMultilingual Sample Distribution:")
    lang_dist = df.groupby(["language", "target"]).size().unstack(fill_value=0)
    lang_dist.columns = ["Safe (0)", "Vulnerable (1)"]
    print(lang_dist)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nSaved cleaned multilingual dataset to: {output_file.resolve()}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download multilingual CrossVul dataset.")
    parser.add_argument("--output", type=str, default="data/multilingual_sample.csv")
    parser.add_argument("--sample_size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    download_and_format_crossvul(output_path=args.output, sample_size=args.sample_size, seed=args.seed)
