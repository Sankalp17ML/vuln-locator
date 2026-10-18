# data/download.py
from datasets import load_dataset
import pandas as pd

dataset = load_dataset("colin/PrimeVul", split="train")
df = pd.DataFrame(dataset)

print(f"Total samples: {len(df)}")
print(f"Columns available: {df.columns.tolist()}") 

# The label column is likely 'target' for colin/PrimeVul
# We'll dynamically find it to be safe
label_col = 'target' if 'target' in df.columns else 'vul'

print(f"Using label column: '{label_col}'")
print(f"Label distribution:\n{df[label_col].value_counts()}")

# Look at a few vulnerable examples
vuln_examples = df[df[label_col] == 1].head(3)
for i, row in vuln_examples.iterrows():
    print(f"\n=== VULNERABLE EXAMPLE ===")
    print(row['func'][:500])  # The code is stored in the 'func' column

# Save a sample for Phase 2
df.sample(20000, random_state=42).to_csv("data/primevul_sample.csv", index=False)