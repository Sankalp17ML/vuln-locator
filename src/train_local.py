from pathlib import Path
from runpy import run_path


if __name__ == "__main__":
    training_script = Path(__file__).with_name("train_baseline.py")
    run_path(str(training_script), run_name="__main__")