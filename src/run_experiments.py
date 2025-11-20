"""Small runner to run train pipeline and compare saved results.
This script is a convenience wrapper and prints the results.json content.
"""
from pathlib import Path
import json
import subprocess
import sys


def main():
    # Run the pipeline (this will create artifacts/ and results.json)
    print("Running training pipeline...")
    subprocess.check_call([sys.executable, "-m", "src.train_pipeline"]) 

    results_path = Path("results.json")
    if results_path.exists():
        print("Results:")
        print(results_path.read_text())
    else:
        print("No results.json found.")


if __name__ == '__main__':
    main()
