import os
import glob
import json
import argparse
import pandas as pd

def compile_json_summaries_to_csv(runs_dir: str, output_csv: str):
    """Recursively finds all summary.json files in runs_dir and compiles all keys into a CSV."""
    runs_dir = os.path.abspath(runs_dir)
    json_pattern = os.path.join(runs_dir, "**", "summary.json")
    summary_files = glob.glob(json_pattern, recursive=True)

    if not summary_files:
        print(f"No summary.json files found under '{runs_dir}'")
        return

    print(f"Found {len(summary_files)} summary.json file(s) in {runs_dir}. Processing...")

    records = []
    for i, json_path in enumerate(summary_files, 1):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Append raw dictionary directly to preserve all key-value pairs
            records.append(data)
            print(f"  [{i}/{len(summary_files)}] Parsed: {json_path}")

        except Exception as e:
            print(f"  [{i}/{len(summary_files)}] Error reading {json_path}: {e}")

    if not records:
        print("No valid summary records were extracted.")
        return

    # Pandas maps dictionary keys directly to columns
    df = pd.DataFrame(records)

    output_abs = os.path.abspath(output_csv)
    os.makedirs(os.path.dirname(output_abs), exist_ok=True)
    
    df.to_csv(output_abs, index=False)
    print(f"\nCompilation complete! Exported {len(df)} row(s) with {len(df.columns)} column(s) to '{output_abs}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile all summary.json files and all keys into a single CSV.")
    parser.add_argument("--runs_dir", default="./runs", help="Root directory to search for summary.json files")
    parser.add_argument("--output", default="compiled_summary.csv", help="Destination path for the generated CSV")
    
    args = parser.parse_args()
    compile_json_summaries_to_csv(args.runs_dir, args.output)
