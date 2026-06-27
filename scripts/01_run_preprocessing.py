import sys
from pathlib import Path

# Add src to python path so we can import sleep_next
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import argparse
import time
from sleep_next.data import loader, preprocess

def main():
    parser = argparse.ArgumentParser(description="Run preprocessing for sleep classification dataset.")
    parser.add_argument("--subject", type=str, default=None, help="Subject ID to preprocess. If omitted, all subjects are preprocessed.")
    args = parser.parse_args()
    
    start_time = time.time()
    
    if args.subject:
        subjects = [args.subject]
    else:
        subjects = loader.get_all_subject_ids()
        
    print(f"Starting preprocessing for {len(subjects)} subjects...")
    for idx, subject_id in enumerate(subjects, 1):
        print(f"[{idx}/{len(subjects)}] Cropping data from subject {subject_id}...")
        try:
            preprocess.crop_subject_data(subject_id)
            print(f"[{idx}/{len(subjects)}] Building features for subject {subject_id}...")
            preprocess.build_features_for_subject(subject_id)
        except Exception as e:
            print(f"Error preprocessing subject {subject_id}: {e}", file=sys.stderr)
            
    end_time = time.time()
    print(f"Preprocessing completed in {(end_time - start_time) / 60:.2f} minutes.")

if __name__ == "__main__":
    main()
