"""
DeepDrive Project: Pipeline Orchestrator
===============================================
This is the master orchestrator script for the DeepDrive project. It links the three core 
modules (Data Preparation, Model Training, and Evaluation) into a single continuous pipeline.

Usage:
  python run_pipeline.py

After completion, the trained models are ready for real-time inference via:
  python step3_score.py path/to/session.csv   <- score a new session
  streamlit run app.py                         <- launch the web dashboard
"""

# Orchestration script: This runs the entire deep learning pipeline sequentially.
# Automating this process ensures consistency across data prep, training, and evaluation.
import subprocess
import sys
import os


def run(script, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        print(f"\n  ✗ FAILED at {script}")
        print(f"    Fix the error above, then re-run: python {script}")
        sys.exit(1)
    print(f"  ✓ Done")


def main():
    # Make sure we're in the script's directory
    # Enforce execution from the script's directory to prevent relative path resolution errors
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    for d in ['data', 'models', 'outputs']:
        os.makedirs(d, exist_ok=True)

    run('step0_prepare.py',  'Step 0 — Prepare data sequences')
    run('step1_train.py',    'Step 1 — Train LSTM / GRU / Transformer')
    run('step2_evaluate.py', 'Step 2 — Evaluate models + generate charts')

    print(f"\n{'='*55}")
    print("  PIPELINE COMPLETE")
    print(f"{'='*55}")
    print("\n  Score a new session:")
    print("    python step3_score.py path/to/session.csv")
    print("\n  Launch dashboard:")
    print("    streamlit run app.py")
    print(f"{'='*55}\n")


if __name__ == '__main__':
    main()
