import subprocess
import sys
import time
import logging
import os
from pathlib import Path
from datetime import datetime

# Add project root to sys.path to allow importing src.core_config
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core_config import LOG_DIR, get_venv_python

# --- Logging Setup ---
log_file = LOG_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_script(script_path, name):
    logging.info(f"Running Step: {name} ({script_path})")
    python_exe = get_venv_python()
    t0 = time.time()
    
    try:
        # Use subprocess to run the script
        result = subprocess.run(
            [python_exe, str(script_path)],
            check=True,
            capture_output=False, # Show output in real-time
            text=True
        )
        duration = time.time() - t0
        logging.info(f"Step '{name}' COMPLETED in {duration:.2f}s")
        return True, duration
    except subprocess.CalledProcessError as e:
        logging.error(f"Step '{name}' FAILED with exit code {e.returncode}")
        return False, 0
    except Exception as e:
        logging.error(f"Step '{name}' FAILED with error: {e}")
        return False, 0

def main():
    logging.info("="*60)
    logging.info("STARTING DROWSINESS DETECTION PIPELINE")
    logging.info(f"Log file: {log_file}")
    logging.info("="*60)

    # Pipeline definition
    steps = [
        (PROJECT_ROOT / 'Frame_exrtaction' / '4fps.py', 'Frame Extraction & CLAHE'),
        (PROJECT_ROOT / 'frame' / 'Mesh_apply.py',      'Face Mesh & Landmark Generation'),
        (PROJECT_ROOT / 'to_csv.py',                    'Feature Extraction & Smoothing'),
        (PROJECT_ROOT / 'src' / 'calibration.py',       'Calibration (Dynamic Alpha)'),
        (PROJECT_ROOT / 'src' / 'eye_state.py',         'Eye State Calculation B(t)'),
        (PROJECT_ROOT / 'src' / 'analyze_failures.py',  'Failure Analysis Report')
    ]

    total_duration = 0
    success_count = 0

    for script, name in steps:
        if not script.exists():
            logging.error(f"Script not found: {script}")
            break
            
        success, duration = run_script(script, name)
        if not success:
            break
            
        success_count += 1
        total_duration += duration
        logging.info("-" * 40)

    logging.info("="*60)
    if success_count == len(steps):
        logging.info(f"PIPELINE COMPLETED SUCCESSFULLY ({success_count}/{len(steps)} steps)")
        logging.info(f"Total execution time: {total_duration/60:.2f} minutes")
    else:
        logging.info(f"PIPELINE FAILED at step {success_count + 1}/{len(steps)}")
        logging.info(f"Total execution time: {total_duration/60:.2f} minutes")
    logging.info("="*60)

if __name__ == '__main__':
    main()
