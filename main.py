import subprocess
import sys
import time
import logging
import os
from pathlib import Path
from datetime import datetime

# --- Configuration ---
PROJECT_ROOT = Path(r'E:\Buồn ngủ')
LOG_DIR = PROJECT_ROOT / 'logs'
LOG_DIR.mkdir(exist_ok=True)

# Setup logging
now_str = datetime.now().strftime('%Y%m%d_%H%M%S')
log_file = LOG_DIR / ('pipeline_' + now_str + '.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def run_script(script_path):
    logging.info('>>> Starting: ' + script_path.name)
    start_time = time.time()
    try:
        venv_python = PROJECT_ROOT / '.venv' / 'Scripts' / 'python.exe'
        python_exe = str(venv_python) if venv_python.exists() else sys.executable
        process = subprocess.run(
            [python_exe, str(script_path)],
            check=True,
            text=True,
            capture_output=True,
            encoding='utf-8'
        )
        if process.stdout:
            for line in process.stdout.splitlines():
                if line.strip(): logging.info('  [STDOUT] ' + line)
        duration = time.time() - start_time
        logging.info('<<< Finished: ' + script_path.name + ' in ' + '{:.2f}'.format(duration) + 's')
        return True
    except subprocess.CalledProcessError as e:
        logging.error('!!! Error in ' + script_path.name + ':')
        if e.stdout: logging.error('  [STDOUT] ' + e.stdout)
        if e.stderr: logging.error('  [STDERR] ' + e.stderr)
        return False
    except Exception as e:
        logging.error('!!! Unexpected error: ' + str(e))
        return False

def main():
    start_pipeline = time.time()
    logging.info('='*60)
    logging.info('DROWSINESS DETECTION PIPELINE STARTING')
    logging.info('Project Root: ' + str(PROJECT_ROOT))
    logging.info('Log File: ' + str(log_file))
    logging.info('='*60)
    steps = [
        ('Frame Extraction', PROJECT_ROOT / 'Frame_exrtaction' / '4fps.py'),
        ('Face Mesh Processing', PROJECT_ROOT / 'frame' / 'mesh_apply.py'),
        ('CSV Feature Export', PROJECT_ROOT / 'to_csv.py')
    ]
    success_count = 0
    for name, script in steps:
        logging.info('\n--- STEP ' + str(success_count + 1) + ': ' + name + ' ---')
        if script.exists():
            if run_script(script):
                success_count += 1
            else:
                logging.error('Pipeline halted at step: ' + name)
                break
        else:
            logging.error('Script missing: ' + str(script))
            break
    total_duration = time.time() - start_pipeline
    logging.info('\n' + '='*60)
    if success_count == len(steps):
        logging.info('PIPELINE COMPLETED SUCCESSFULLY!')
    else:
        logging.info('PIPELINE FAILED at step ' + str(success_count + 1) + '/' + str(len(steps)))
    logging.info('Total execution time: ' + '{:.2f}'.format(total_duration/60) + ' minutes')
    logging.info('='*60)

if __name__ == '__main__':
    main()
