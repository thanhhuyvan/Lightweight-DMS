import subprocess
import sys
from pathlib import Path

def run_script(script_path):
    print(f"\n--- Running: {script_path} ---")
    try:
        # Sử dụng python từ venv nếu có
        venv_python = Path(r"E:\Buồn ngủ\.venv\Scripts\python.exe")
        python_exe = str(venv_python) if venv_python.exists() else sys.executable
        
        result = subprocess.run([python_exe, str(script_path)], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        return False

def main():
    project_root = Path(r"E:\Buồn ngủ")
    
    # Danh sách các bước xử lý
    steps = [
        project_root / "Frame_exrtaction" / "4fps.py",
        project_root / "frame" / "mesh_apply.py",
        project_root / "to_csv.py"
    ]
    
    print("Starting Drowsiness Detection Pipeline...")
    
    for script in steps:
        if script.exists():
            success = run_script(script)
            if not success:
                print("Pipeline stopped due to error.")
                break
        else:
            print(f"Warning: Script not found: {script}")

    print("\nPipeline execution finished.")

if __name__ == "__main__":
    main()
