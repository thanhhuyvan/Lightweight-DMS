# Master Project Management & Pipeline Plan

This document outlines the workflow for managing group contributions, merging code via GitHub, and ensuring the correct execution sequence of the project pipeline.

## 1. Git Workflow (Manager's Guide)

To protect the core dataset and maintain code quality, follow this branching and merging strategy:

### Branching Strategy
- **`main` Branch**: Reserved for stable, verified code and the core preprocessed data (e.g., your 500MB CSV).
- **Feature Branches**: Every group member must work on a separate branch named after their task (e.g., `feature/analysis`, `feature/modeling`).

### PR Review & Merging Process
1.  **Submission**: Members submit a Pull Request (PR) from their feature branch to `main`.
2.  **Local Verification**: Before merging, the Manager fetches the PR branch to test locally:
    ```powershell
    git fetch origin
    git checkout feature/member-task-name
    python main.py  # Run the pipeline to check for regressions
    ```
3.  **Architectural Audit**: Ensure the new code:
    - Uses `pathlib` for all file operations.
    - References `config.py` for all paths and constants.
    - Logs errors/status via the `logging` system instead of `print()`.
4.  **Approval**: Once verified, merge the PR on GitHub.

---

## 2. Pipeline Sequencing (Master Orchestration)

To ensure all scripts run in the correct order, `main.py` acts as the single source of truth for execution.

### The `steps` List
All new scripts must be added to the `steps` array in `main.py` in their logical order. **Never run individual scripts in isolation for production results.**

**Current Sequence:**
1. `Frame_exrtaction/4fps.py` (Preprocessing)
2. `frame/mesh_apply.py` (Landmark Detection)
3. `to_csv.py` (Feature Extraction)

**How to Add New Tasks:**
Edit `main.py` and append the new script path to the `steps` list:
```python
steps = [
    ('Frame Extraction', PROJECT_ROOT / 'Frame_exrtaction' / '4fps.py'),
    ('Face Mesh Processing', PROJECT_ROOT / 'frame' / 'mesh_apply.py'),
    ('CSV Feature Export', PROJECT_ROOT / 'to_csv.py'),
    ('New Task Name', PROJECT_ROOT / 'path' / 'to' / 'new_script.py') # Added by Manager
]
```

---

## 3. Data Integrity Standards

- **Path Safety**: All scripts must use the `PROJECT_ROOT` variable from `config.py`.
- **Large Files**: The 500MB CSV (`landmarks_full.csv`) should be treated as **read-only** by analysis scripts to prevent accidental corruption.
- **Environment**: All members must use the shared `.venv` and update `requirements.txt` if new libraries are introduced.

---

## 4. Manager's Checklist for PRs
- [ ] Does the code follow `pathlib` standards?
- [ ] Is the script correctly registered in `main.py`?
- [ ] Does it run without breaking previous steps?
- [ ] Are new requirements added to `requirements.txt`?
