## Team Collaboration
- **Task & Branch Mapping**: Check [KANBAN.md](./KANBAN.md) to see which branch you should use for your assigned task.
- **Rules**: 
    - Never push directly to `main`. 
    - Create a feature branch following the `feature/ID-description` format.
    - Submit a Pull Request for review.

## Pro-Tips
- **Using AI**: If you are using an AI assistant, tell it:
  > "Read the GEMINI.md, MASTER_PLAN.md, and KANBAN.md files first to understand our project structure and rules."
**Sequence workflow**
  >   1. src/core_config.py (The Map)
   * Why: This is the "single source of truth" for paths. You need to know exactly where SUMMARY_FEATURES_CSV is located
     and where to define a new path for your saved model (e.g., MODEL_SAVE_PATH).

  >   2. frame/csv/features_summary.csv (The Data)
   * Why: Read the header (first few lines). This is your training set. You need to identify your Features (e.g.,
     mean_EAR_smooth, MAR_smooth, head_dx_smooth) and your Target (likely eye_state).

  >   3. to_csv.py (The Feature Logic)
   * Why: Understand how the "smooth" columns were calculated (Savitzky-Golay filter). This ensures you understand the
     signal quality and why you should prefer _smooth columns over raw ones.

  >   4. src/eye_state.py (The Label Logic)
   * Why: This script generates the eye_state column (0 for open, 1 for closed). Since you are training a model, you
     need to know exactly how your "ground truth" labels were derived.

  >   5. KANBAN.md & MASTER_PLAN.md (The Rules)
   * Why:
       * ML-03 Dependency: The Kanban mentions that data splitting must be Participant-Based (GroupKFold). You cannot
         just do a random split, or you will leak participant data.
       * Workflow: Check the "Git Workflow" in MASTER_PLAN.md to ensure you create the correct branch (e.g.,
         feature/ML-04-random-forest) before you start.

  >   6. main.py (The Integration)
   * Why: See how steps are added to the steps list. Your final training script will eventually need to be registered
     here.
---
**Data Source**: [Google Drive CSV Folder](https://drive.google.com/drive/folders/1WuRqUJwTGjcF9nzPSVg-F7JMZeJ3gdkG?usp=sharing)
