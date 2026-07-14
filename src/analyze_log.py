import re

log = open(r'E:\Buồn_ngủ\logs\train_film_gru.log', encoding='utf-8', errors='replace').read()

runs = log.split('LateFusionDataset:')
print(f'Total runs: {len(runs)-1}')

for i, run in enumerate(runs[1:], 1):
    windows = re.search(r'windows=(\d+)', run)
    print(f'\n{"="*60}')
    print(f'RUN {i} | windows={windows.group(1) if windows else "?"}')
    print(f'{"="*60}')

    folds = re.split(r'Fold \d+ held_out=', run)
    for j, fold in enumerate(folds[1:], 1):
        held = re.search(r'\[.*?\]', fold)
        held_str = held.group() if held else ''
        f1s = re.findall(r'val_f1=([\d.]+)', fold)
        recalls = re.findall(r'drowsy_recall=([\d.]+)', fold)
        stopped = 'early stop' in fold

        if f1s:
            best_f1 = max(float(x) for x in f1s)
            best_ep = max(range(len(f1s)), key=lambda k: float(f1s[k])) + 1
            last_f1 = float(f1s[-1])
            dropped = best_f1 - last_f1

            # Check if val_f1 is unstable (big swings)
            vals = [float(x) for x in f1s]
            swings = [abs(vals[k]-vals[k-1]) for k in range(1, len(vals))]
            max_swing = max(swings) if swings else 0

            print(f'  Fold {j} {held_str}:')
            print(f'    best_f1={best_f1:.4f} @ ep{best_ep}  last_f1={last_f1:.4f}  drop={dropped:.4f}  max_swing={max_swing:.4f}')
            print(f'    f1_per_epoch: {[round(float(x),3) for x in f1s]}')
            if dropped > 0.2:
                print(f'    *** WARNING: best F1 dropped {dropped:.3f} after peak — unstable training ***')

    # Final result
    mean_f1 = re.search(r'macro_F1\s+=\s+([\d.]+)', run)
    if mean_f1:
        print(f'\n  FINAL mean macro_F1 = {mean_f1.group(1)}')
