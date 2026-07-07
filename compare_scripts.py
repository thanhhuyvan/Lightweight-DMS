import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

with open(r'E:\Buồn_ngủ\logs\training_outside\train_film_gru (2).py', encoding='utf-8') as f:
    outside = f.read()
with open(r'E:\Buồn_ngủ\src\s4_training\train_film_gru.py', encoding='utf-8') as f:
    src = f.read()

features = [
    ('--onecycle',          'OneCycleLR flag'),
    ('--swa',               'SWA flag'),
    ('swa_start',           'SWA start param'),
    ('OneCycleLR',          'OneCycleLR implementation'),
    ('AveragedModel',       'SWA AveragedModel'),
    ('update_bn',           'SWA BatchNorm update'),
    ('RUN COMMAND',         'Run command logging'),
    ('FileHandler',         'Log file handler'),
    ('confidence',          'Confidence decay'),
    ('--attention',         'Attention flag'),
    ('onecycle_ready',      'OneCycleLR deferred fix'),
    ('exclude_participants','Exclude participants'),
    ('swa_lr',              'SWA lr param'),
    ('in_swa_phase',        'SWA phase tracking'),
    ('active_scheduler',    'Scheduler per-batch step'),
    ('freeze_cnn_epochs',   'CNN freeze warmup'),
]

print('Feature comparison: outside script vs src/s4_training/train_film_gru.py')
print('-' * 70)
print(f'{"Feature":<35} {"Outside":^10} {"Src":^10} {"Status":^10}')
print('-' * 70)
mismatches = []
for keyword, label in features:
    in_outside = keyword in outside
    in_src = keyword in src
    status = 'OK' if in_outside == in_src else 'MISMATCH'
    if status == 'MISMATCH':
        mismatches.append((label, in_outside, in_src))
    print(f'{label:<35} {str(in_outside):^10} {str(in_src):^10} {status:^10}')

print()
if mismatches:
    print('MISMATCHES FOUND:')
    for label, o, s in mismatches:
        print(f'  - {label}: outside={o}, src={s}')
else:
    print('All features match.')

# Also check line counts
print(f'\nLine counts: outside={outside.count(chr(10))}, src={src.count(chr(10))}')
