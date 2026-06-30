import pandas as pd, numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('frame/csv/behavioral_vectors.csv')
FEATS = ['PERCLOS','Blink_Rate','Blink_Avg_Duration','EAR_Mean','EAR_Std','MAR_Mean','MAR_Max','Pitch_Jitter','Yaw_Jitter','Roll_Jitter','Pose_Jitter']
df2 = df[df['video_id'].isin([0,10])].copy()
df2['label'] = (df2['video_id']==10).astype(int)

train = df2[df2['participant_id'] != 'participant6']
test  = df2[df2['participant_id'] == 'participant6']
sc = MinMaxScaler().fit(train[FEATS])
clf = GradientBoostingClassifier(random_state=42).fit(sc.transform(train[FEATS]), train['label'])

imp = sorted(zip(FEATS, clf.feature_importances_), key=lambda x: -x[1])
print('Cross-participant model feature importances:')
for f, i in imp[:6]:
    print(f'  {f:20s} {i:.4f}')

print()
for feat in ['Pose_Jitter', 'MAR_Max', 'EAR_Mean']:
    ta = train[train.label==0][feat].mean()
    td = train[train.label==1][feat].mean()
    p6a = test[test.label==0][feat].mean()
    p6d = test[test.label==1][feat].mean()
    # scaled values
    ta_s  = sc.transform(train[train.label==0][FEATS].head(1))[0][FEATS.index(feat)]
    print(f'{feat}:  train alert={ta:.1f} drowsy={td:.1f} | p6 alert={p6a:.1f} drowsy={p6d:.1f}')

print()
preds = clf.predict(sc.transform(test[FEATS]))
proba = clf.predict_proba(sc.transform(test[FEATS]))[:,1]
print(f'p6 predictions: alert_pred={(preds==0).sum()} drowsy_pred={(preds==1).sum()} out of {len(preds)}')
print(f'p6 mean drowsy prob: alert_windows={proba[test.label.values==0].mean():.3f}  drowsy_windows={proba[test.label.values==1].mean():.3f}')

# Scaled p6 pose jitter vs training range
print()
p6_scaled = sc.transform(test[FEATS])
train_scaled = sc.transform(train[FEATS])
pose_idx = FEATS.index('Pose_Jitter')
print(f'Pose_Jitter SCALED:')
print(f'  train range: [{train_scaled[:,pose_idx].min():.3f}, {train_scaled[:,pose_idx].max():.3f}]')
print(f'  p6 alert:     {p6_scaled[test.label.values==0, pose_idx].mean():.3f}')
print(f'  p6 drowsy:    {p6_scaled[test.label.values==1, pose_idx].mean():.3f}')
ear_idx = FEATS.index('EAR_Mean')
print(f'EAR_Mean SCALED:')
print(f'  train drowsy mean: {train_scaled[train.label.values==1, ear_idx].mean():.3f}')
print(f'  p6 alert mean:     {p6_scaled[test.label.values==0, ear_idx].mean():.3f}')
print(f'  p6 drowsy mean:    {p6_scaled[test.label.values==1, ear_idx].mean():.3f}')
