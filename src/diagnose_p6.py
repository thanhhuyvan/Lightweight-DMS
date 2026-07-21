import pandas as pd
import numpy as np

df = pd.read_csv('frame/csv/behavioral_vectors.csv')
df2 = df[df['video_id'].isin([0, 10])].copy()

PARTICIPANTS = ['partcipant2', 'partcipant4', 'participant3', 'participant5', 'participant6']

print("=== EAR_Mean by participant (alert vs drowsy) ===")
for pid in PARTICIPANTS:
    a = df2[(df2['participant_id']==pid) & (df2['video_id']==0)]['EAR_Mean']
    d = df2[(df2['participant_id']==pid) & (df2['video_id']==10)]['EAR_Mean']
    print(f"  {pid:15s}  alert={a.mean():.4f}  drowsy={d.mean():.4f}  diff={a.mean()-d.mean():.4f}")

print()
print("=== After MinMaxScaler fitted on others — where does p6 land? ===")
others = df2[df2['participant_id'] != 'participant6']
p6     = df2[df2['participant_id'] == 'participant6']
lo, hi = others['EAR_Mean'].min(), others['EAR_Mean'].max()
p6_alert_s  = (p6[p6.video_id==0]['EAR_Mean'].mean()  - lo) / (hi - lo)
p6_drowsy_s = (p6[p6.video_id==10]['EAR_Mean'].mean() - lo) / (hi - lo)
print(f"  p6 alert  EAR scaled: {p6_alert_s:.3f}")
print(f"  p6 drowsy EAR scaled: {p6_drowsy_s:.3f}")
print(f"  Others EAR range:     [{lo:.4f}, {hi:.4f}]")

print()
print("=== Other participants scaled alert/drowsy for comparison ===")
for pid in ['partcipant2', 'participant3', 'participant5']:
    p = df2[df2['participant_id']==pid]
    a_s = (p[p.video_id==0]['EAR_Mean'].mean()  - lo) / (hi - lo)
    d_s = (p[p.video_id==10]['EAR_Mean'].mean() - lo) / (hi - lo)
    print(f"  {pid:15s}  alert_scaled={a_s:.3f}  drowsy_scaled={d_s:.3f}  diff={a_s-d_s:.3f}")

print()
print("=== MAR (yawning) — same analysis ===")
lo_m, hi_m = others['MAR_Mean'].min(), others['MAR_Mean'].max()
for pid in PARTICIPANTS:
    a = df2[(df2['participant_id']==pid) & (df2['video_id']==0)]['MAR_Mean'].mean()
    d = df2[(df2['participant_id']==pid) & (df2['video_id']==10)]['MAR_Mean'].mean()
    print(f"  {pid:15s}  alert={a:.4f}  drowsy={d:.4f}  diff={d-a:.4f}")
