import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
n = 1200
study = rng.uniform(0.5, 8, n)
attendance = rng.uniform(45, 100, n)
previous = rng.uniform(20, 100, n)
assign = rng.uniform(25, 100, n)
sleep = rng.uniform(4, 10, n)
extracurricular = rng.choice(['Yes', 'No'], n, p=[.4,.6])
internet = rng.choice(['Yes', 'No'], n, p=[.85,.15])
parent_education = rng.choice(['School', 'Undergraduate', 'Postgraduate'], n, p=[.45,.35,.2])
family_support = rng.choice(['Low', 'Medium', 'High'], n, p=[.2,.5,.3])
noise = rng.normal(0, 8, n)
score = 0.22*previous + 2.8*study + 0.18*attendance + 0.14*assign + 1.2*(sleep-6) + 2.5*(extracurricular=='Yes') + 2*(internet=='Yes') + 1.5*(family_support=='High') - 1.2*(family_support=='Low') + noise
score = np.clip(score, 0, 100)
df = pd.DataFrame({'study_hours':study.round(1),'attendance_pct':attendance.round(1),'previous_score':previous.round(1),'assignment_score':assign.round(1),'sleep_hours':sleep.round(1),'extracurricular':extracurricular,'internet_access':internet,'parental_education':parent_education,'family_support':family_support,'final_score':score.round(1)})
Path('data/raw').mkdir(parents=True, exist_ok=True)
df.to_csv('data/raw/student_data.csv', index=False)
print('Created synthetic dataset:', len(df), 'rows')
