import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


st.set_page_config(page_title='Student Performance Intelligence', page_icon='🎓', layout='wide')
st.title('🎓 Student Performance Intelligence')
st.caption('Academic analytics demo • Predictions support staff judgement and are not final decisions.')
model_path = Path('models/score_model.pkl'); risk_path=Path('models/risk_model.pkl')
if not model_path.exists() or not risk_path.exists():
    st.warning('Models are missing. Run `python src/generate_data.py` and `python src/train.py` first.')
    st.stop()
score_model=joblib.load(model_path); risk_model=joblib.load(risk_path)
with st.sidebar:
    st.header('Student profile')
    study=st.slider('Study hours/day',0.0,12.0,4.0,.5)
    attendance=st.slider('Attendance (%)',0.0,100.0,80.0,1.0)
    previous=st.slider('Previous score',0.0,100.0,65.0,1.0)
    assignment=st.slider('Assignment score',0.0,100.0,70.0,1.0)
    sleep=st.slider('Sleep hours/day',0.0,12.0,7.0,.5)
    extracurricular=st.selectbox('Extracurricular', ['Yes','No'])
    internet=st.selectbox('Internet access', ['Yes','No'])
    parental=st.selectbox('Parental education', ['School','Undergraduate','Postgraduate'])
    support=st.selectbox('Family support', ['Low','Medium','High'])
row=pd.DataFrame([{'study_hours':study,'attendance_pct':attendance,'previous_score':previous,'assignment_score':assignment,'sleep_hours':sleep,'extracurricular':extracurricular,'internet_access':internet,'parental_education':parental,'family_support':support}])
score=float(score_model.predict(row)[0]); riskprob=float(risk_model.predict_proba(row)[0,1]); risklabel='High risk' if riskprob>=.65 else ('Moderate risk' if riskprob>=.35 else 'Low risk')
col1,col2,col3=st.columns(3)
col1.metric('Predicted final score',f'{score:.1f}/100')
col2.metric('Underperformance risk',f'{riskprob:.0%}')
col3.metric('Risk classification',risklabel)
st.subheader('Recommended actions')
if riskprob>=.65:
    st.error('Suggested: faculty check-in, attendance plan, weekly progress review, and targeted practice.')
elif riskprob>=.35:
    st.warning('Suggested: monitor attendance and assignment completion; provide focused practice resources.')
else:
    st.success('Suggested: maintain study routine and monitor performance at regular intervals.')
st.caption('Risk probabilities are model estimates trained on synthetic data. Validate on real, consented, anonymized institutional data before any operational use.')
