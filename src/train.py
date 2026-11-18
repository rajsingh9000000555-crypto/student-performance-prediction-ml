from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report, roc_auc_score
import joblib

DATA = 'data/raw/student_data.csv'
NUM = ['study_hours','attendance_pct','previous_score','assignment_score','sleep_hours']
CAT = ['extracurricular','internet_access','parental_education','family_support']

def main():
    df = pd.read_csv(DATA)
    X = df[NUM+CAT]
    y = df['final_score']
    # Risk is defined as predicted/observed score under 50 for demo. This is a configurable academic threshold.
    risk = (y < 50).astype(int)
    X_train, X_test, y_train, y_test, r_train, r_test = train_test_split(X, y, risk, test_size=.2, random_state=42)
    prep = ColumnTransformer([('num', StandardScaler(), NUM), ('cat', OneHotEncoder(handle_unknown='ignore'), CAT)])
    models = {
      'RandomForest': RandomForestRegressor(n_estimators=350, min_samples_leaf=2, random_state=42),
      'HistGradientBoosting': HistGradientBoostingRegressor(max_iter=200, learning_rate=.08, random_state=42)
    }
    best_name, best_pipe, best_mae = None, None, float('inf')
    results = {}
    for name, model in models.items():
        pipe = Pipeline([('prep', prep), ('model', model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        rmse = mean_squared_error(y_test, pred) ** .5
        r2 = r2_score(y_test, pred)
        results[name] = {'MAE':float(mae),'RMSE':float(rmse),'R2':float(r2)}
        if mae < best_mae:
            best_name, best_pipe, best_mae = name, pipe, mae
    # Risk classifier with same feature pipeline
    risk_pipe = Pipeline([('prep', ColumnTransformer([('num',StandardScaler(),NUM),('cat',OneHotEncoder(handle_unknown='ignore'),CAT)])),('model',RandomForestClassifier(n_estimators=300,class_weight='balanced',random_state=42))])
    risk_pipe.fit(X_train, r_train)
    risk_pred = risk_pipe.predict(X_test)
    try:
        risk_auc = roc_auc_score(r_test, risk_pipe.predict_proba(X_test)[:,1])
    except ValueError:
        risk_auc = None
    Path('models').mkdir(exist_ok=True); Path('reports').mkdir(exist_ok=True)
    joblib.dump(best_pipe,'models/score_model.pkl'); joblib.dump(risk_pipe,'models/risk_model.pkl')
    report = {'best_regression_model':best_name,'regression_metrics':results,'risk_auc':risk_auc,'risk_report':classification_report(r_test,risk_pred,output_dict=True,zero_division=0),'risk_threshold_score':50,'data_note':'Synthetic demo data. Do not present results as real institutional performance.'}
    with open('reports/model_metrics.json','w') as f: json.dump(report,f,indent=2)
    print(json.dumps({'best':best_name,'metrics':results,'risk_auc':risk_auc},indent=2))
if __name__=='__main__': main()
