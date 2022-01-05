from enum import IntEnum
from starlette.responses import RedirectResponse
from fastapi import FastAPI, Query
import pickle
import pandas as pd


app = FastAPI()
model = pickle.load(open('./model.pkl', 'rb'))

class SexEnum(IntEnum):
    FEMALE = 0
    MALE = 1


@app.get("/")
def read_root():
    redirect_response = RedirectResponse('/docs')
    return redirect_response
        

@app.get('/predict')
def predict_heart_disease(
    age: int = Query(..., title='Age'),
    sex: SexEnum = Query(..., title='Sex'),
    cp: int = Query(..., title='Chest pain type', ge=0, le=3),
    trestbps: int = Query(..., title='Resting blood pressure'),
    chol: int = Query(..., title='Serum cholestoral in mg/dl'),
    fbs: int = Query(..., title='Fasting blood sugar > 120 mg/dl', ge=0, le=1),
    restecg: int = Query(..., title='Resting electrocardiographic results', ge=0, le=2),
    thalach: int = Query(..., title='Maximum heart rate achieved'),
    exang: int = Query(..., title='Exercise induced angina', ge=0, le=1),
    oldpeak: float = Query(..., title='Oldpeak = ST depression induced by exercise relative to rest'),
    slope: int = Query(..., title='The slope of the peak exercise ST segment', ge=0, le=2),
    ca: int = Query(..., title='Number of major vessels (0-3) colored by flourosopy', ge=0, le=4),
    thal: int = Query(..., title='Thal: 3 = normal; 6 = fixed defect; 7 = reversable defect', ge=0, le=3)):

    cat_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    num_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    d = {
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'restecg': restecg,
        'thalach': thalach,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }
    
    try:
        df = pd.DataFrame(d, index=[0])
        prediction = model.predict(df[num_features + cat_features])[0]

        # prediction is numpy.int that couldn't be serialized
        result = int(prediction)

        return { 'result': result }
    except Exception as e:
        return { 'error': str(e) }
