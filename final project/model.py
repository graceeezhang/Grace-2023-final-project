import pandas as pd
import numpy as np
import pickle


data = pd.read_csv('heart_2020_cleaned.csv')

from sklearn.linear_model import LogisticRegression

X = data.drop(columns = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory','Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease', 'SkinCancer'] )
X.head(2)
y = data['HeartDisease']

lgr = LogisticRegression()
X = pd.get_dummies(X)

lgr.fit(X, y)

pickle.dump(lgr, open ('heart_2020_cleaned (1).pkl','wb'))
model = pickle.load(open('heart_2020_cleaned (1).pkl','wb'))
result = model.predict(model)
print(result)