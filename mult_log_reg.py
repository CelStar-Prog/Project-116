from random import random
import pandas as pd
import csv
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Admission_Predict.csv")
toefl_score = df["TOEFL Score"].tolist()
chance = df["Chance of admit"].tolist()
gre_score = df["GRE Score"].tolist()
#fig = px.scatter(x = toefl_score, y = chance)
#fig.show()
colors = []
for data in chance:
    if data == 1:
        colors.append('green')

    else:
        colors.append('red')

fig = go.Figure(data = go.Scatter(
    x = toefl_score,
    y = gre_score,
    mode = 'markers',
    marker = dict(color = colors)
))
#fig.show()

scores = df[["GRE Score", "TOEFL Score"]]
results = df["Chance of admit"]
score_train, score_test, result_train, result_test = train_test_split(scores, results, test_size = 0.25, random_state = 0) 
classifier = LogisticRegression(random_state = 0)
classifier.fit(score_train, result_train)
results_prediction = classifier.predict(score_test)
print("Accuracy: ", accuracy_score(result_test, results_prediction))
sc = StandardScaler()
score_train = sc.fit_transform(score_train)
user_gre_score = int(input("Enter the GRE Score: "))
user_toefl_score = int(input("Enter the TOEFL Score: "))
user_test = sc.transform([[user_gre_score, user_toefl_score]])
user_result_prediction = classifier.predict(user_test)
if user_result_prediction[0] == 1:
    print("This user may pass")

else:
    print("Sed")