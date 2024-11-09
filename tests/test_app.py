import pytest
from flask import Flask,request
from flask.testing import FlaskClient
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Setup the Flask app (same as in your main app file)
app = Flask(__name__)

df = pd.read_csv("salary_prediction_data.csv")

# Encoding the categorical data
lb_ed = LabelEncoder()
df["Education"] = lb_ed.fit_transform(df["Education"])

lb_l = LabelEncoder()
df["Location"] = lb_l.fit_transform(df["Location"])

lb_jt = LabelEncoder()
df["Job_Title"] = lb_jt.fit_transform(df["Job_Title"])

lb_g = LabelEncoder()
df["Gender"] = lb_g.fit_transform(df["Gender"])

# Features and target variable
x = df.drop(columns=["Salary"], axis=1)
y = df["Salary"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(x_train, y_train)

# Define the main Flask route for testing
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':

        education = request.form['education'] 
        location = request.form['location'] 
        job_title = request.form['job_title']
        gender = request.form['gender'] 
        age = request.form['age']
        experience = float(request.form['experience']) 
        
        education_encoded = lb_ed.transform([education])[0]
        location_encoded = lb_l.transform([location])[0]
        job_title_encoded = lb_jt.transform([job_title])[0]
        gender_encoded = lb_g.transform([gender])[0]
        
        input_data = np.array([[education_encoded, location_encoded, job_title_encoded, gender_encoded, age, experience]])
        
        predicted_salary = rf.predict(input_data)[0]
        
        return f'Predicted Salary: {predicted_salary}' 
    return 'Flask App is Running'


@pytest.fixture
def client():
    with app.test_client() as client:
        yield client


def test_index(client: FlaskClient):
    response = client.get('/')
    assert response.status_code == 200


def test_predict_salary(client: FlaskClient):

    data = {
        'education': 'Master',  
        'location': 'NYC',     
        'job_title': 'Engineer',
        'gender': 'Male',       
        'age': 30,
        'experience': 5
    }
    
    response = client.post('/', data=data)
    assert 'Predicted Salary' in response.data.decode() 
