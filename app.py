import numpy as np
from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open('final_model', 'rb')) 


@app.route("/")
# @app.route("/home")
def home_page():
    return  render_template("index.html")


@app.route("/result", methods =['POST']) 
def result_page():
    Pregnancies = request.form('pregnancies')
    Glucose = request.form('glucose')
    BP = request.form('blood_pressure')
    S_Thickness = request.form('skin_thickness')
    Insulin = request.form('insulin')
    BMI = request.form('bmi')
    Diabetes_PF = request.form('diabetes_pf')
    Age = request.form('age')
         


    input_query = np.array([[Pregnancies,Glucose,BP,S_Thickness,Insulin,BMI,Diabetes_PF,Age]])
    result = model.predict(input_query)[0]
    return jsonify({'diabetes':str(result)})




if __name__ == "__main__":
    app.run(debug=True)
    
    
    
    
    
    
