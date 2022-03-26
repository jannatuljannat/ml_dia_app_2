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
    if request.method == 'POST':
        Pregnancies = float(request.form['pregnancies'])
        Glucose = float(request.form['glucose'])
        BP = float(request.form['blood_pressure'])
        S_Thickness = float(request.form['skin_thickness'])
        Insulin = float(request.form['insulin'])
        BMI = float(request.form['bmi'])
        Diabetes_PF = float(request.form['diabetes_pf'])
        Age = float(request.form['age']) 


# =============================================================================
#     result = {'preg':Pregnancies, 'glu':Glucose, 'bp':BP, 'st':S_Thickness, 'in':Insulin,'b_m_i':BMI, 'dpf':Diabetes_PF, 'A_g_e':Age}
#     return jsonify(result)
# 
# =============================================================================



    input_query = np.array([[Pregnancies,Glucose,BP,S_Thickness,Insulin,BMI,Diabetes_PF,Age]])
    result = model.predict(input_query)[0]
    return jsonify({'diabetes':str(result)})




# =============================================================================
#         prediction = model.predict([[Pregnancies,Glucose,BP,S_Thickness,Insulin,BMI,Diabetes_PF,Age]])
#         
#         if prediction == 0:
#             return render_template('index.html', prediction_text="Congrats! You don't have diabetes")
#         else:
#             return render_template('index.html', prediction_text="Oops! You have diabetes") 
#     else:
#         return render_template('index.html')     
# =============================================================================




if __name__ == "__main__":
    app.run()
    
    
    
    
    
    