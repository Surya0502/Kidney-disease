from flask import Flask,render_template,request
from flask_material import Material

# EDA PKg
import numpy as np 

# ML Pkg
from sklearn.externals import joblib


app = Flask(__name__)
Material(app)

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/preview')
def preview():
    return render_template("preview.html")

@app.route('/',methods=["POST"])
def analyze():
    if request.method == 'POST':
        age = request.form['age']
        sg = request.form['sg']
        al = request.form['al']
        su = request.form['su']
        bgr = request.form['bgr']
        sc= request.form['sc']
        pot = request.form['pot']
        pcv = request.form['pcv']
        wc = request.form['wc']
        rc = request.form['rc']
        dm = request.form['dm']
        model_choice = request.form['model_choice']
        

		# Clean the data by convert from unicode to float 
        sample_data = [age,sg,al,su,bgr,sc,pot,pcv,wc,rc,dm]
        clean_data = [float(i) for i in sample_data]

		# Reshape the Data as a Sample not Individual Features
        ex1 = np.array(clean_data).reshape(1,-1)

		# ex1 = np.array([6.2,3.4,5.4,2.3]).reshape(1,-1)
            
		# Reloading the Model
        if model_choice == 'logitmodel':
            logit_model = joblib.load('data/logit_model.pkl')
            result_prediction = logit_model.predict(ex1)
        elif model_choice == 'dtreemodel':
            knn_model = joblib.load('data/dtree_model.pkl')
            result_prediction = knn_model.predict(ex1)
        elif model_choice == 'ranmodel':
            svm_model = joblib.load('data/ran_model.pkl')
            result_prediction = svm_model.predict(ex1)
       

    return render_template('index.html',age=age,sg=sg,al=al,su=su,bgr=bgr,sc=sc,pot=pot,pcv=pcv,wc=wc,rc=rc,dm=dm,clean_data=clean_data,result_prediction=result_prediction,model_selected=model_choice)


if __name__ == '__main__':
	app.run()
