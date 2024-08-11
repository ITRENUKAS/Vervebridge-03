import numpy as np
from flask import Flask, request, render_template
import pickle
import warnings
warnings.simplefilter("ignore", UserWarning)

# Create flask app
app = Flask(__name__)

# Load the model once when the app starts
model_path = r"C:\Users\Admin\Downloads\Campus-Placement-Prediction-main\Campus-Placement-Prediction-main\model.pkl"
model = pickle.load(open(model_path, "rb"))

# Prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)
    result = model.predict(to_predict)  # Use the preloaded model
    return result[0]

@app.route("/")
def Home():
    print('Request for index page received')
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
    print('Request for predict page received')
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        if int(result) == 1:
            prediction = 'Placed'
        else:
            prediction = 'Not Placed'
    return render_template("result.html", prediction_text=prediction)

if __name__ == "__main__":
    app.run(debug=True)
