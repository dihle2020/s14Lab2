from flask import Flask, render_template
import joblib
app = Flask(__name__)

# Load ML model
tts_model = joblib.load('./notebooks/test_train_split.pkl')
dt_model = joblib.load('./notebooks/decision_tree.pkl')
@app.route('/')
def index():
    return render_template('index.html', bCE=estimate_tts(), decision_tree=estimate_dt())


def estimate_tts():
    tts_prediction = tts_model.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(1)
    tts_prediction = str(tts_prediction)
    return tts_prediction


def estimate_dt():
    dt_prediction = dt_model.predict([[4, 2.5, 3005, 15, 17903.0, 1]])[0][0].round(1)
    dt_prediction = str(dt_prediction)
    return dt_prediction
