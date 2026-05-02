import psycopg2
from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import joblib
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
import os
import io
from datetime import datetime

# ---------------- DATABASE CONNECTION ----------------
def get_db_connection():
    url = os.environ.get("DATABASE_URL")

    if not url:
        raise Exception("DATABASE_URL not set")

    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)

    return psycopg2.connect(url, sslmode='require')


# ---------------- APP SETUP ----------------
app = Flask(__name__)
app.secret_key = "super_secret_key"

# ---------------- LOGIN SETUP ----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ---------------- USER CLASS ----------------
class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

# ---------------- LOAD USER ----------------
@login_manager.user_loader
def load_user(user_id):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, username FROM users WHERE id=%s", (user_id,))
    user = cur.fetchone()

    cur.close()
    conn.close()

    if user:
        return User(user[0], user[1])
    return None

# ---------------- LOAD MODEL ----------------
model = joblib.load("loan_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ---------------- PREDICTION ----------------
def predict_loan(input_data_dict):
    input_df = pd.DataFrame([input_data_dict])
    input_df["loan_to_income_ratio"] = input_df["loan_amnt"] / input_df["person_income"]

    input_df_encoded = pd.get_dummies(input_df)

    for col in feature_names:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    input_df_final = input_df_encoded[feature_names]

    default_prob = model.predict_proba(input_df_final)[0][1] * 100

    if default_prob < 40:
        risk = "Low Risk"
    elif default_prob < 55:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    decision = "Loan Rejected" if default_prob > 55 else "Loan Approved"

    return decision, risk, float(round(default_prob, 2))

# ---------------- EXPLANATION ----------------
def generate_explanation(input_data, default_prob):
    reasons = []
    ratio = input_data["loan_amnt"] / input_data["person_income"]

    if ratio > 0.5:
        reasons.append("High loan vs income")
    elif ratio < 0.2:
        reasons.append("Loan manageable")
    else:
        reasons.append("Moderate ratio")

    if input_data["credit_score"] < 600:
        reasons.append("Low credit score")
    elif input_data["credit_score"] > 700:
        reasons.append("Strong credit score")

    if default_prob > 60:
        reasons.append("High default risk")

    return reasons

# ---------------- ROUTES ----------------

@app.route('/')
@login_required
def home():
    return render_template('index.html', user=current_user.username)

# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        login_id = request.form['login_id']
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            "SELECT id, username, password FROM users WHERE username=%s OR email=%s",
            (login_id, login_id)
        )
        user = cur.fetchone()

        cur.close()
        conn.close()

        if not user:
            return render_template('login.html', error="User not found")

        if not check_password_hash(user[2], password):
            return render_template('login.html', error="Invalid password")

        login_user(User(user[0], user[1]))
        return redirect('/')

    return render_template('login.html')

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        input_data = {
            "person_age": int(request.form["person_age"]),
            "person_income": float(request.form["person_income"]),
            "person_emp_exp": float(request.form["person_emp_exp"]),
            "loan_amnt": float(request.form["loan_amount"]),
            "loan_percent_income": float(request.form["loan_percent_income"]) / 100,
            "credit_score": float(request.form["credit_score"]),
            "loan_intent": request.form["loan_intent"].upper(),
            "person_gender": request.form["person_gender"].lower(),
            "person_home_ownership": request.form["person_home_ownership"].upper()
        }

        decision, risk, default_prob = predict_loan(input_data)

        return render_template(
            "result.html",
            approval_status=decision,
            probability=default_prob,
            risk=risk
        )

    except Exception as e:
        return f"ERROR: {str(e)}"

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)
