import psycopg2
from flask import Flask, render_template, request, redirect, send_file
import pandas as pd
import joblib
from werkzeug.security import generate_password_hash, check_password_hash

# 🔐 Login imports
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

# ---------------- DATABASE CONNECTION ----------------
import os

def get_db_connection():
    return psycopg2.connect(os.environ.get("DATABASE_URL"))

# ---------------- APP SETUP ----------------
app = Flask(__name__)
app.secret_key = "your_secret_key_here"

# ---------------- LOGIN SETUP ----------------
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# ---------------- USER CLASS ----------------
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# ---------------- LOAD USER ----------------
@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# ---------------- LOAD MODEL ----------------
model = joblib.load("loan_model.pkl")
feature_names = joblib.load("feature_names.pkl")

# ---------------- PREDICTION FUNCTION ----------------
def predict_loan(input_data_dict):

    input_df = pd.DataFrame([input_data_dict])

    input_df["loan_to_income_ratio"] = input_df["loan_amnt"] / input_df["person_income"]

    input_df_encoded = pd.get_dummies(input_df)

    for col in feature_names:
        if col not in input_df_encoded.columns:
            input_df_encoded[col] = 0

    input_df_final = input_df_encoded[feature_names]

    default_prob = model.predict_proba(input_df_final)[0][1] * 100

    if default_prob < 30:
        risk = "Low Risk"
    elif default_prob < 60:
        risk = "Medium Risk"
    else:
        risk = "High Risk"

    if default_prob > 60:
        decision = "Loan Rejected"
    else:
        decision = "Loan Approved"

    return decision, risk, round(default_prob, 2)


def calculate_feature_importance(input_data):
    score = input_data["credit_score"]
    income = input_data["person_income"]
    loan = input_data["loan_amnt"]

    credit_weight = score / 850
    income_weight = income / (income + loan)
    loan_weight = loan / (income + loan)

    total = credit_weight + income_weight + loan_weight

    credit = (credit_weight / total) * 100
    income_p = (income_weight / total) * 100
    loan_p = (loan_weight / total) * 100

    # Fix rounding drift
    total_sum = credit + income_p + loan_p
    correction = 100 - total_sum

    credit += correction  # adjust one value slightly

    return [
        ("Credit Score", round(credit, 1)),
        ("Income", round(income_p, 1)),
        ("Loan Amount", round(loan_p, 1))
    ]

# ---------------- EXPLANATION FUNCTION ----------------
def generate_explanation(input_data, default_prob):
    reasons = []

    ratio = input_data["loan_amnt"] / input_data["person_income"]

    # ---------------- LOAN TO INCOME ----------------
    if ratio > 0.5:
        reasons.append(f"Loan amount is high relative to income ({round(ratio*100,1)}%), increasing financial burden")
    elif ratio < 0.2:
        reasons.append(f"Loan amount is comfortably manageable ({round(ratio*100,1)}% of income)")
    else:
        reasons.append(f"Loan-to-income ratio is moderate ({round(ratio*100,1)}%)")

    # ---------------- CREDIT SCORE ----------------
    if input_data["credit_score"] < 600:
        reasons.append("Low credit score indicates higher credit risk")
    elif input_data["credit_score"] > 700:
        reasons.append("Strong credit score improves repayment reliability")
    else:
        reasons.append("Average credit score reflects moderate creditworthiness")

    # ---------------- EMPLOYMENT ----------------
    if input_data["person_emp_exp"] < 2:
        reasons.append("Limited employment experience suggests unstable income source")
    else:
        reasons.append("Stable employment history supports consistent income flow")

    # ---------------- FINAL DECISION ----------------
    if default_prob > 60:
        reasons.append(f"High default probability ({default_prob}%) resulted in loan rejection")
    elif default_prob < 30:
        reasons.append(f"Low default probability ({default_prob}%) supports loan approval")
    else:
        reasons.append(f"Moderate default probability ({default_prob}%) indicates balanced risk")

    return reasons

# ---------------- PDF FUNCTION ----------------
def generate_pdf(data):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()

    from reportlab.platypus import Table, TableStyle, Image
    from reportlab.lib import colors

    content = []

    # ---------------- HEADER ----------------
    content.append(Paragraph("<b>LoanPredict AI - Loan Report</b>", styles['Title']))
    content.append(Spacer(1, 15))

    content.append(Paragraph(f"User: {data['user']}", styles['Normal']))
    content.append(Spacer(1, 10))

    # ---------------- APPLICANT TABLE ----------------
    content.append(Paragraph("<b>Applicant Details</b>", styles['Heading2']))

    applicant_data = [
        ["Field", "Value"],
        ["Income", data['income']],
        ["Loan Amount", data['loan']],
        ["Credit Score", data['credit']]
    ]

    table = Table(applicant_data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.grey),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,1), (-1,-1), colors.whitesmoke)
    ]))

    content.append(table)
    content.append(Spacer(1, 15))

    # ---------------- RESULT SECTION ----------------
    content.append(Paragraph("<b>Prediction Result</b>", styles['Heading2']))

    decision_color = "green" if data['decision'] == "Loan Approved" else "red"

    content.append(Paragraph(
        f"<font color='{decision_color}'><b>{data['decision']}</b></font>",
        styles['Heading3']
    ))

    result_data = [
        ["Metric", "Value"],
        ["Approval Probability", f"{data['approval_prob']}%"],
        ["Risk of Default", f"{data['default_prob']}%"],
        ["Risk Level", data['risk']]
    ]

    table2 = Table(result_data)
    table2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige)
    ]))

    content.append(table2)
    content.append(Spacer(1, 15))

    # ---------------- EXPLANATION ----------------
    content.append(Paragraph("<b>Explanation</b>", styles['Heading2']))

    for reason in data['explanations']:
        content.append(Paragraph(f"• {reason}", styles['Normal']))

    content.append(Spacer(1, 20))

    # ---------------- FOOTER ----------------
    content.append(Paragraph(
        "<i>This is an AI-generated loan assessment report.</i>",
        styles['Normal']
    ))

    doc.build(content)

    buffer.seek(0)
    return buffer

# ---------------- ROUTES ----------------

@app.route('/')
@login_required
def home():
    return render_template('index.html', user=current_user.id)

# ---------------- SIGNUP ----------------
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':

        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            return render_template('signup.html', error="Passwords do not match ❌")

        if len(password) < 6:
            return render_template('signup.html', error="Password must be at least 6 characters ❌")

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE username=%s", (username,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return render_template('signup.html', error="Username already exists ❌")

        cur.execute("SELECT * FROM users WHERE email=%s", (email,))
        if cur.fetchone():
            cur.close()
            conn.close()
            return render_template('signup.html', error="Email already registered ❌")

        hashed_password = generate_password_hash(password)

        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
            (username, email, hashed_password)
        )

        conn.commit()
        cur.close()
        conn.close()

        return redirect('/login')

    return render_template('signup.html')

# ---------------- LOGIN ----------------
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':

        login_id = request.form['login_id']
        password = request.form['password']

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            "SELECT * FROM users WHERE username=%s OR email=%s",
            (login_id, login_id)
        )
        user = cur.fetchone()

        cur.close()
        conn.close()

        if not user:
            return render_template('login.html', error="User not found ❌")

        if not check_password_hash(user[2], password):
            return render_template('login.html', error="Invalid password ❌")

        login_user(User(user[1]))
        return redirect('/')

    return render_template('login.html')

# ---------------- LOGOUT ----------------
@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect('/login')

# ---------------- PREDICT ----------------
@app.route('/predict', methods=['POST'])
@login_required
def predict():

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
    approval_prob = 100 - default_prob

    explanations = generate_explanation(input_data, default_prob)
    feature_data = calculate_feature_importance(input_data)

    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
    INSERT INTO predictions (username, income, loan, credit_score, probability, decision)
    VALUES (%s, %s, %s, %s, %s, %s)
    """, (
        current_user.id,
        input_data["person_income"],
        input_data["loan_amnt"],
        input_data["credit_score"],
        default_prob,
        decision
    ))

    conn.commit()
    cur.close()
    conn.close()

    return render_template(
        "result.html",
        approval_status=decision,
        probability=default_prob,
        approval_prob=approval_prob,
        risk=risk,
        income=input_data["person_income"],
        loan=input_data["loan_amnt"],
        credit=input_data["credit_score"],
        explanations=explanations,
        feature_data=feature_data,
        user=current_user.id
    )

# ---------------- DOWNLOAD PDF ----------------
@app.route('/download_report')
@login_required
def download_report():
    data = {
        "user": current_user.id,
        "income": request.args.get("income"),
        "loan": request.args.get("loan"),
        "credit": request.args.get("credit"),
        "decision": request.args.get("decision"),
        "approval_prob": request.args.get("approval_prob"),
        "default_prob": request.args.get("probability"),
        "risk": request.args.get("risk"),
        "explanations": request.args.getlist("explanations")
    }

    pdf = generate_pdf(data)

    return send_file(
        pdf,
        as_attachment=True,
        download_name="loan_report.pdf",
        mimetype='application/pdf'
    )

# ---------------- HISTORY ----------------
@app.route('/history')
@login_required
def history():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute(
        "SELECT income, loan, credit_score, probability, decision FROM predictions WHERE username=%s",
        (current_user.id,)
    )

    data = cur.fetchall()

    cur.close()
    conn.close()

    return render_template("history.html", data=data)

# ---------------- DASHBOARD ----------------
@app.route('/dashboard')
@login_required
def dashboard():
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        SELECT income, loan, probability, decision, created_at
        FROM predictions
        WHERE username=%s
        ORDER BY created_at
    """, (current_user.id,))

    data = cur.fetchall()

    cur.close()
    conn.close()

    income = [row[0] for row in data]
    loan = [row[1] for row in data]
    probability = [row[2] for row in data]
    dates = [row[4].strftime("%d-%b")for row in data]

    approved = sum(1 for row in data if row[3] == "Loan Approved")
    rejected = sum(1 for row in data if row[3] == "Loan Rejected")

    total = len(data)

    avg_risk = round(sum(probability) / len(probability),1)if probability else 0

    if avg_risk < 30:
        risk_level = "Low"
        risk_class = "risk-low"
    elif avg_risk < 60:
        risk_level = "Medium"
        risk_class = "risk-medium"
    else:
        risk_level = "High"
        risk_class = "risk-high"

    return render_template(
        "dashboard.html",
        income=income,
        loan=loan,
        probability=probability,
        approved=approved,
        rejected=rejected,
        total=total,
        avg_risk=avg_risk,
        dates=dates,
        user=current_user.id,
        risk=risk_level,  # ✅ ADD THIS
        risk_class=risk_class  # ✅ ADD THIS
    )

# ---------------- RUN ----------------
if __name__ == '__main__':
    app.run(debug=True)
