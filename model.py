def predict_heart_disease(age, gender, chest_pain, bp, chol, sugar):
    # Simple logic (replace with your ML model later)
    score = age + bp + chol

    if score > 300:
        return "High risk of Heart Disease ❌"
    else:
        return "Low risk of Heart Disease ✅"
