import os
import json
import joblib
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

file_students = "students.csv"
file_model = "rf_model.pkl"
file_summary = "model_summary.json"


def make_data(count=300, seed_val=7):
    gen = np.random.RandomState(seed_val)
    hrs = gen.normal(5, 2, count).clip(0, 12).round(2)
    attend = gen.uniform(50, 100, count).round(1)
    done = gen.randint(0, 10, count)
    prev_marks = gen.normal(65, 15, count).clip(0, 100).round(1)
    final = hrs * 6 + attend * 0.2 + done * 3 + prev_marks * 0.4 + gen.normal(0, 10, count)
    passed = (final >= 60).astype(int)
    df = pd.DataFrame({
        "study_hours": hrs,
        "attendance_rate": attend,
        "assignments_done": done,
        "previous_score": prev_marks,
        "passed": passed
    })
    df.to_csv(file_students, index=False)
    return df


def read_data():
    if os.path.exists(file_students):
        return pd.read_csv(file_students)
    return make_data()


def train(df=None):
    if df is None:
        df = read_data()
    x_vals = df[["study_hours", "attendance_rate", "assignments_done", "previous_score"]].values
    y_vals = df["passed"].values
    x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.2, random_state=3, stratify=y_vals)
    model = RandomForestClassifier(n_estimators=100, random_state=3)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    acc_val = float(accuracy_score(y_test, preds))
    rpt = classification_report(y_test, preds, target_names=["fail", "pass"], zero_division=0)
    summary = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "accuracy": acc_val,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "classification_report": rpt
    }
    joblib.dump(model, file_model)
    with open(file_summary, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return model, summary


def get_model():
    if not os.path.exists(file_model):
        return None
    return joblib.load(file_model)


def check_student(hrs, attend, done, prev_marks):
    model = get_model()
    if model is None:
        return None
    arr = np.array([[hrs, attend, done, prev_marks]])
    res = int(model.predict(arr)[0])
    probs = model.predict_proba(arr)[0]
    return {"label": res, "prob_fail": float(probs[0]), "prob_pass": float(probs[1])}


def rank_features():
    model = get_model()
    if model is None:
        return None
    cols = ["study_hours", "attendance_rate", "assignments_done", "previous_score"]
    scores = model.feature_importances_
    return sorted(zip(cols, scores), key=lambda t: t[1], reverse=True)


def run_menu():
    data_set = read_data()
    mdl = get_model()
    if mdl is None:
        mdl, s = train(data_set)
        print("Model trained. Accuracy:", round(s["accuracy"], 3))
    else:
        print("Model loaded.")
    while True:
        print("\nOptions:")
        print("1) Retrain")
        print("2) Predict student")
        print("3) Feature ranking")
        print("4) Exit")
        choice = input("Choice: ").strip()
        if choice == "1":
            mdl, s = train()
            print("Retrained. Accuracy:", round(s["accuracy"], 3))
        elif choice == "2":
            try:
                h = float(input("Study hours: "))
                a = float(input("Attendance %: "))
                d = int(input("Assignments done: "))
                p = float(input("Previous score: "))
            except Exception:
                print("Invalid input.")
                continue
            out = check_student(h, a, d, p)
            if out is None:
                print("Model missing.")
                continue
            print("\nPrediction:", "pass" if out["label"] == 1 else "fail")
            print("Pass prob:", round(out["prob_pass"], 3))
            print("Fail prob:", round(out["prob_fail"], 3))
        elif choice == "3":
            ranked = rank_features()
            if ranked is None:
                print("Model missing.")
                continue
            print("\nFeature importance:")
            for name, sc in ranked:
                print(name + ":", round(sc, 3))
        elif choice == "4":
            break
        else:
            print("Invalid option.")


if __name__ == "__main__":
    run_menu()