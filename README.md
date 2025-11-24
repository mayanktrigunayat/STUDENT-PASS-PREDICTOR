**Student Pass Predictor.**

An AI/ML-driven Python project that generates synthetic student data, trains a Random Forest classifier to predict whether a student will pass, and provides a simple command-line menu to retrain the model, make predictions, and inspect feature importances.


---

**Features (AI/ML Focus).**

Generate reproducible synthetic student dataset (students.csv).

Train and save a Random Forest model (rf_model.pkl).

Save a simple JSON training summary (model_summary.json).

Command-line interactive menu to:

Retrain the model

Predict whether a student will pass given features

Display feature importance ranking




---

**Files produced / used.**

students.csv — generated synthetic dataset (if not present).

rf_model.pkl — serialized trained model saved with joblib.

model_summary.json — training metadata and evaluation report.

The main script file (the code you provided) runs the CLI.



---

**Requirements.**

Python 3.8+

Packages:

numpy

pandas

scikit-learn

joblib



Install via pip if needed:

pip install numpy pandas scikit-learn joblib


---

**How it works  (AI/ML workflow)).**

1. If students.csv does not exist, make_data() generates a dataset with features:

study_hours (float): hours studied per day (0–12)

attendance_rate (float): attendance percentage (50–100)

assignments_done (int): number of assignments done (0–9)

previous_score (float): previous exam score (0–100)

passed (0/1): label computed from a noisy linear function of the features



2. train() reads the dataset, splits it, trains a RandomForestClassifier, evaluates on the test set, writes a JSON summary, and saves the model to disk.


3. check_student(...) loads the saved model and returns the predicted class (0 = fail, 1 = pass) and prediction probabilities.


4. rank_features() returns the feature importances from the model.


5. run_menu() exposes a simple terminal menu for interactive use.




---

**Usage.**

Run the script from the command line:

python your_script.py

When you run it:

If no model is found, the script will train one automatically and print the accuracy.

You will see a menu with four options: Retrain, Predict student, Feature ranking, Exit.
