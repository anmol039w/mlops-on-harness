import os, json, mlflow
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# ---- Fixed params (no envs) ----
C = 1.0
MAX_ITER = 200
ARTIFACT_DIR = "artifacts"
MODEL_NAME = "iris-logreg"

MLFLOW_TRACKING_URI = "http://mlflow.mlflow:5000"    # change if your svc differs
MLFLOW_EXPERIMENT_NAME = "mlops-demo"

# ---- MLflow config ----
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# ---- Data ----
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# ---- Training + logging ----
with mlflow.start_run():
    mlflow.log_param("C", C)
    mlflow.log_param("max_iter", MAX_ITER)

    clf = LogisticRegression(C=C, max_iter=MAX_ITER)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    out_path = f"{ARTIFACT_DIR}/{MODEL_NAME}.joblib"
    joblib.dump(clf, out_path)
    mlflow.log_artifact(out_path)

    # Emit a simple summary for downstream steps
    summary = {
        "accuracy": acc,
        "artifact_path": out_path,
        "model_name": MODEL_NAME,
    }
    with open(f"{ARTIFACT_DIR}/summary.json", "w") as f:
        json.dump(summary, f)

    print(json.dumps(summary))
