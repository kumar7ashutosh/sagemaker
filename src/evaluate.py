import argparse
import os
import pandas as pd
import tarfile
import xgboost as xgb
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-data", type=str, default="/opt/ml/processing/test")
    parser.add_argument("--model-dir", type=str, default="/opt/ml/processing/model")
    parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/evaluation")
    args = parser.parse_args()

    # ✅ Load test data
    test_file = os.path.join(args.test_data, "test.csv")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"❌ test.csv not found at {test_file}")

    df = pd.read_csv(test_file)
    X_test = df.drop("Churn", axis=1)
    y_test = df["Churn"]

    # ✅ Unpack model.tar.gz first
    tar_path = os.path.join(args.model_dir, "model.tar.gz")
    if not os.path.exists(tar_path):
        raise FileNotFoundError(f"❌ model.tar.gz not found at {tar_path}")

    with tarfile.open(tar_path) as tar:
        tar.extractall(path=args.model_dir)

    # ✅ Now load the model.bst
    model_path = os.path.join(args.model_dir, "model.bst")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ model.bst not found at {model_path}")

    model = xgb.Booster()
    model.load_model(model_path)

    dtest = xgb.DMatrix(X_test)
    preds = model.predict(dtest)
    preds_binary = [1 if p > 0.5 else 0 for p in preds]
    accuracy = accuracy_score(y_test, preds_binary)

    print(f"✅ Accuracy: {accuracy:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "evaluation.json"), "w") as f:
        f.write(f'{{"accuracy": {accuracy:.4f}}}')
