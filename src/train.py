# training/train.py
import os
import argparse
import pandas as pd
import xgboost as xgb

def train_model(train_data_path, model_output_path):
    print("📥 Loading training data from:", train_data_path)
    df = pd.read_csv(train_data_path)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "eta": 0.1,
        "verbosity": 1
    }

    model = xgb.train(params, dtrain, num_boost_round=100)

    os.makedirs(model_output_path, exist_ok=True)
    model.save_model(os.path.join(model_output_path, "model.bst"))

    print("✅ Model saved to:", os.path.join(model_output_path, "model.bst"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # !!! SageMaker will mount the training channel at:
    # /opt/ml/input/data/train/train.csv
    parser.add_argument(
        "--train",
        type=str,
        default="/opt/ml/input/data/train/train.csv"
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="/opt/ml/model"
    )

    args = parser.parse_args()
    train_model(args.train, args.model_dir)
