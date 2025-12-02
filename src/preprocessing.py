import pandas as pd
import os
from sklearn.model_selection import train_test_split


def preprocess():
    input_path = os.path.join('/opt/ml/processing/input', 'churn.csv')
    train_output_path = os.path.join('/opt/ml/processing/train', 'train.csv')
    test_output_path = os.path.join('/opt/ml/processing/test', 'test.csv')

    df = pd.read_csv(input_path)

    print("🔍 Raw data preview:")
    print(f"Rows: {df.shape[0]}")
    print("Columns:", df.columns.tolist())
    print("\n🔍 Missing values per column:")
    print(df.isnull().sum())
    print("\n🔍 Sample rows:")
    print(df.head(5))

    # Cleaning
    df.drop(columns=["CustomerID"], errors='ignore', inplace=True)
    df = pd.get_dummies(df, columns=['Gender', 'Subscription Type', 'Contract Length'])
    df = df[df['Churn'].notna()]
    df.fillna(method='ffill', inplace=True)

    print("\n✅ Cleaned data preview:")
    print(f"Rows: {df.shape[0]}")
    print("Column dtypes:")
    print(df.dtypes)

    # Train-test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Save outputs
    os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    print(f"\n✅ Preprocessed data saved to:")
    print(f"  - Train: {train_output_path}")
    print(f"  - Test:  {test_output_path}")


if __name__ == "__main__":
    preprocess()
