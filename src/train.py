import pandas as pd
from src import preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np

def classification_model(player_name=None):
    df = pd.read_csv("./saved_data/merged_df.csv")
    df = preprocess.feature_engineering(df)

    # just to see the featured merged df
    df.to_csv("./saved_data/featured_merged_df.csv")
    # Convert "done" column to boolean
    df["done"] = df["done"].astype(bool).astype(int)

    df["label"] = ((df["seconds_to_save"] <= 10) | (df["kills_remaining"] <= 20)).astype(int)

    df.fillna(0, inplace=True)

    features = [
        "attack", "defence", "strength", "hitpoints", "ranged", "magic", "prayer", "done", "tier",
        "comp", "kills_remaining", "progress_ratio", "potential_to_save_time"
    ]

    X = df[features]
    y = df["label"]

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build and train logistic regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))


    df["completion_score"] = model.predict_proba(X)[:, 1]

    top_tasks = df[df["done"] == 0].sort_values("completion_score", ascending=False)
    top_tasks.to_csv("./saved_data/logistic_regression_output.csv")
    print(top_tasks[["task_name", "display_name", "completion_score", "kills_remaining", "seconds_to_save"]])

    for feature, coef in zip(features, model.coef_[0]):
        print(f"{feature}: {coef:.4f}")

    

    