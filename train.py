import pandas as pd
import preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
import lightgbm as lgb
import numpy as np

def classification_model(player_name=None):
    df = pd.read_csv("./saved_data/merged_df.csv")
    df = preprocess.feature_engineering(df)

    # Prep labels.
    df["done"] = df["done"].astype(bool).astype(int)
    df["readiness"] = df["readiness"].fillna(0)

    original_df = df.copy()

    df = pd.get_dummies(df, columns=["task_name", "type"])
    df.columns = df.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)
    non_feature_cols = ["display_name", "done", "name", "description", "monster", "progress_ratio", "kills_remaining", "seconds_to_save", "time_to_completion", "time_to_kill", "potential_to_save_time", "slayer_gap", "slayerReq"]
    features = [
        col for col in df.columns if col not in non_feature_cols
    ]

    X = df[features]
    y = df["done"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test)

    model = lgb.train(
        {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1
        },
        train_set=train_data,
        valid_sets=[val_data],
        valid_names=["validation"],
        num_boost_round=100,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=10)
        ]
    )

    # y_probs = model.predict(X_test)
    # y_preds = (y_probs >= 0.5).astype(int)

    # print(classification_report(y_test, y_preds))

    original_df["could_complete_score"] = model.predict(X)

    # save it so recommend_tasks_by_score and use it.
    original_df.to_csv("./saved_data/featured_merged_df.csv", index=False)

    for feature, weight in zip(features, model.feature_importance()):
        print(f"{feature}: {weight:.4f}")

    model.save_model("./models/lgbm_model_v1.txt")