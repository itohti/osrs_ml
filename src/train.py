import pandas as pd
from src import preprocess
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
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

def get_points(player_name: str):
    df = pd.read_csv("./saved_data/merged_df.csv")
    done_df = df[(df["done"].astype(bool).astype(int) == 1) & (df["display_name"] == player_name)]
    total_points = 0
    for _,row in done_df.iterrows():
        total_points += row["tier"]
    
    return total_points

def recommend_tasks_by_score(player_name: str, point_threshold: int, filepath: str = "./saved_data/merged_df.csv"):
    df = pd.read_csv(filepath)
    df = preprocess.feature_engineering(df)
    df.to_csv("./saved_data/featured_merged_df.csv")

    player_df = df[(df["display_name"] == player_name) & (df["done"] == 0)].copy()

    player_df["progress_ratio"] = player_df["progress_ratio"].fillna(0)
    player_df["potential_to_save_time"] = player_df["potential_to_save_time"].fillna(0)
    player_df["kills_remaining"] = player_df["kills_remaining"].fillna(0) 
    player_df["readiness"] = player_df["readiness"].fillna(0)

    player_df["estimated_time"] = player_df["time_to_kill"] * player_df["kills_remaining"]

    AWAKENED_TASKS = [
        "Duke Sucellus Sleeper",
        "Whispered",
        "Leviathan Sleeper",
        "Vardorvis Sleeper"
    ]

    def score_task(row):
        if ((row["type"] == "Kill Count") or (row["type"] == "Speed")) and (row["task_name"] not in AWAKENED_TASKS):
            progress_weight = 1
            if row["progress_ratio"] < 0.85 and row["type"] == "Speed":
                progress_weight = 0.5
            return (row["progress_ratio"] * progress_weight) * (row["tier"] + row["comp"]) / (row["estimated_time"] + 1)
        elif (row["type"] == "Perfection") or (row["type"] == "Mechanical") or (row["task_name"] in AWAKENED_TASKS):
            return row["readiness"] * (row["tier"])
        else:
            return 0

    player_df["score"] = player_df.apply(score_task, axis=1)
    scaler = MinMaxScaler()
    player_df[["score"]] = scaler.fit_transform(player_df[["score"]])

    player_df = player_df.sort_values("score", ascending=False)

    selected_tasks = []
    accumulated_points = get_points(player_name)

    for _, row in player_df.iterrows():
        if accumulated_points >= point_threshold:
            break
        selected_tasks.append(row)
        accumulated_points += row["tier"]

    result_df = pd.DataFrame(selected_tasks)
    return result_df[["task_name", "description", "monster", "tier", "score", "progress_ratio", "kills_remaining", "estimated_time"]]

    

    