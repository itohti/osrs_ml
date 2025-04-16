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

    # Convert "done" column to boolean
    df["done"] = df["done"].astype(bool).astype(int)

    df["readiness"] = df["readiness"].fillna(0)

    features = [
        "attack", "defence", "strength", "hitpoints", "ranged", "magic", "prayer", "tier", "comp", "readiness", "ehb", "boss_kc"
    ]

    X = df[features]
    y = df["done"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    df["could_complete_score"] = model.predict_proba(X)[:, 1]

    # save it so recommend_tasks_by_score and use it.
    df.to_csv("./saved_data/featured_merged_df.csv", index=False)

    for feature, weight in zip(features, model.coef_[0]):
        print(f"{feature}: {weight:.4f}")

def get_points(player_name: str):
    df = pd.read_csv("./saved_data/merged_df.csv")
    done_df = df[(df["done"].astype(bool).astype(int) == 1) & (df["display_name"] == player_name)]
    total_points = 0
    for _,row in done_df.iterrows():
        total_points += row["tier"]
    
    return total_points

def recommend_tasks_by_score(player_name: str, point_threshold: int, filepath: str = "./saved_data/featured_merged_df.csv"):
    df = pd.read_csv(filepath)

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

    player_df = player_df[player_df["slayer_gap"] >= 0]

    def score_task(row):
        if ((row["type"] == "Kill Count") or (row["type"] == "Speed")) and (row["task_name"] not in AWAKENED_TASKS):
            return (row["progress_ratio"] * row["tier"] / (row["estimated_time"] + 1)) * row["could_complete_score"]
        elif (row["type"] == "Perfection") or (row["type"] == "Mechanical") or (row["type"] == "Stamina") or (row["task_name"] in AWAKENED_TASKS):
            return row["tier"] * row["could_complete_score"]
        else:
            return 0

    player_df["score"] = player_df.apply(score_task, axis=1)

    player_df = player_df.sort_values("score", ascending=False)

    selected_tasks = []
    accumulated_points = get_points(player_name)

    for _, row in player_df.iterrows():
        if accumulated_points >= point_threshold:
            break
        selected_tasks.append(row)
        accumulated_points += row["tier"]

    result_df = pd.DataFrame(selected_tasks)
    return result_df[["task_name", "description", "monster", "tier", "type", "score", "kills_remaining", "seconds_to_save", "estimated_time"]]

    

    