import lightgbm as lgb
import pandas as pd
import data_loader
import preprocess

MODEL_PATH = "./models/lgbm_model_v1.txt"

model = lgb.Booster(model_file=MODEL_PATH)
feature_names = model.feature_name()

def get_points(player_name: str, df: pd.DataFrame):
    done_df = df[(df["done"].astype(bool).astype(int) == 1) & (df["display_name"] == player_name)]
    return done_df["tier"].sum()

def recommend_tasks(player_name: str, point_threshold: int, lgbm_df: pd.DataFrame, df):
    player_df = lgbm_df[(lgbm_df["display_name"] == player_name) & (lgbm_df["done"] == 0)].copy()

    if player_df.empty:
        return []

    player_df.fillna({
        "progress_ratio": 0,
        "potential_to_save_time": 0,
        "kills_remaining": 0,
        "readiness": 0
    }, inplace=True)

    X = player_df[feature_names]
    player_df = df[(df["display_name"] == player_name) & (df["done"] == 0)].copy()
    player_df["estimated_time"] = player_df["time_to_kill"] * player_df["kills_remaining"]
    player_df["could_complete_score"] = model.predict(X)

    # scoring
    AWAKENED_TASKS = [
        "Duke Sucellus Sleeper", "Whispered", "Leviathan Sleeper", "Vardorvis Sleeper"
    ]

    player_df = player_df[player_df["slayer_gap"] >= 0]

    def score(row):
        if ((row["type"] == "Kill Count") or (row["type"] == "Speed")) and (row["task_name"] not in AWAKENED_TASKS):
            return (row["progress_ratio"] * row["tier"] / (row["estimated_time"] + 1)) * row["could_complete_score"]
        elif row["type"] in ["Perfection", "Mechanical", "Stamina"] or row["task_name"] in AWAKENED_TASKS:
            return row["tier"] * row["could_complete_score"]
        else:
            return 0

    player_df["score"] = player_df.apply(score, axis=1)
    player_df = player_df.sort_values("score", ascending=False)

    accumulated_points = get_points(player_name, df)
    selected_tasks = []

    for _, row in player_df.iterrows():
        if accumulated_points >= point_threshold:
            break
        selected_tasks.append(row)
        accumulated_points += row["tier"]

    return pd.DataFrame(selected_tasks)[[
        "task_name", "description", "monster", "tier", "type",
        "score", "kills_remaining", "seconds_to_save", "estimated_time"
    ]].to_dict(orient="records")
