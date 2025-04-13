from matplotlib import axes
import numpy as np
import pandas as pd
import re
import ast
from sklearn.preprocessing import MinMaxScaler, StandardScaler

AWAKENED_TASKS = [
    "Duke Sucellus Sleeper",
    "Whispered",
    "Leviathan Sleeper",
    "Vardorvis Sleeper"
]


def preprocess(users_df, tasks_df):
    # users_df.to_csv('./saved_data/users.csv')
    # tasks_df.to_csv('./saved_data/tasks.csv')

    convert_comp_percentage(tasks_df)
    convert_string_to_dict(users_df)
    task_to_users = relate_user_to_task(tasks_df, users_df)
    task_to_users.to_csv("./saved_data/tasks_to_users.csv")
    merged_df = task_to_users.merge(tasks_df, left_on='task_name', right_on='name', how='left')
    merged_df = merged_df.drop(columns=["Unnamed: 0"], errors='ignore')
    merged_df.to_csv("./saved_data/merged_df.csv", index=False)


def feature_engineering(merged_df):
    # kills remaining feature
    merged_df = kills_feature(merged_df)
    merged_df = speed_feature(merged_df)
    merged_df = perfect_mechanical_stamina_feature(merged_df)
    merged_df = merge_progress_ratio(merged_df)
    merged_df["time_to_kill"] = merged_df.apply(lambda row: row["ehb"] / (row["boss_kc"] + 0.00001), axis=1)
    merged_df["time_to_completion"] = merged_df.apply(lambda row: row["time_to_kill"] * row["kills_remaining"], axis=1)
    return merged_df

def convert_comp_percentage(tasks_df):
    tasks_df["comp"] = tasks_df["comp"].str.rstrip('%').astype(float) / 100

def convert_string_to_dict(users_df):
    users_df['boss_info'] = users_df['boss_info'].apply(ast.literal_eval)
    users_df['combat_stats'] = users_df['combat_stats'].apply(ast.literal_eval)
    users_df['tasks'] = users_df['tasks'].apply(ast.literal_eval)

def relate_user_to_task(tasks_df, users_df):
    users_related_tasks = {"task_name": [], "display_name": [], "attack": [], "defence": [], "strength": [], "hitpoints": [], "ranged": [], "magic": [], "prayer": [], "boss_kc": [], "ehb": [], "pb": [], "done": []}  

    for _, user in users_df.iterrows():
        for task_key in user["tasks"].keys():
            format_name = task_key.split("_")[0]
            users_related_tasks["task_name"].append(format_name)
            users_related_tasks["done"].append(user["tasks"][task_key])

            monster_of_task = tasks_df.loc[tasks_df["name"] == format_name, "monster"]
            if not monster_of_task.empty and isinstance(monster_of_task.iloc[0], str):
                monster_of_task = monster_of_task.iloc[0].replace("-", " ")
                format_monster_name = monster_of_task.split(" ")
                cleaned = [word.replace("'", "").replace(":", "").capitalize() for word in format_monster_name]
                formatted_name = " ".join(cleaned)

                # Boss name mappings
                boss_name_mappings = {
                    "Corrupted Hunllef": "The Corrupted Gauntlet",
                    "Crystalline Hunllef": "The Gauntlet",
                    "Tombs Of Amascut Expert Mode": "Tombs Of Amascut Expert",
                    "The Nightmare": "Nightmare",
                    "Moons Of Peril": "Lunar Chests",
                    "Fortis Colosseum": "Sol Heredit",
                    "Barrows": "Barrows Chests",
                    "Theatre Of Blood Entry Mode": "Theatre Of Blood"
                }

                # Apply boss name mapping if exists
                formatted_name = boss_name_mappings.get(formatted_name, formatted_name)
                kc_query = f"{formatted_name}_kc"
                ehb_query = f"{formatted_name}_ehb"
                pb_query = f"{formatted_name}_pb"
                kc = user["boss_info"].get(kc_query, -2)
                # if we successfully got the kc back and if its -1 that means the user did 0 kills.
                if (kc == -1):
                    kc = 0
                ehb = user["boss_info"].get(ehb_query, 0)
                pb = user["boss_info"].get(pb_query, -1)

                # -2 will indicate that the monster is an npc thus no kc can be recorded on it.
                if kc == -2 and ehb == 0:
                    formatted_name = f"The {formatted_name}"
                    kc = user["boss_info"].get(f"{formatted_name}_kc", -2)
                    if (kc == -1):
                        kc = 0
                    elif (kc == -2):
                        kc = -1
                    ehb = user["boss_info"].get(f"{formatted_name}_ehb", 0)
                    pb = user["boss_info"].get(f"{formatted_name}_pb", -1)

                users_related_tasks["boss_kc"].append(kc)
                users_related_tasks["ehb"].append(ehb)
                users_related_tasks["pb"].append(pb)
            else:
                users_related_tasks["boss_kc"].append(-1)
                users_related_tasks["ehb"].append(0)
                users_related_tasks["pb"].append(-1)

            users_related_tasks["display_name"].append(user["display_name"])

            for stat_key, stat_value in user["combat_stats"].items():
                stat_name = stat_key.split("_")[0].lower()
                users_related_tasks[stat_name].append(stat_value)

    df = pd.DataFrame(users_related_tasks)
    return df

def perfect_mechanical_stamina_feature(merged_df):
    # Tasks that are either labeled as Perfection/Mechanical or are special cases
    is_relevant = (
        merged_df["type"].isin(["Perfection", "Mechanical", "Stamina"]) |
        merged_df["task_name"].isin(AWAKENED_TASKS)
    )
    
    perfect_mechanical_tasks = merged_df[is_relevant]

    tasks_to_comp = dict(zip(perfect_mechanical_tasks["name"], perfect_mechanical_tasks["comp"]))

    def compute_readiness(row):
        task_name = row["task_name"]
        if task_name in tasks_to_comp:
            return row["ehb"] * (tasks_to_comp[task_name] ** 1.5)
        
    merged_df["readiness"] = merged_df.apply(compute_readiness, axis=1)

    return merged_df

def kills_feature(merged_df):
    kill_counts_tasks = merged_df.loc[merged_df["type"] == "Kill Count"]

    kill_counts_description = dict(zip(kill_counts_tasks["name"], kill_counts_tasks["description"]))

    def compute_kills_remaining(row):
        task_name = row["task_name"]
        done = row["done"]
        if task_name in kill_counts_description:
            if row["boss_kc"] == -1:
                # infer the player only needs to kill one
                if done:
                    return 0
                else:
                    return 1
            
            # extract the amount needed
            description = kill_counts_description[task_name]
            if re.search(r"\bonce\b", description, re.IGNORECASE):
                return max(1 - row["boss_kc"], 0)
            match = re.search(r"(\d+)\s+times?", description, re.IGNORECASE)
            if match:
                return max(int(match.group(1)) - row["boss_kc"], 0)
            if re.search(r"\bKill a\b", description, re.IGNORECASE):
                return max(1 - row["boss_kc"], 0)
            if re.search(r"\bKill the\b", description, re.IGNORECASE):
                return max(1 - row["boss_kc"], 0)
            # lol hard coded jankkkkk
            if description == "Open the Reward Chest after defeating all three Moons.":
                return max(1 - row["boss_kc"], 0)
            
            # maybe just infer 1 kc is needed?
            return 1
        else:
            return None
        
    def compute_progress_ratio(row):
        if (row["boss_kc"] + row["kills_remaining"] == 0):
            return 0
        return row["boss_kc"] / (row["boss_kc"] + row["kills_remaining"])
    
    merged_df["kills_remaining"] = merged_df.apply(compute_kills_remaining, axis=1)
    merged_df["kills_remaining_progress_ratio"] = merged_df.apply(compute_progress_ratio, axis=1)
    return merged_df

def speed_feature(merged_df):
    speed_running_tasks = merged_df.loc[merged_df["type"] == "Speed"]
    speed_running_description = dict(zip(speed_running_tasks["name"], speed_running_tasks["description"]))

    def compute_seconds_to_save(row):
        task_name = row["task_name"]
        if task_name in speed_running_description:
            def get_time(description):
                min_to_sec = 0
                secs = 0
                match_minutes = re.search(r"(\d+)\s+minutes?", description, re.IGNORECASE)
                match_minute = re.search(r"(\d+)\s+minute?", description, re.IGNORECASE)
                match_mins = re.search(r"(\d+)\s+mins?", description, re.IGNORECASE)
                match_min = re.search(r"(\d+)\s+min?", description, re.IGNORECASE)
                if match_minutes:
                    min_to_sec = int(match_minutes.group(1)) * 60
                if match_minute:
                    min_to_sec = int(match_minute.group(1)) * 60
                if match_mins:
                    min_to_sec = int(match_mins.group(1)) * 60
                if match_min:
                    min_to_sec = int(match_min.group(1)) * 60
                match_seconds = re.search(r"(\d+)\s+seconds?", description, re.IGNORECASE)
                if match_seconds:
                    secs = int(match_seconds.group(1))
                match_time_format = re.search(r"(\d+):(\d+)", description, re.IGNORECASE)
                if match_time_format:
                    min_to_sec = int(match_time_format.group(1)) * 60
                    secs = int(match_time_format.group(2))
                # Hard coded jankkkkkkkkkkk
                if description == "Complete a Chambers of Xeric Challenge mode raid in the target time.":
                    return 50 * 60 # just assume they're running a 3 man CM
                if description == "Complete the Theatre of Blood: Hard Mode within the challenge time.":
                    return 27 * 60 # assuming they're running a 4 man
                return min_to_sec + secs


            target_time = get_time(speed_running_description[task_name])
            # this mean that the player does not have a recorded time
            if row["pb"] == -1:
                return target_time
            return row["pb"] - target_time
        
    def compute_progress_ratio(row):
        pb = row["pb"]
        if (pb == -1):
            pb = 0
        if (pb + row["seconds_to_save"] == 0):
            return 1
        else:
            return min(pb / (pb + row["seconds_to_save"]), 1)
        
    def compute_seconds_to_save_per_ehb(row):
        return row["seconds_to_save"] / (row["ehb"] + 0.000001)

    
    merged_df["seconds_to_save"] = merged_df.apply(compute_seconds_to_save, axis=1)
    merged_df["speed_progress_ratio"] = merged_df.apply(compute_progress_ratio, axis=1)
    merged_df["potential_to_save_time"] = merged_df.apply(compute_seconds_to_save_per_ehb, axis=1)
    clip_upper = merged_df["potential_to_save_time"].quantile(0.95)
    merged_df["potential_to_save_time"] = merged_df["potential_to_save_time"].clip(lower=0, upper=clip_upper)

    speed_mask = merged_df["type"] == "Speed"
    scaler = MinMaxScaler()

    scaled_values = scaler.fit_transform(
        merged_df.loc[speed_mask, "potential_to_save_time"].values.reshape(-1, 1)
    )

    merged_df.loc[speed_mask, "potential_to_save_time"] = scaled_values
    return merged_df

def merge_progress_ratio(merged_df):
    merged_df["progress_ratio"] = merged_df["kills_remaining_progress_ratio"].combine_first(merged_df["speed_progress_ratio"])
    merged_df.drop(["kills_remaining_progress_ratio", "speed_progress_ratio"], axis=1, inplace=True)

    return merged_df