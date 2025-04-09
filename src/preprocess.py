import pandas as pd
import ast

def preprocess(users_df, tasks_df):
    # users_df.to_csv('./saved_data/users.csv')
    # tasks_df.to_csv('./saved_data/tasks.csv')

    convert_comp_percentage(tasks_df)
    convert_string_to_dict(users_df)
    relate_user_to_task_df = relate_user_to_task(tasks_df, users_df)
    relate_user_to_task_df.to_csv("./saved_data/users_related_to_task.csv")
    print(users_df)
    print(tasks_df)

def convert_comp_percentage(tasks_df):
    tasks_df["comp"] = tasks_df["comp"].str.rstrip('%').astype(float) / 100

def convert_string_to_dict(users_df):
    users_df['boss_info'] = users_df['boss_info'].apply(ast.literal_eval)
    users_df['combat_stats'] = users_df['combat_stats'].apply(ast.literal_eval)
    users_df['tasks'] = users_df['tasks'].apply(ast.literal_eval)

def relate_user_to_task(tasks_df, users_df):
    users_related_tasks = {"task_name": [], "display_name": [], "attack": [], "defence": [], "strength": [], "hitpoints": [], "ranged": [], "magic": [], "prayer": [], "boss_kc": [], "ehb": [], "done": []}  

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
                kc = user["boss_info"].get(kc_query, 0)
                ehb = user["boss_info"].get(ehb_query, 0)

                if kc == 0 and ehb == 0:
                    formatted_name = f"The {formatted_name}"
                    kc = user["boss_info"].get(f"{formatted_name}_kc", 0)
                    ehb = user["boss_info"].get(f"{formatted_name}_ehb", 0)

                users_related_tasks["boss_kc"].append(kc)
                users_related_tasks["ehb"].append(ehb)
            else:
                users_related_tasks["boss_kc"].append(0)
                users_related_tasks["ehb"].append(0)

            users_related_tasks["display_name"].append(user["display_name"])

            for stat_key, stat_value in user["combat_stats"].items():
                stat_name = stat_key.split("_")[0].lower()
                users_related_tasks[stat_name].append(stat_value)

    df = pd.DataFrame(users_related_tasks)
    return df

def kill_count_task_progress(users_df, tasks_df):
    pass