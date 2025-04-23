import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

_app = None

def init_firestore():
    global _db
    if not firebase_admin._apps:
        cred = credentials.Certificate("./secrets/caroadmap-firebase-adminsdk-fbsvc-ace27393f8.json")
        _app = firebase_admin.initialize_app(cred)

def flatten_data(data):
    flatten = {}
    for key, value in data.items():
        for k, v in value.items():
            flatten[f'{key}_{k}'] = v
    return flatten

def get_data():
    init_firestore()
    db = firestore.client()

    all_users_data = []

    users_ref = db.collection('users').stream()
    for user in users_ref:
        user_data = user.to_dict()
        display_name = user_data.get("display_name")

        boss_kc = db.collection("users").document(display_name).collection("boss_info").stream()
        boss_info_data = {}
        for doc in boss_kc:
            boss_info_data[doc.id] = doc.to_dict()

        boss_info_data = flatten_data(boss_info_data)

        combat_stats = db.collection("users").document(display_name).collection("combat_stats").stream()
        combat_stats_info = {}
        for doc in combat_stats:
            skill_doc = doc.to_dict()
            skill_name = skill_doc.get("skill_name")
            level = skill_doc.get("level")
            if skill_name and level is not None:
                combat_stats_info[f"combat_{skill_name.lower()}"] = level

        tasks = db.collection("users").document(display_name).collection("tasks").stream()
        tasks_completed_info = {}
        for doc in tasks:
            tasks_completed_info[doc.id] = doc.to_dict()

        tasks_completed_info = flatten_data(tasks_completed_info)

        user_data["boss_info"] = boss_info_data
        user_data["combat_stats"] = combat_stats_info
        user_data["tasks"] = tasks_completed_info

        all_users_data.append(user_data)

    tasks = [doc.to_dict() for doc in db.collection('tasks').stream()]

    users_df = pd.DataFrame(all_users_data)
    task_df = pd.DataFrame(tasks)

    return users_df, task_df

def get_data_by_user(user: str):
    init_firestore()
    db = firestore.client()

    all_users_data = []

    user_ref = db.collection('users').document(user).get()
    user_data = user_ref.to_dict()

    if not user_data:
        print(f"\u26a0\ufe0f No user found with display name: {user}")
        return pd.DataFrame(), pd.DataFrame()

    display_name = user_data.get("display_name")

    boss_kc = db.collection("users").document(display_name).collection("boss_info").stream()
    boss_info_data = {}
    for doc in boss_kc:
        boss_info_data[doc.id] = doc.to_dict()

    boss_info_data = flatten_data(boss_info_data)

    combat_stats = db.collection("users").document(display_name).collection("combat_stats").stream()
    combat_stats_info = {}
    for doc in combat_stats:
        skill_doc = doc.to_dict()
        skill_name = skill_doc.get("skill_name")
        level = skill_doc.get("level")
        if skill_name and level is not None:
            combat_stats_info[f"combat_{skill_name}"] = level

    tasks = db.collection("users").document(display_name).collection("tasks").stream()
    tasks_completed_info = {}
    for doc in tasks:
        tasks_completed_info[doc.id] = doc.to_dict()

    tasks_completed_info = flatten_data(tasks_completed_info)

    user_data["boss_info"] = boss_info_data
    user_data["combat_stats"] = combat_stats_info
    user_data["tasks"] = tasks_completed_info

    all_users_data.append(user_data)

    tasks = [doc.to_dict() for doc in db.collection('tasks').stream()]

    users_df = pd.DataFrame(all_users_data)
    task_df = pd.DataFrame(tasks)

    return users_df, task_df