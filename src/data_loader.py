import firebase_admin
from firebase_admin import credentials, firestore
import pandas as pd

_app = None

def init_firestore():
    global _db
    if not firebase_admin._apps:
        cred = credentials.Certificate("./secrets/caroadmap-firebase-adminsdk-fbsvc-ace27393f8.json")
        _app = firebase_admin.initialize_app(cred)

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

        combat_stats = db.collection("users").document(display_name).collection("combat_stats").stream()
        combat_stats_info = {}
        for doc in combat_stats:
            combat_stats_info[doc.id] = doc.to_dict()
        
        tasks = db.collection("users").document(display_name).collection("tasks").stream()
        tasks_completed_info = {}
        for doc in tasks:
            tasks_completed_info[doc.id] = doc.to_dict()

        user_data["boss_info"] = boss_info_data
        user_data["combat_stats"] = combat_stats_info
        user_data["tasks"] = tasks_completed_info

        all_users_data.append(user_data)

    tasks = [doc.to_dict() for doc in db.collection('tasks').stream()]

    return all_users_data, tasks