import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

def classification_model(player_name=None):
    df = pd.read_csv("./saved_data/merged_df.csv")

    # Convert "done" column to boolean
    df["done"] = df["done"].astype(bool).astype(int)

    