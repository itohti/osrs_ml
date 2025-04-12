import pandas as pd
from src import data_loader
from src import preprocess
from src import train
def main():
    # users, tasks = data_loader.get_data()
    users = pd.read_csv('./saved_data/users.csv')
    tasks = pd.read_csv('./saved_data/tasks.csv')
    preprocess.preprocess(users, tasks)

    train.recommend_tasks_by_score("Him-alayan", 1014).to_csv("./saved_data/recommended_tasks.csv")



if __name__ == "__main__":
    main()