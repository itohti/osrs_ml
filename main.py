import pandas as pd
from src import data_loader
from src import preprocess
def main():
    # users, tasks = data_loader.get_data()
    users = pd.read_csv('./saved_data/users')
    tasks = pd.read_csv('./saved_data/tasks')
    preprocess.preprocess(users, tasks)


if __name__ == "__main__":
    main()