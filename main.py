from src import data_loader
from src import preprocess
def main():
    users, tasks = data_loader.get_data()
    preprocess.preprocess(users, tasks)


if __name__ == "__main__":
    main()