import pandas as pd
import csv
import numpy as np

def write_submission(recommender, submission_name):
    print("Writing submission")
    targetUsers = pd.read_csv("../Data/data_target_users_test.csv")['user_id']
    targetUsers = targetUsers.tolist()

    path_to_open = "../Submissions/" + submission_name
    with open(path_to_open, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['user_id', 'item_list'])

        for userID in targetUsers:
            writer.writerow([userID, str(np.array(recommender.recommend(userID, 10)))[1:-1]])
