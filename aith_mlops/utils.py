from lakefs.client import Client
import lakefs
import pandas as pd
import json


TEST_FILENAME = "flight_delays_test.csv"
TRAIN_FILENAME = "flight_delays_train.csv"


client = Client(
    host="http://localhost:8000",
    username="AKIAIOSFOLKFSSAMPLES",
    password="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
)

repo = lakefs.Repository("repository", client=client)

def get_data(dataframe_name):
    with repo.branch("main").object(dataframe_name).reader(mode="rb") as f:
       df = pd.read_csv(f)
    return df

