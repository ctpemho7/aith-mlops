from lakefs.client import Client
import lakefs
import pandas as pd


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

    
def transform_data(train, test):
    train['Month'] = train['Month'].str[2:].astype('int')
    train['DayofMonth'] = train['DayofMonth'].str[2:].astype('int')
    train['DayOfWeek'] = train['DayOfWeek'].str[2:].astype('int')
    train['DepTime_hour'] =  train['DepTime'] // 100
    train['DepTime_minute'] =  train['DepTime'] % 100

    test['Month'] = test['Month'].str[2:].astype('int')
    test['DayofMonth'] = test['DayofMonth'].str[2:].astype('int')
    test['DayOfWeek'] = test['DayOfWeek'].str[2:].astype('int')
    test['DepTime_hour'] =  test['DepTime'] // 100
    test['DepTime_minute'] =  test['DepTime'] % 100
    train['s_hour'] = np.sin(2*np.pi*train['DepTime_hour'] / 24)
    train['c_hour'] = np.cos(2*np.pi*train['DepTime_hour'] / 24)

    test['s_hour'] = np.sin(2*np.pi*test['DepTime_hour']/24)
    test['c_hour'] = np.cos(2*np.pi*test['DepTime_hour']/24)
    train['morning'] = ((train['DepTime_hour'] >= 6) & (train['DepTime_hour'] < 12)).astype('int')
    train['day'] = ((train['DepTime_hour'] >= 12) & (train['DepTime_hour'] < 18)).astype('int')
    train['evening'] = ((train['DepTime_hour'] >= 18) & (train['DepTime_hour'] < 24)).astype('int')
    train['night'] = ((train['DepTime_hour'] >= 0) & (train['DepTime_hour'] < 6)).astype('int')

    train['low_delay'] = ((train['DepTime_hour'] >= 4) & (train['DepTime_hour'] < 9)).astype('int')
    train['high_delay'] = ((train['DepTime_hour'] >= 4) & (train['DepTime_hour'] < 9)).astype('int')

    test['morning'] = ((test['DepTime_hour'] >= 6) & (test['DepTime_hour'] < 12)).astype('int')
    test['day'] = ((test['DepTime_hour'] >= 12) & (test['DepTime_hour'] < 18)).astype('int')
    test['evening'] = ((test['DepTime_hour'] >= 18) & (test['DepTime_hour'] < 24)).astype('int')
    test['night'] = ((test['DepTime_hour'] >= 0) & (test['DepTime_hour'] < 6)).astype('int')
    test['low_delay'] = ((test['DepTime_hour'] >= 4) & (test['DepTime_hour'] < 9)).astype('int')
    test['other_time'] = ((test['DepTime_hour'] < 9) & (test['DepTime_hour'] >=5)).astype('int')
    test['delay_time'] = ((test['DepTime_hour'] >= 13) & (test['DepTime_hour'] < 24) | (test['DepTime_hour'] < 5)).astype('int')
    test['middle_time'] = ((test['DepTime_hour'] >= 9) & (test['DepTime_hour'] < 13)).astype('int')
    train['delay_time'] = ((train['DepTime_hour'] >= 13) & (train['DepTime_hour'] < 24) | (train['DepTime_hour'] < 5)).astype('int')
    train['other_time'] = ((train['DepTime_hour'] < 9) & (train['DepTime_hour'] >=5)).astype('int')
    train['middle_time'] = ((train['DepTime_hour'] >= 9) & (train['DepTime_hour'] < 13)).astype('int')
    train['month_x'] = ((train['Month'].isin([12, 6, 7]))).astype('int')
    test['month_x'] = ((test['Month'].isin([12, 1]))).astype('int')

    train['month_y'] = ((train['Month'].isin([4, 5, 9, 2]))).astype('int')
    test['month_y'] = ((test['Month'].isin([12, 1]))).astype('int')
    train['winter'] = ((train['Month'].isin([12, 1, 2]))).astype('int')
    test['winter'] = ((test['Month'].isin([12, 1, 2]))).astype('int')

    train['spring'] = ((train['Month'].isin([3, 4, 5]))).astype('int')
    test['spring'] = ((test['Month'].isin([3, 4, 5]))).astype('int')

    train['summer'] = ((train['Month'].isin([6, 7, 8]))).astype('int')
    test['summer'] = ((test['Month'].isin([6, 7, 8]))).astype('int')

    train['autumn'] = ((train['Month'].isin([9, 10, 11]))).astype('int')
    test['autumn'] = ((test['Month'].isin([9, 10, 11]))).astype('int')
    train['week_high'] = ((train['DayOfWeek'].isin([4, 5, 1, 7]))).astype('int')
    test['week_high'] = ((test['DayOfWeek'].isin([12, 1, 2]))).astype('int')

    train['week_low'] = ((train['DayOfWeek'].isin([6, 2, 3]))).astype('int')
    test['week_low'] = ((test['DayOfWeek'].isin([12, 1, 2]))).astype('int')


    cols = ['Month', 'DayofMonth', 'DayOfWeek', 's_hour', 'c_hour', 'Distance', ]
    X_train = train[cols + ['dep_delayed_15min']]
    X_test = test[cols]

    return X_train, X_test


def train_model(train, test):
    cols = ['Month', 'DayofMonth', 'DayOfWeek', 's_hour', 'c_hour', 'Distance']
    X_train = train[cols]
    X_test = test[cols]
    y_train = y

    np.random.seed(51)
    X_train_part, X_valid, y_train_part, y_valid = train_test_split(
        X_train, y_train, test_size=0.3
    )

    print("Обучение модели")

    xgb_model = XGBClassifier()
    xgb_model.fit(X_train_part, y_train_part)

    return xgb_model, X_valid, y_valid

def test_model(xgb_model, X_valid, y_valid):
    preds = xgb_model.predict(X_valid)
    preds_probas = xgb_model.predict_proba(X_valid)[:, 1]
    print("Результат обучения:")
    print("Accuracy:", accuracy_score(y_valid, preds))
    print("Precision:", precision_score(y_valid, preds))
    print("Recall:", recall_score(y_valid, preds))
    print("MSE:", mean_squared_error(y_valid, preds))
    print("ROC-AUC:", roc_auc_score(y_valid, preds_probas))

