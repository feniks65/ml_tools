import pandas as pd

def read_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    test = test.drop(columns=['id'])
    return train, test

def split_data(df_for_split, target=None):
    training_data = df_for_split.sample(frac=0.7, random_state=25)
    validation_data = df_for_split.drop(training_data.index)
    y_tr = training_data[target]
    y_va = validation_data[target]
    training_data = training_data.drop(columns=[target, 'id'])
    validation_data = validation_data.drop(columns=[target, 'id'])
    return training_data, validation_data, y_tr, y_va
