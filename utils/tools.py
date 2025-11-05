import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold



def self_split_with_scaler(file_dir, classification_num=3, drop_columns=None, rs=43, test_size=0.2, fold_num=5,
                           return_num=0):
    
    """
    input: file_dir: the file directory of the dataset
        classification_num: the number of classification, 3 or 4
        drop_columns: the columns need to be dropped
        rs: random seed
        test_size: the size of testing data
        fold_num: the number of folders
        return_num: the number of folder to return
    
    return: X_train, X_test, y_train, y_test

    """

    kf = KFold(n_splits=5, shuffle=True, random_state=rs)
    
    df = pd.read_csv(file_dir)
    df = df.dropna()
    
    if classification_num == 3:
        df = df[df['level'] != 0]
    elif classification_num == 4:
        df = df
    else:
        print('Error: classification_num must be 3 or 4')
        return None
    
    df = df.iloc[:, 1:]  ## remove the first column

    if drop_columns is not None:
        df = df.drop(drop_columns, axis=1)
    
    ## based on numbers of folders to creat cross-validation datasets, default is 5 folders
    ## return the training and testing data, and the labels

    for idx, (train_index, test_index) in enumerate(kf.split(df)):
        if idx == return_num:
            train = df.iloc[train_index]
            test = df.iloc[test_index]
            break

    # train = df.sample(frac=1 - test_size, random_state=rs)
    # test = df.drop(train.index)

    X_train = train.drop(['level'], axis=1)
    y_train = train['level']

    X_test = test.drop(['level'], axis=1)
    y_test = test['level']

    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def cross_val_split(input_df, random_state=42, fold_num=5, return_num=0, saclared=True):
    scaler = None
    kf = KFold(n_splits=fold_num, shuffle=True, random_state=random_state)
    # for train_index, test_index in kf.split(input_df):
    #     train = input_df.iloc[train_index]
    #     test = input_df.iloc[test_index]
    #     yield train, test
    for idx, (train_index, test_index) in enumerate(kf.split(input_df)):
        if idx == return_num:
            train = input_df.iloc[train_index]
            test = input_df.iloc[test_index]
            break
    
    X_train = train.drop(['level'], axis=1)
    X_train_columns = X_train.columns

    y_train = train['level']

    X_test = test.drop(['level'], axis=1)
    X_test_columns = X_test.columns
    y_test = test['level']
    

    if saclared:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        ## transform data from ndarray to DataFrame
        X_train = pd.DataFrame(X_train, columns=X_train_columns, )
        X_test = pd.DataFrame(X_test, columns=X_test_columns)
        scaler = scaler
    return X_train, X_test, y_train, y_test, scaler

