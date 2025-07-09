import os.path
import random
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score




# First we split the dataset into 20 % for test and 80 % for training
# I will split the data based on the products

def split_data():
    data_file = "data_fakes.csv"
    df = pd.read_csv(data_file, sep=",")

    all_products = df["ProductName"].unique().tolist()
    test_products_size = len(all_products) * 2 // 10

    test_products = random.sample(all_products, test_products_size)
    train_products = [product for product in all_products if product not in test_products]

    df_test = df[df["ProductName"].isin(test_products)]
    df_train = df[df["ProductName"].isin(train_products)]
    df_test.to_csv("test.csv")
    df_train.to_csv("train.csv")

    return df_test, df_train


def extract_features(df):
    """
    The function extract freatures from a dataframe and adds it as a column. As features we use the len of the description
    of the product and is average price
    :param df: input split dataframe that contains the features and the class Fake
    :return: a dataframe that contains the new features
    """
        df["len-desc"] = df["product_description"].map(lambda x: len(x))
    #X = df["len-desc"].values.reshape(-1, 1)
    X = df[["len-desc","AveragePrice"]].values
    Y = df["Fake"].values
    print(X.shape)
    print(Y.shape)
    return X, Y

def develop_baseline( y_test):
    """
    Develop a random baseline return the f1-score on the fake class  on the test data
    :return:
    """
    y_predict = np.random.randint(0,2,y_test.shape)
    print(y_predict)
    f1 = f1_score(y_test, y_predict, labels=[1])
    return f1
def develop_classifier(X_test, y_test, X_train, y_train, depth):
    """
    Develop a random forest classifier that with a maximum depth of depth.
    Evaluate it on the y_test and return the f1-score on the fake class
    :param X_test: test features
    :param y_test: test fake values
    :param X_train: train features
    :param y_train: train values
    :param depth:
    :return:
    """
    classifier = RandomForestClassifier(max_depth=depth, random_state=0)
    classifier.fit(X_train, y_train)
    y_predict = classifier.predict(X_test)
    f1 = f1_score(y_test, y_predict, labels=[1])
    return classifier, f1



def main():

    if not os.path.exists("train.csv"):
        df_test, df_train = split_data()
    else:
        df_test = pd.read_csv("test.csv")
        df_train = pd.read_csv("train.csv")

    X_test, y_test = extract_features(df_test)
    X_train, y_train = extract_features(df_train)

    f1_baseline = develop_baseline(y_test)
    print(f"The f1 score of the baseline on the fake class is {f1_baseline}")
    max_f1 = 0
    best_depth = None
    for depth in range(10,22):

        classifier, f1_classifier  = develop_classifier(X_test, y_test, X_train, y_train, depth)
        if f1_classifier > max_f1:
            max_f1 = f1_classifier
            best_depth = depth
    print(f"The f1 score of a random forest with {best_depth} depth is {f1_classifier}")



main()
