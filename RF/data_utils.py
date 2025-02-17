import numpy as np
from sklearn.preprocessing import StandardScaler


def split_train_test(data, train_size=0.75):
    """
    Splits the data into train and test sets.

    Args:
        data (pandas.DataFrame): The data to split.
        train_size (float): The proportion of the data to use for training.

    Returns:
        tuple: Two pandas.DataFrames, the train and test sets.
    """
    train_size = int(len(data) * train_size)
    train = data[:train_size]
    test = data[train_size:]

    return train, test


def scale_data(train, test=None):
    """
    Scales the data using the Standard Scaler.

    Args:
        train (pandas.DataFrame): The training data to scale.
        test (pandas.DataFrame): The testing data to scale. Optional.

    Returns:
        tuple: The scaler used for scaling and the scaled training and testing data.
    """
    scaler = StandardScaler()
    scaler.fit(train)

    train_scaled = scaler.transform(train)

    if test is not None:
        test_scaled = scaler.transform(test)
        return scaler, train_scaled, test_scaled

    return scaler, train_scaled


def prepare_data_for_rf(data):
    """
    Prepares the data for a RandomForest model.

    Args:
        data (pandas.DataFrame): The data to prepare.

    Returns:
        tuple: The input (X) and output (Y) data arrays for RandomForest.
    """
    # Assume that the last column is the target (Y) and others are features (X)
    X = data.drop(columns=['temp'])  # Replace 'temp' with your target column
    y = data['temp']  # Replace 'temp' with your target column

    return X, y
