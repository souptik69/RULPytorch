import pandas as pd
import numpy as np
from add_remaining_useful_life import *




def moving_average_filter(data, window_size=5):
    return data.rolling(window=window_size, min_periods=1).mean()

def create_sequences(data, sequence_length=30):
    sequences = []
    for start in range(len(data) - sequence_length + 1):
        end = start + sequence_length
        sequences.append(data[start:end].values)
    return np.array(sequences)


def loading_FD001():

    # define filepath to read data
    dir_path = './CMAPSSData/'

    # define column names for easy indexing
    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i) for i in range(1, 22)]
    col_names = index_names + setting_names + sensor_names

    # read data
    train = pd.read_csv((dir_path + 'train_FD001.txt'), sep='\s+', header=None, names=col_names)
    test = pd.read_csv((dir_path + 'test_FD001.txt'), sep='\s+', header=None, names=col_names)
    y_test = pd.read_csv((dir_path + 'RUL_FD001.txt'), sep='\s+', header=None, names=['RUL'])

    # drop non-informative features in training set
    drop_sensors = ['s_1', 's_5', 's_6', 's_10', 's_16', 's_18', 's_19']
    drop_labels = setting_names + drop_sensors
    train.drop(labels=drop_labels, axis=1, inplace=True)

    # separate title information and sensor data
    title = train.iloc[:, 0:2]
    data = train.iloc[:, 2:]

    # min-max normalization of the sensor data
    data_norm = (data - data.min()) / (data.max() - data.min())
    train_norm = pd.concat([title, data_norm], axis=1)

    # add piece-wise target remaining useful life
    train_norm = add_remaining_useful_life(train_norm)
    train_norm['RUL'].clip(upper=125, inplace=True) # in the paper the MAX RUL is mentioned as 125

    # group the training set with unit
    group = train_norm.groupby(by="unit_nr")

    # process each unit separately to create sequences
    train_sequences = []
    train_labels = []
    for unit_nr, unit_data in group:
        unit_data_sorted = unit_data.sort_values('time_cycles')
        unit_sequences = create_sequences(unit_data_sorted.iloc[:, 2:], sequence_length=30)
        train_sequences.extend(unit_sequences)
        train_labels.extend(unit_data_sorted['RUL'].iloc[29:].values)  # Get RUL for each sequence

    # drop non-informative features in testing set
    test.drop(labels=drop_labels, axis=1, inplace=True)
    title = test.iloc[:, 0:2]
    data = test.iloc[:, 2:]

        # apply moving average filter
    data_filtered = moving_average_filter(data)

    # min-max normalization of the sensor data
    data_norm = (data_filtered - data_filtered.min()) / (data_filtered.max() - data_filtered.min())
    test_norm = pd.concat([title, data_norm], axis=1)

    # group the testing set with unit
    group_test = test_norm.groupby(by="unit_nr")

    # process each unit in test set to create sequences
    test_sequences = []
    for unit_nr, unit_data in group_test:
        unit_data_sorted = unit_data.sort_values('time_cycles')
        unit_sequences = create_sequences(unit_data_sorted.iloc[:, 2:], sequence_length=30)
        test_sequences.extend(unit_sequences)

    return np.array(train_sequences), np.array(train_labels), np.array(test_sequences), y_test


    # data_norm = (data - data.min()) / (data.max() - data.min())
    # test_norm = pd.concat([title, data_norm], axis=1)

    # # group the testing set with unit
    # group_test = test_norm.groupby(by="unit_nr")

    # return group, y_test, group_test
