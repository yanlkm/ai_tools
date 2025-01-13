import numpy as np
import pandas as pd


# data formalization

def load_and_process_data(csv_file_path):
    # load csv file
    data = pd.read_csv(csv_file_path)

    # handle missing or invalid data
    data = data.dropna(subset=["AvgTemperature"])

    # remove rows with temperature values less than -28 or greather than 92
    data = data[(data["AvgTemperature"] >= -28) & (data["AvgTemperature"] <= 92)]

    # filter by country and year
    grouped_data = data.groupby(["Country", "Year"])

    # create a list of sequences
    sequences = {}

    # loop over the grouped data
    for (country, year), group in grouped_data:
        # sort the data by month and day
        group = group.sort_values(by=["Month", "Day"])
        # get the temperature values
        temperatures = group["AvgTemperature"].values
        # add the sequence to the dictionary
        sequences[(country, year)] = temperatures

    return sequences


# create a subsequence function
def create_subsequences(sequence, seq_length):
    # create a list to hold the subsequences
    subsequences = []
    # loop over the sequence
    for i in range(len(sequence) - seq_length):
        subsequences.append(sequence[i:i + seq_length + 1])
    # return the list of subsequences
    return np.array(subsequences)


# generate the input and output data
def generate_subsequences_for_single_sequence(sequence, seq_length):
    # check if the sequence is long enough
    if len(sequence) <= seq_length:
        return None, None
    # create the subsequences
    subsequences = create_subsequences(sequence, seq_length)
    # normalize the sequence in intervals of 0 and 1
    subsequences = normalize_sequence(subsequences)
    # return the input and output data formatted
    X = subsequences[:, :-1]
    y = subsequences[:, -1]
    return X, y


# normalize the sequence
def normalize_sequence(sequence):
    min_val = np.min(sequence)
    max_val = np.max(sequence)
    return (sequence - min_val) / (max_val - min_val)
