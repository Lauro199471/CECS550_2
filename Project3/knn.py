import pandas as pd
import numpy as np
import operator
from dask.distributed import Client, LocalCluster
from dask import compute, delayed
import dask
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import euclidean_distances


# calculate the Euclidean distance between two vectors
def l2_distance(instance1, instance2):
    instance1 = np.array(instance1)
    instance2 = np.array(instance2)
    return np.linalg.norm(instance1 - instance2)


def data_no_nan(data):
    ## Red Data
    red_Data = data[data[data.columns.tolist()[0]] == 'red'][data.columns.tolist()[:]]
    red_Data.reset_index(drop=True, inplace=True)

    red_Data_raw = red_Data.loc[:, "fixed acidity":"quality"]  # get rid of label col
    red_Data_raw = red_Data_raw.values

    # Obtain mean of columns as you need, nanmean is just convenient.
    col_mean = np.nanmean(red_Data_raw, axis=0)

    # Find indicies that you need to replace
    inds = np.where(np.isnan(red_Data_raw))

    # Place column means in the indices. Align the arrays using take
    red_Data_raw[inds] = np.take(col_mean, inds[1])

    pd_red_Data_raw = pd.DataFrame(data=red_Data_raw, columns=data.columns.tolist()[1:])

    newRed = pd.concat([red_Data[red_Data.columns.tolist()[0]], pd_red_Data_raw], axis=1)
    # ====================================================================================
    ## White Data
    white_Data = data[data[data.columns.tolist()[0]] == 'white'][data.columns.tolist()[:]]
    white_Data.reset_index(drop=True, inplace=True)

    white_Data_raw = white_Data.loc[:, "fixed acidity":"quality"]  # get rid of label col
    white_Data_raw = white_Data_raw.values

    # Obtain mean of columns as you need, nanmean is just convenient.
    col_mean = np.nanmean(white_Data_raw, axis=0)

    # Find indicies that you need to replace
    inds = np.where(np.isnan(white_Data_raw))

    # Place column means in the indices. Align the arrays using take
    white_Data_raw[inds] = np.take(col_mean, inds[1])

    pd_white_Data_raw = pd.DataFrame(data=white_Data_raw, columns=data.columns.tolist()[1:])

    newWhite = pd.concat([white_Data[white_Data.columns.tolist()[0]], pd_white_Data_raw], axis=1)

    newData = pd.concat([newWhite, newRed], axis=0)
    newData.reset_index(drop=True, inplace=True)
    return newData


# Abiding to scikit-rules: for cross_val_score
# All arguments of __init__must have default value,
#   so it's possible to initialize the classifier just by typing MyClassifier()
# No confirmation of input parameters should be in __init__ method! That belongs to fit method.
# Do not take data as argument here! It should be in fit method.
class Knn(BaseEstimator):
    def __init__(self, k=3):
        self.data = None
        self.k = k
        self.target = None

    def fit(self, data, target):
        self.data = data
        self.data = data
        self.target = target
        return self

    def predict(self, instance_data):
        distances = {}
        # Get l2_norm for all points from instance point
        instance1 = instance_data
        for dt_pt in range(len(self.data)):
            instance2 = self.data.iloc[dt_pt]
            dist = l2_distance(instance1, instance2)
            distances[str(dt_pt)] = dist
        sorted_d = sorted(distances.items(), key=operator.itemgetter(1))

        # Get row index of training data up to K
        neighbors_row_index = []
        for i in range(self.k):
            neighbors_row_index.append(sorted_d[i][0])

        # Count Classes associated with these row index
        labels = []
        # Get label list
        for row_index_class in neighbors_row_index:
            label = self.target.iloc[int(row_index_class)]
            labels.append(label)
        unique_elements, counts_elements = np.unique(labels, return_counts=True)

        # Get label freq.
        freq = {}
        for i in range(len(unique_elements)):
            freq[unique_elements[i]] = int(counts_elements[i])
        # Return Label with high freq
        prediction = sorted(freq.items(), key=operator.itemgetter(1), reverse=True)
        return prediction[0][0]

    def score(self, X, Y):
        total = np.shape(X)[0]
        correct = 0
        delayed_results = [delayed(self.predict)(X.iloc[j]) for j in range(0, total)]
        predicitons_par = compute(*delayed_results)

        for (pred, real) in zip(predicitons_par, Y):
            if pred == real:
                correct += 1
        return correct / total
