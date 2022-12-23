import os
import pickle
from sklearn.base import BaseEstimator
import numpy as np
from collections import defaultdict
from sklearn.base import BaseEstimator


class FeatureExtractor(BaseEstimator):
    def __init__(self):
        self.meta_data = {}
    
    def fit(self, session_dict):
        log_padding = "<pad>"
        log_oov = "<oov>"
        total_logs = session_dict['original']
        self.ulog_train = set(total_logs)
        self.id2log_train = {0: log_padding, 1: log_oov}
        self.id2log_train.update(
            {idx: log for idx, log in enumerate(self.ulog_train, 2)}
        )
        self.log2id_train = {v: k for k, v in self.id2log_train.items()}
        self.meta_data["num_labels"] = len(self.log2id_train)
        self.meta_data["vocab_size"] = len(self.log2id_train)

    def __windows2sequential(self, windows):
        total_features = []
        for window in windows.values:
            ids = [self.log2id_train.get(x, 1) for x in window]
            total_features.append(ids)
        return np.array(total_features)
    
    def transform(self, session_dict):
        feature_dict = defaultdict(list)
        windows = session_dict["templates"]
        feature_dict["sequentials"] = self.__windows2sequential(windows)
        session_dict["features"] = feature_dict
        return session_dict
    
    def fit_transform(self, session_dict):
        self.fit(session_dict)
        return self.transform(session_dict)

    def save_ext(self, path):
        with open(file=os.path.join(path, "estimator.pickle"), mode='wb') as f:
            pickle.dump(self, f)
    
    def load_ext(self, path):
        with open(file=os.path.join(path, "estimator.pickle"), mode='rb') as f:
            ext = pickle.load(f)
        return ext