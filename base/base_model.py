import os
import time
import logging
from collections import defaultdict 
import torch
import numpy as np
import pandas as pd
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from utils.utils import set_device
from utils.utils import tensor2flatten_arr
from base.base_embedder import Embedder



class ForcastBasedModel(nn.Module):
    def __init__(
        self,
        meta_data,
        model_save_path,
        embedding_dim,
        gpu=-1,
        anomaly_ratio=None,
        patience=3,
        **kwargs,
    ):
        super(ForcastBasedModel, self).__init__()
        self.device = set_device(gpu)
        self.meta_data = meta_data
        self.anomaly_ratio = float(anomaly_ratio)
        self.patience = int(patience)
        self.time_tracker = {}
        self.model_save_file = os.path.join(model_save_path, "model.ckpt")
        self.embedder = Embedder(
            meta_data["vocab_size"],
            embedding_dim=embedding_dim
        )

    def evaluate(self, test_loader, dtype="test"):
        return self.__evaluate_recst(test_loader, dtype=dtype)

    def inference(self, test_loader):
        self.eval() 
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = self.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]
                store_dict["preds"].extend(tensor2flatten_arr(y_pred))
            infer_end = time.time()
            logging.info("Finish inference [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)

            use_cols = ["preds"]
            session_df = (
                store_df[use_cols]
            )
            assert (self.anomaly_ratio is not None), "anomaly_ratio should be specified for autoencoder!"
            thre = np.percentile(
                session_df[f"preds"].values, 100 - self.anomaly_ratio * 100
            )
            session_df['labels'] = (session_df[f"preds"] > thre).astype(int)

            return session_df[["preds","labels"]]

        
    def test(self, test_loader):
        self.eval() 
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = self.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]
                store_dict["label"].extend(
                    tensor2flatten_arr(batch_input["label"])
                )
                store_dict["preds"].extend(tensor2flatten_arr(y_pred))
            infer_end = time.time()
            logging.info("Finish inference [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)

            use_cols = ["label", "preds"]
            session_df = (
                store_df[use_cols]
            )
            assert (self.anomaly_ratio is not None), "anomaly_ratio should be specified for autoencoder!"
            thre = np.percentile(
                session_df[f"preds"].values, 100 - self.anomaly_ratio * 100
            )
            pred = (session_df[f"preds"] > thre).astype(int)
        
            return pred

    def __evaluate_recst(self, test_loader):
        self.eval()  # set to evaluation mode
        with torch.no_grad():
            y_pred = []
            store_dict = defaultdict(list)
            infer_start = time.time()
            for batch_input in test_loader:
                return_dict = self.forward(self.__input2device(batch_input))
                y_pred = return_dict["y_pred"]
                store_dict["label"].extend(
                    tensor2flatten_arr(batch_input["label"])
                )
                store_dict["preds"].extend(tensor2flatten_arr(y_pred))
            infer_end = time.time()
            logging.info("Finish inference [{:.2f}s]".format(infer_end - infer_start))
            self.time_tracker["test"] = infer_end - infer_start

            store_df = pd.DataFrame(store_dict)

            use_cols = ["label", "preds"]
            session_df = (
                store_df[use_cols]
            )
            assert (
                self.anomaly_ratio is not None
            ), "anomaly_ratio should be specified for autoencoder!"
            thre = np.percentile(
                session_df[f"preds"].values, 100 - self.anomaly_ratio * 100
            )
            pred = (session_df[f"preds"] > thre).astype(int)
            y = (session_df["label"] > 0).astype(int)

            eval_results = {
                "f1": f1_score(y, pred),
                "rc": recall_score(y, pred),
                "pc": precision_score(y, pred),
                "acc": accuracy_score(y, pred),
            }
            logging.info({k: f"{v:.3f}" for k, v in eval_results.items()})
            return eval_results