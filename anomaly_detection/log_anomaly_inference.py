import os
import sys
from datetime import datetime, timedelta, timezone
import pandas as pd
import joblib
from elasticsearch import ElasticsearchDeprecationWarning
import warnings
warnings.filterwarnings(action='ignore', category=ElasticsearchDeprecationWarning)
from log_anomaly_train import Log_Anomaly_Model

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from utils.logs.log import standardLog
standardLog = standardLog()

class Log_Anomaly_Inference(Log_Anomaly_Model):
    def __init__(self,config_path ,save_path):
        super().__init__(config_path ,save_path)
        self.log_anomaly = Log_Anomaly_Model(config_path, save_path)
        self.project_path = os.path.dirname(__file__)
        self.model_path = save_path

        self.now = datetime.now(timezone.utc)
        self.gte_hour = 1
        self.gte = self.now - timedelta(hours=self.gte_hour)

        self.min_log_counts = 200

    @Log_Anomaly_Model.timed
    def retrieve_anomaly_score(self, model, df):
        res_df = model.decision_function(df)
        return res_df

    @Log_Anomaly_Model.timed
    def report_result(self, start_time, result, error, end_time=None):

        end_time = datetime.now(timezone.utc)
        res = {
            "job": "Inference Log Anomaly",
            "start_time": start_time,
            "end_time": end_time,
            "result": result,
            "error": error
        }
        self.es.index(index="symphony_job_schedule", doc_type="_doc", body=res)
        return None