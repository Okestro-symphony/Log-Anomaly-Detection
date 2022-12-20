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

    @Log_Anomaly_Model.timed
    def run(self):
        start_time = datetime.now(timezone.utc)
        result = 'success'
        error = '없습니다'
        host_list = super().retrieve_host_list()
        for host in host_list:
            path_model = self.model_path + '/log_anomaly_model/{hostname}_iforest.pkl'.format(hostname=host)
            path_vect = self.model_path + '/log_anomaly_model/{hostname}_CountVect.pkl'.format(hostname=host)
            if os.path.isfile(path_model):
                model = joblib.load(path_model)
                vect = joblib.load(path_vect)
                df_log = super().load_host_data(host, self.gte, self.now)
                if len(df_log) < self.min_log_counts:
                    standardLog.sending_log('warning').warning(f'AnoamlyDetection logs from {host} {len(df_log)} are less than minimum log counts ({self.min_log_counts})')
                    continue
                df_log_preprocessed, vect = super().preprocess_host(df_log, vect)

                res_df = self.retrieve_anomaly_score(model, df_log_preprocessed)

                df_log['anomaly_score'] = res_df

                anomaly_count = len(df_log[df_log['anomaly_score'] < 0])
                log_count = len(df_log)
                anomaly_pct = anomaly_count/log_count*100

                df_log = df_log[df_log['anomaly_score'] < 0]
                df_log['anomaly_score'] = df_log['anomaly_score'] * 2 + 1
                df_log['timestamp'] = pd.to_datetime(df_log['@timestamp']).astype(int)//10**6
                df_log['datetime'] = df_log['@timestamp']
                standardLog.sending_log('warning').warning(f'AnoamlyDetection logs from {anomaly_count}({anomaly_pct}%) anomaly logs detected in {log_count} total logs in {host}')
                super().save_result_es(df_log)
            else:
                standardLog.sending_log('warning').warning(f'AnoamlyDetection model not found for {host}')
                continue
        self.report_result(start_time, result, error)
        return None

if __name__ == '__main__':
    inference = Log_Anomaly_Inference()
    inference.run()