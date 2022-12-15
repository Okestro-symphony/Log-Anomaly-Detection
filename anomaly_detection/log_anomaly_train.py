from datetime import datetime, timedelta, timezone
import time
import csv
import requests
import os
from functools import wraps
import sys
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.ensemble import IsolationForest
import joblib
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan, bulk
from elasticsearch import ElasticsearchDeprecationWarning
import configparser

import warnings
warnings.filterwarnings(action='ignore', category=ElasticsearchDeprecationWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))))
from utils.logs.log import standardLog
standardLog = standardLog()

class Log_Anomaly_Model():
    def __init__(self, config_path, save_path):
        self.project_path = save_path
        self.save_path = save_path
        # read config
        self.config = configparser.ConfigParser()
        self.config.read(config_path)
        # db info
        self.es = Elasticsearch(hosts=self.config['ES']['HOST'], port=int(self.config['ES']['PORT']), http_auth=(
            self.config['ES']['USER'], self.config['ES']['PASSWORD']
        ))
        self.api_url = 'https://{}/api/v1'.format(self.config['META']['DOMAIN'])
        self.api_key = {'X-Api-key' : self.config['META']['API_KEY']}

        # meta
        self.gte_date = 1
        self.now = datetime.now(timezone.utc)
        self.gte_date = datetime.date(self.now) - timedelta(days=self.gte_date)
        self.gte_date = datetime.combine(self.gte_date, datetime.min.time())

        # parameter
        self.min_log_counts = 2000
        self.max_features = 500
        self.contamination = 0.01
        self.max_log_counts = 1000000

    def timed(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now(timezone.utc)
            result = func(*args, **kwargs)
            end = datetime.now(timezone.utc)
            return result
        return wrapper


    def response(self, conn, index, query=None):
        if query is None:
            query = {"query": {"match_all": {}}}

        return scan(conn, scroll='10m', index=index, doc_type='_doc', size=10000, query=query)

    def retrieve_meta_data(self):
        """
        api 활용 vm metadata 불러오기
        return : vm_metadata(dataframe)
        """
        meta_providers = requests.get(self.api_url + '/meta/providers', headers = self.api_key,
                                      verify = False).json()['resResult']
        provider_list = list(set([t['type'] for t in meta_providers]))
        df = pd.DataFrame()
        for provider in meta_providers:
            if provider['type'] == 'openstack':
                prvd_id = provider['id']
                res = requests.get(self.api_url + '/meta/vms', params = {'prvdId' : prvd_id}, headers = self.api_key,
                                   verify = False)
                try:
                    df = df.append(pd.DataFrame(res.json()['resResult']))
                except Exception as e:
                    standardLog.sending_log('error', e).error(f'AnomalyDetection error occurs during loading meta data from {prvd_id}')
        return df.reset_index(drop = True)

    @timed
    def retrieve_host_list(self):

        body={
          "query": {
            "bool":{
          "must":[
        {"range": {
          "@timestamp": {
            "gte": self.gte_date.isoformat(sep="T")
          }
        }}]
       }
          },
          "aggs":{
            "keys":{
              "terms":{
                "field":"host.hostname.keyword",
                "size":1000
              }
            }
          },
          "size": 0,
         "track_total_hits": "true"
        }
        data = self.es.search(index= 'sym-log-syslog*',body = body, track_total_hits= True, size=0)
        host_list = [hash_map['key'] for hash_map in data['aggregations']['keys']['buckets']]
        return host_list

    @timed
    def load_host_data(self, hostname, gte=None, lte=None):
        if gte is None:
            gte = self.gte_date
        if lte is None:
            lte = self.now
        os.makedirs('{project_path}/host_log'.format(project_path=self.project_path), exist_ok=True)
        program_filter_list = ["logstash"]

        index = "sym-log-syslog"
        body = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "range": {
                                "@timestamp": {
                                    "gte": gte.isoformat(sep="T"),
                                    "lte": lte.isoformat(sep="T")
                                }
                            }
                        },
                            {'terms' : {
                                'host.hostname.keyword' : [hostname]
                            }}
                    ],
                    "must_not": [
                        {
                            "terms": {
                                "program.keyword": program_filter_list
                            }
                        }
                    ]
                }
            }
        }
        res = self.response(self.es, index, body)
        res_df = pd.DataFrame()

        syslog_fields = ["@timestamp", "@version", "agent.type",
                         "host.hostname", "host.id",
                         "input.type", "log.file.path", "log.offset", "message", "pid", "program"]

        count = 0
        now = time.time()
        start = now
        with open('{project_path}/host_log/{hostname}.csv'.format(hostname = hostname,
                                                                  project_path=self.project_path), 'w') as file:
            writer = csv.DictWriter(file, fieldnames=syslog_fields)
            header = writer.writeheader()
            for row in res:
                result_dict = {}
                dict_list = []
                org_row = row['_source']
                for item in syslog_fields:
                    flag = True
                    row = org_row
                    for i in item.split("."):
                        try:
                            row = row[i]
                        except KeyError:
                            flag = False
                    if flag is True:
                        result_dict[item] = row
                    else:
                        pass
                # df_dict = pd.DataFrame([result_dict])
                # res_df = pd.concat([res_df, df_dict], ignore_index=True)
                writer.writerow(result_dict)
                count +=1
                if count%100000 == 0:
                    standardLog.sending_log('warning').warning(\
                        f'AnoamlyDetection elapsed time for 100000 rows : {time.time() - now}')
                    now = time.time()
                if count%1000000 == 0:
                    standardLog.sending_log('warning').warning(\
                        f'AnoamlyDetection host has more than maximum log counts model will be trained with {self.max_log_counts} logs')
                    break
            standardLog.sending_log('success').info(f'AnoamlyDetection total {count} rows loaded during {time.time() - start} secs')
        res_df = pd.read_csv('{project_path}/host_log/{hostname}.csv'.format(project_path=self.project_path,
                                                                             hostname = hostname))
        res_df.fillna(0, inplace=True)
        return res_df