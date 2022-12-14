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