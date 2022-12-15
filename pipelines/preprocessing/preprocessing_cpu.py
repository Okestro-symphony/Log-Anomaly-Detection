import datetime
import os
from configparser import ConfigParser
from elasticsearch import Elasticsearch
import pandas as pd


def preprocessing_main():
    # config parser
    config = ConfigParser()
    abs_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config.read(abs_path + '/config.ini')
    # data path
    data_dir = config.get('Preprocessing', 'DATA_DIR')
    # es config
    es_host=config.get('ES','HOST')
    es_port=config.get('ES','PORT')
    es_id=config.get('ES','USER')
    es_pw=config.get('ES','PASSWORD')
    now=datetime.datetime.now()

    es = Elasticsearch(hosts=f"http://{es_id}:{es_pw}@{es_host}:{es_port}/", timeout=1000)