import datetime
import os
from configparser import ConfigParser
from elasticsearch import Elasticsearch
import pandas as pd


def preprocessing_main():
    config = ConfigParser()
    abs_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config.read(abs_path + '/config.ini')
    data_dir = config.get('Preprocessing', 'DATA_DIR')
    es_host=config.get('ES','HOST')
    es_port=config.get('ES','PORT')
    es_id=config.get('ES','USER')
    es_pw=config.get('ES','PASSWORD')
    now=datetime.datetime.now()

    es = Elasticsearch(hosts=f"http://{es_id}:{es_pw}@{es_host}:{es_port}/", timeout=1000)
    def get_unique_hosts(es):
        body={
          "query": {
            "bool":{
          "must":[
        {"range": {
          "datetime": {
            "gte": "now-3M",
            "lte": "now"
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
        data = es.search(index= 'sym-metric-memory*',body = body, track_total_hits= True, size=0)
        hosts = [ hash_map['key'] for hash_map in data['aggregations']['keys']['buckets']]
        return hosts
    hosts=get_unique_hosts(es)
    host_dataframe_list = list()
    for host in hosts:
        body={
          "query": {
              "bool": {
                  "must": [
                      {"range": {
                          "datetime": {
                              "gte": "now-3M",
                              "lte": "now"
                          }
                      }},
                      {"term": {
                          "host.hostname.keyword": host
                      }}]
              }
          },
          "aggs": {
            "tmp_result" :{
              "date_histogram": {
                "field": "datetime",
                "fixed_interval": "5m"
              },
                  "aggs": {
                    "mean_memory_usage": {
                      "avg": {
                        "field": "system.memory.actual.used.pct"
                  }
                },
                      "mean_swap": {
                          "avg": {
                              "field": "system.memory.swap.used.pct"
                        }
                    }
              }
            }
          },"size": 0
        }
        data = es.search(index='sym-metric-memory*', body=body, track_total_hits=True, size=0)
        data = pd.DataFrame(data['aggregations']['tmp_result']['buckets'])
        data['host_name'] = host
        data['mean_memory_usage'] = data['mean_memory_usage'].apply(lambda x: x['value'])
        data['mean_swap'] = data['mean_swap'].apply(lambda x: x['value'])
        if data.shape[0]<30:
            continue
        data = data.drop(['key','doc_count'],axis=1)
        data = data.rename(columns={'key_as_string': 'datetime'})
        data = data.fillna(data.mean())
        host_dataframe_list.append(data)
    final_df = pd.concat(host_dataframe_list, ignore_index= True)
    final_df.to_csv(data_dir+"/memory.csv", index=False)



if __name__ == '__main__':
    preprocessing_main()