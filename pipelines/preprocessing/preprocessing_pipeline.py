import kfp
from kfp import dsl
from kfp import components
from kfp.components import func_to_container_op
from elasticsearch import Elasticsearch
import kubernetes.client
client = kfp.Client(host='http://34.64.153.235:30020')



def preprocessing(metric : str):
    import sys
    sys.path.append('/symphony/croffle/pipelines/preprocessing')

    if metric == "filesystem":
        # preprocessing filesystem
        from preprocessing_filesystem import preprocessing_main
        preprocessing_main() 
    elif metric == "cpu":
        # prerpocessing cpu
        from preprocessing_cpu import preprocessing_main
        preprocessing_main() 
    elif metric == "disk":
        # preprocessing disk
        from preprocessing_disk import preprocessing_main
        preprocessing_main() 
    elif metric == "memory":
        # preprocessing memory
        from preprocessing_memory import preprocessing_main
        preprocessing_main() 
    elif metric == "network":
        # preprocessing network
        from preprocessing_network import preprocessing_main
        preprocessing_main()  
    elif metric == "pod":
        # preprocessing pod
        from preprocessing_pod import preprocessing_main
        preprocessing_main()   

     

# pipelines components
preprocessing_cpu_component = components.create_component_from_func(
        func=preprocessing,                 
        base_image='okestroaiops/prediction:latest'
    )