import kfp
from kfp import dsl
from kfp import components
from kfp.components import func_to_container_op
from elasticsearch import Elasticsearch
import kubernetes.client
client = kfp.Client(host='ip address')



def preprocessing(metric : str):
    import sys
    sys.path.append('/symphony/croffle/pipelines/preprocessing')

    if metric == "filesystem":
        from preprocessing_filesystem import preprocessing_main
        preprocessing_main() 
    elif metric == "cpu":
        from preprocessing_cpu import preprocessing_main
        preprocessing_main() 
    elif metric == "disk":
        from preprocessing_disk import preprocessing_main
        preprocessing_main() 
    elif metric == "memory":
        from preprocessing_memory import preprocessing_main
        preprocessing_main() 
    elif metric == "network":
        from preprocessing_network import preprocessing_main
        preprocessing_main()  
    elif metric == "pod":
        from preprocessing_pod import preprocessing_main
        preprocessing_main()   


preprocessing_cpu_component = components.create_component_from_func(
        func=preprocessing,                 
        base_image='okestroaiops/prediction:latest'
    )



@dsl.pipeline(
    name="croffle-preprocessing_pvc",
    description = "croffle preprocessing pipeline pvc"
)
def preprocessing_pipeline(cpu_request :str="4000m", 
                        cpu_limit : str="8000m",
                        memory_request : str="4000Mi",
                        memory_limit : str="16000Mi"):
    dsl.get_pipeline_conf().set_image_pull_secrets([kubernetes.client.V1LocalObjectReference(name="okestroaiops")])
    vop = dsl.PipelineVolume(pvc='croffle-pvc')    

    mount_path = "/symphony/"

    preprocessing_task_cpu = preprocessing_cpu_component("cpu").set_cpu_limit(cpu_limit)\
                                            .set_memory_limit(memory_limit)\
                                            .set_cpu_request(cpu_request)\
                                            .set_memory_request(memory_request)\
                                            .add_pvolumes({mount_path: vop})
    preprocessing_task_memory = preprocessing_cpu_component("memory").set_cpu_limit(cpu_limit)\
                                            .set_memory_limit(memory_limit)\
                                            .set_cpu_request(cpu_request)\
                                            .set_memory_request(memory_request)\
                                            .add_pvolumes({mount_path: vop})\
                                            .after(preprocessing_task_cpu)
    preprocessing_task_disk = preprocessing_cpu_component("disk").set_cpu_limit(cpu_limit)\
                                            .set_memory_limit(memory_limit)\
                                            .set_cpu_request(cpu_request)\
                                            .set_memory_request(memory_request)\
                                            .add_pvolumes({mount_path: vop})\
                                            .after(preprocessing_task_memory)
    preprocessing_task_filesystem = preprocessing_cpu_component("filesystem").set_cpu_limit(cpu_limit)\
                                            .set_memory_limit(memory_limit)\
                                            .set_cpu_request(cpu_request)\
                                            .set_memory_request(memory_request)\
                                            .add_pvolumes({mount_path: vop})\
                                            .after(preprocessing_task_disk)
    preprocessing_task_network = preprocessing_cpu_component("network").set_cpu_limit(cpu_limit)\
                                            .set_memory_limit(memory_limit)\
                                            .set_cpu_request(cpu_request)\
                                            .set_memory_request(memory_request)\
                                            .add_pvolumes({mount_path: vop})\
                                            .after(preprocessing_task_filesystem)
    preprocessing_task_pod = preprocessing_cpu_component("pod").set_cpu_limit(cpu_limit)\
                                            .set_memory_limit(memory_limit)\
                                            .set_cpu_request(cpu_request)\
                                            .set_memory_request(memory_request)\
                                            .add_pvolumes({mount_path: vop})\
                                            .after(preprocessing_task_network)

    dsl.get_pipeline_conf().set_ttl_seconds_after_finished(20)


client.create_run_from_pipeline_func(preprocessing_pipeline, arguments={})



kfp.compiler.Compiler().compile(
    pipeline_func=preprocessing_pipeline,
    package_path='preprocessing_pl.yaml'
)

client.create_recurring_run(
    experiment_id = client.get_experiment(experiment_name="Default").id,
    job_name="preprocessing_croffle_pvc",
    description="version: croffle:preprocessing_pvc",
    cron_expression="0 0 16 * * *",
    pipeline_package_path = "preprocessing_pl.yaml",
)



client.upload_pipeline(
    pipeline_package_path='preprocessing_pl.yaml',
    pipeline_name = "preprocessing_croffle_pvc",
    description = "version: croffle:preprocessing"
)