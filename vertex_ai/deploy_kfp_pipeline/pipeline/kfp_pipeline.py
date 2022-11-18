# General
import os
import sys
import random
from datetime import datetime, timedelta
import json

# # Vertex Pipelines
# from typing import NamedTuple
# import kfp
# from kfp.v2 import dsl
# from kfp.v2.dsl import Artifact, Dataset, Input, InputPath, Model, Output, OutputPath, Metrics, ClassificationMetrics, Condition, component
# from kfp.v2 import compiler

from google.cloud import aiplatform as vertex_ai
from google_cloud_pipeline_components import aiplatform as vertex_ai_components
from google_cloud_pipeline_components.v1.batch_predict_job import ModelBatchPredictOp
from google.cloud import storage
# from kfp.v2.google.client import AIPlatformClient as VertexAIClient


# import google.cloud.aiplatform as aip
# from google_cloud_pipeline_components.experimental.custom_job import utils
# from kfp.v2 import compiler, dsl
# from kfp.v2.dsl import component

import google.cloud.aiplatform as vertex_ai
import tensorflow as tf
import kfp
from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.dsl import component

# These variables would be passed from Cloud Build in CI/CD.
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")
PROJECT_ID = os.getenv("PROJECT_ID", "")
BUCKET_NAME = f"{PROJECT_ID}-vision-workshop"

client = storage.Client()
bucket =  client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob('config/notebook_env.py')
config = blob.download_as_string()
exec(config)



# TODO to load it from config file
IMAGE_REPOSITORY = f"vision-{ID}"
IMAGE_NAME="image-classifier"
IMAGE_TAG="v1"
IMAGE_URI=f"europe-west4-docker.pkg.dev/{PROJECT_ID}/{IMAGE_REPOSITORY}/{IMAGE_NAME}:{IMAGE_TAG}"

#Pipeline
# #Pipeline
PIPELINE_NAME = f'vision-workshop-tf-pipeline-{ID}'
PIPELINE_DIR=os.path.join(os.curdir, 'pipelines')
PIPELINE_ROOT = f"gs://{BUCKET_NAME}/pipelines"
PIPELINE_PACKAGE_PATH = f"{PIPELINE_DIR}/pipeline_{ID}.json"
COMPONENTS_DIR=os.path.join(os.curdir, 'pipelines', 'components')

#Feature Store component
START_DATE_TRAIN = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
END_DATE_TRAIN = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

#Dataset component
DATASET_NAME = f'vision_workshop_dataset_{END_DATE_TRAIN}'

#Training component
JOB_NAME = f'image-classifier-train-tf-{ID}'
MODEL_NAME = f'image-classifier-tf-{ID}'
TRAIN_MACHINE_TYPE = 'n1-standard-4'
# CONTAINER_URI = 'us-docker.pkg.dev/vertex-ai/training/xgboost-cpu.1-1:latest'
MODEL_SERVING_IMAGE_URI = "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-8:latest"
ARGS=[ "--lr=0.003", "--epochs=5"]

#endpoint
ENDPOINT_NAME = 'vision_workshop_tf_prediction'


@dsl.pipeline(
    pipeline_root=PIPELINE_ROOT,
    name=PIPELINE_NAME,
)
def pipeline(project_id:str = PROJECT_ID, 
             region:str = REGION, 
             bucket_name:str = f"gs://{BUCKET_NAME}",
             replica_count:int = 1,
             machine_type:str = "n1-standard-4",
            ):
    
    #create dataset 
    dataset_create_op = vertex_ai_components.ImageDatasetCreateOp(project=project_id,
                                                       location=region,
                                                       display_name=DATASET_NAME,
                                                       import_schema_uri=vertex_ai.schema.dataset.ioformat.image.single_label_classification,
                                                       gcs_source=f"gs://{BUCKET_NAME}/prod/flowers.csv")
    
    #custom training job component - script
    train_model_op = vertex_ai_components.CustomContainerTrainingJobRunOp(
        display_name=JOB_NAME,
        model_display_name=MODEL_NAME,
        container_uri=IMAGE_URI,
        staging_bucket=bucket_name,
        dataset=dataset_create_op.outputs['dataset'],
        annotation_schema_uri=vertex_ai.schema.dataset.annotation.image.classification,
        base_output_dir=bucket_name,
        args = ARGS,
        replica_count= replica_count,
        machine_type= machine_type,
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        model_serving_container_image_uri=MODEL_SERVING_IMAGE_URI,
        project=project_id,
        location=region).after(dataset_create_op)
    
    batch_op = ModelBatchPredictOp(
        project=project_id,
        location=region,
        job_display_name="batch_predict_job",
        model=train_model_op.outputs["model"],
        gcs_source_uris=[f"gs://{BUCKET_NAME}/test2.jsonl"],
        gcs_destination_output_uri_prefix=f"gs://{BUCKET_NAME}",
        instances_format="jsonl",
        predictions_format="jsonl",
        model_parameters={},
        machine_type=machine_type,
        starting_replica_count=1,
        max_replica_count=1,
    )

    
    #create endpoint
    create_endpoint_op = vertex_ai_components.EndpointCreateOp(
        display_name=ENDPOINT_NAME,
        project=project_id, 
        location=region).after(train_model_op)

    #deploy th model
    custom_model_deploy_op = vertex_ai_components.ModelDeployOp(
        model=train_model_op.outputs["model"],
        endpoint=create_endpoint_op.outputs["endpoint"],
        deployed_model_display_name=MODEL_NAME,
        dedicated_resources_machine_type=machine_type,
        dedicated_resources_min_replica_count=replica_count
    ).after(create_endpoint_op)
