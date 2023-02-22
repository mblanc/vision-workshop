"""Cloud Function to be triggered by Pub/Sub."""
import os
import json
import logging
from google.cloud import storage
import base64
from datetime import datetime
from google.cloud import aiplatform as vertex_ai


if 'GCP_PROJECT' in os.environ:
    project_id = os.environ['GCP_PROJECT']
else:
    raise Exception('Failed to determine project_id')
        
BUCKET_NAME = f"{project_id}-vision-workshop"

logging.info(BUCKET_NAME)

client = storage.Client()
bucket =  client.get_bucket(BUCKET_NAME)
blob = bucket.get_blob('config/notebook_env.py')
config = blob.download_as_string()
exec(config)

PIPELINE_ROOT = f'gs://{BUCKET_NAME}/pipelines'
PIPELINE_NAME = f'vision-workshop-tf-pipeline-{ID}'

def trigger_pipeline(event, context):    
    data = base64.b64decode(event["data"]).decode("utf-8")
    logging.info(f"Event data: {data}")

    data = json.loads(data)
    
    vertex_ai.init(project=PROJECT, staging_bucket=BUCKET_NAME, location=REGION)
    
    logging.info(f"pipeline_spec_uri: {data['pipeline_spec_uri']}")

    job = vertex_ai.PipelineJob(display_name = PIPELINE_NAME,
                                 template_path = data['pipeline_spec_uri'],
                                 pipeline_root = PIPELINE_ROOT,
                                 parameter_values = data['parameter_values'],
                                 project = PROJECT,
                                 location = REGION)
    
    response = job.run(sync=False)

    logging.info(response)
