FROM python:3.7
WORKDIR /root

# Installs additional packages
RUN pip install appengine-python-standard gcsfs numpy pandas google-cloud-aiplatform==1.18.2 kfp==1.8.14 google-cloud-pipeline-components==1.0.26 tensorflow==2.8.3 --upgrade

RUN mkdir /app
RUN mkdir /app/pipeline

COPY pipeline/__init__.py /app/pipeline
COPY pipeline/kfp_pipeline.py /app/pipeline
# COPY pipeline/lib/batch_serve.py /app/pipeline/lib/batch_serve.py
# COPY pipeline/lib/__init__.py /app/pipeline/lib
# COPY pipeline/lib/evaluate_model.py /app/pipeline/lib/evaluate_model.py
COPY pipeline_compile.py /app
COPY pipeline_run.py /app
WORKDIR /app
