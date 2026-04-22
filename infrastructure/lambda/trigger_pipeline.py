import os
import boto3
import json
import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sm_client = boto3.client('sagemaker')

def handler(event, context):
    logger.info(f"Received event: {json.dumps(event)}")
    
    # Extract S3 bucket and object key from the EventBridge event
    detail = event.get('detail', {})
    bucket = detail.get('bucket', {}).get('name')
    key = detail.get('object', {}).get('key')
    
    logger.info(f"Triggered by S3 upload: s3://{bucket}/{key}")
    
    pipeline_name = os.environ.get('PIPELINE_NAME', 'Detectron2-TrainingPipeline')
    
    try:
        # Start SageMaker Pipeline Execution
        response = sm_client.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineExecutionDisplayName=f"TriggeredBy-{key.replace('/', '-')}"[:82],
            PipelineParameters=[
                {
                    'Name': 'InputDataS3Uri',
                    'Value': f"s3://{bucket}/{key}"
                }
            ]
        )
        
        logger.info(f"Started pipeline execution: {response['PipelineExecutionArn']}")
        return {
            'statusCode': 200,
            'body': json.dumps({'PipelineExecutionArn': response['PipelineExecutionArn']})
        }
    except Exception as e:
        logger.error(f"Error starting pipeline: {str(e)}")
        raise e
