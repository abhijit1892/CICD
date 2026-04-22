import os
import boto3
import json
from pipeline import get_pipeline

def main():
    region = os.environ.get("AWS_REGION", "us-east-1")
    account = boto3.client("sts").get_caller_identity()["Account"]
    
    # Define bucket and role that the CDK stack created
    # Based on the stack naming scheme we used
    default_bucket = f"mlops-detectron2-data-{account}-{region}"
    
    # We need a Role ARN to pass to the SageMaker pipeline.
    # In a real environment, you might query this via CloudFormation outputs.
    # Here we assume the Stack created the role or we use a fallback Default SageMaker role.
    iam_client = boto3.client('iam')
    try:
        rules = iam_client.list_roles()
        # Find the role CDK created or default SM role
        # we will use AWS standard fallback if we can't find it
    except Exception as e:
        pass

    try:
        role_arn = sagemaker.get_execution_role()
    except Exception:
        # Fallback to constructing the ARN if executing in GitHub Actions without direct sagemaker execution context
        # You'd normally output the Role ARN in CDK Stack and inject as Environment Variable.
        # Here we do a programmatic guess or require it in ENV.
        role_arn = f"arn:aws:iam::{account}:role/Detectron2MLOpsPipelineSta-SageMakerExecutionRole"
    
    print(f"Deploying SageMaker Pipeline to {region}")
    pipeline = get_pipeline(
        region=region,
        role_arn=role_arn,
        default_bucket=default_bucket
    )
    
    print(pipeline.definition())
    
    # Upsert the pipeline definition
    parsed = json.loads(pipeline.definition())
    response = pipeline.upsert(role_arn=role_arn)
    print(f"Pipeline Upsert Response: {response}")

if __name__ == "__main__":
    main()
