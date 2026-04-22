import os
import sys
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

# Ensure AWS Context
try:
    sts = boto3.client('sts')
    identity = sts.get_caller_identity()
    print(f"Logged into AWS as Account: {identity['Account']}")
except Exception as e:
    print("FATAL: Cannot connect to AWS. Did you configure your AWS credentials?")
    print("Run `aws configure` or set env variables AWS_ACCESS_KEY_ID & AWS_SECRET_ACCESS_KEY")
    sys.exit(1)

# Use the specific bucket provided
default_bucket = "try-demo-bucket12"
sagemaker_session = sagemaker.Session(default_bucket=default_bucket)
region = sagemaker_session.boto_region_name

print(f"Using Specific S3 Bucket: {default_bucket} in Region: {region}")

# Ensure IAM Role exists
try:
    role = sagemaker.get_execution_role()
except ValueError:
    print("\nSageMaker Execution Role not found locally.")
    print("Because we skipped CDK deploy, we must specify an IAM Role ARN that has AmazonSageMakerFullAccess.")
    print("Please export the environment variable: export SAGEMAKER_ROLE='arn:aws:iam::your-account:role/your-role'")
    role = os.environ.get("SAGEMAKER_ROLE")
    if not role:
        sys.exit(1)

print(f"Using SageMaker IAM Role: {role}")

# Define the Estimator
estimator = PyTorch(
    entry_point="train.py",
    source_dir="pipeline/scripts",
    role=role,
    framework_version="2.0.0",
    py_version="py310",
    instance_count=1,
    instance_type="ml.g4dn.xlarge", 
    volume_size=60,
    sagemaker_session=sagemaker_session,
    base_job_name="Detectron2-TrainingRun",
    output_path=f"s3://{default_bucket}/detectron2-output/"
)

# Specify your local dataset path.
LOCAL_DATASET_PATH = "/home/abhijit/Downloads/dataset"

print(f"Uploading local dataset {LOCAL_DATASET_PATH} to S3 bucket {default_bucket}...")
# Upload data to the explicitly requested bucket under 'detectron2-dataset' prefix
s3_data_uri = sagemaker_session.upload_data(
    path=LOCAL_DATASET_PATH, 
    bucket=default_bucket, 
    key_prefix="detectron2-dataset"
)

print(f"Dataset successfully uploaded to: {s3_data_uri}")
print("Submitting Training Job to AWS...")
print("This will provision an ml.g4dn.xlarge instance and start training.")

try:
    # Execute Training Job
    estimator.fit({"train": s3_data_uri})
    
    print("\n\nTraining Job Completed successfully!")
    print(f"Find your trained model weights saved here: {estimator.model_data}")
except Exception as e:
    print(f"\nTraining Failed: {str(e)}")
    print("Note: If you got a 'ResourceLimitExceeded' error, you must request a quota increase for 'ml.g4dn.xlarge for training job usage'.")

