import os
import sys
import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

def main():
    print("="*50)
    print("GitHub Actions CI/CD SageMaker Trigger")
    print("="*50)

    try:
        sts = boto3.client('sts')
        identity = sts.get_caller_identity()
        print(f"Logged into AWS as Account: {identity['Account']}")
    except Exception as e:
        print("FATAL: AWS Credentials not found. Did you set them in GitHub Secrets?")
        sys.exit(1)

    sagemaker_session = sagemaker.Session()
    
    # Use the established bucket and region
    default_bucket = "try-demo-bucket12"
    region = "us-east-1"
    print(f"Environment: Bucket={default_bucket}, Region={region}")

    # IAM Role Resolution
    role = os.environ.get("SAGEMAKER_ROLE")
    if not role:
        print("FATAL: SAGEMAKER_ROLE was not found in environment variables.")
        sys.exit(1)

    # Note: We do NOT upload anything locally here. 
    # CI/CD has NO access to the original laptop dataset. 
    # We instead point the estimator directly to the pre-existing S3 data path.
    s3_data_uri = f"s3://{default_bucket}/detectron2-dataset"
    print(f"Binding Training Job to strict S3 Data Target: {s3_data_uri}")

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
        base_job_name="Detectron2-ActionsRun",
        output_path=f"s3://{default_bucket}/detectron2-output/"
    )

    print("Deploying Execution Command...")
    try:
        # Wait=False causes the GitHub Actions to immediately complete with Success after triggering, 
        # so GitHub doesn't burn CI/CD minutes sitting around for hours waiting for Detectron2 to finish.
        estimator.fit({"train": s3_data_uri}, wait=False)
        print("✅ Command Delivered Successfully! Job is now running asynchronously on Amazon.")
    except Exception as e:
        print(f"AWS Execution Rejected: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
