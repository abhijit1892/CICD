from aws_cdk import (
    Stack,
    RemovalPolicy,
    aws_s3 as s3,
    aws_s3_notifications as s3n,
    aws_lambda as _lambda,
    aws_events as events,
    aws_events_targets as targets,
    aws_iam as iam,
    aws_sns as sns,
)
from constructs import Construct

class MLOpsPipelineStack(Stack):

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # 1. S3 Bucket for Data and Model Artifacts
        self.data_bucket = s3.Bucket(
            self, "DataBucket",
            bucket_name=f"mlops-detectron2-data-{self.account}-{self.region}",
            versioned=True,
            removal_policy=RemovalPolicy.DESTROY, # For dev purposes only
            auto_delete_objects=True,
            event_bridge_enabled=True # Enable EventBridge for S3 events
        )

        # 2. SNS Topic for Notifications
        self.notification_topic = sns.Topic(self, "PipelineNotifications")

        # 3. SageMaker Execution Role
        self.sagemaker_role = iam.Role(
            self, "SageMakerExecutionRole",
            assumed_by=iam.ServicePrincipal("sagemaker.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess"),
            ]
        )
        # Give SageMaker access to our specific S3 bucket
        self.data_bucket.grant_read_write(self.sagemaker_role)

        # 4. Lambda Function to trigger SageMaker Pipeline
        self.trigger_lambda = _lambda.Function(
            self, "PipelineTriggerLambda",
            runtime=_lambda.Runtime.PYTHON_3_10,
            handler="trigger_pipeline.handler",
            code=_lambda.Code.from_asset("infrastructure/lambda"),
            environment={
                "PIPELINE_NAME": "Detectron2-TrainingPipeline"
            }
        )

        # Grant Lambda permissions to start SageMaker Pipeline
        self.trigger_lambda.add_to_role_policy(
            iam.PolicyStatement(
                actions=["sagemaker:StartPipelineExecution"],
                resources=[f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/detectron2-trainingpipeline"]
            )
        )

        # 5. EventBridge Rule: Trigger Lambda when new data uploaded to specific S3 path
        rule = events.Rule(
            self, "S3UploadRule",
            event_pattern=events.EventPattern(
                source=["aws.s3"],
                detail_type=["Object Created"],
                detail={
                    "bucket": {
                        "name": [self.data_bucket.bucket_name]
                    },
                    "object": {
                        "key": [{"prefix": "uploads/"}]
                    }
                }
            )
        )
        rule.add_target(targets.LambdaFunction(self.trigger_lambda))

        # 6. EventBridge Rule: Pipeline State Change -> SNS
        pipeline_rule = events.Rule(
            self, "PipelineStateRule",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail_type=["SageMaker Model Building Pipeline Execution Status Change"],
                detail={
                    "pipelineArn": [f"arn:aws:sagemaker:{self.region}:{self.account}:pipeline/detectron2-trainingpipeline"]
                }
            )
        )
        pipeline_rule.add_target(targets.SnsTopic(self.notification_topic))

