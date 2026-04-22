#!/usr/bin/env python3
import os
import aws_cdk as cdk
from infrastructure.stack import MLOpsPipelineStack

app = cdk.App()
MLOpsPipelineStack(app, "Detectron2MLOpsPipelineStack",
    env=cdk.Environment(account=os.getenv('CDK_DEFAULT_ACCOUNT'), region=os.getenv('AWS_REGION', 'us-east-1')),
)

app.synth()
