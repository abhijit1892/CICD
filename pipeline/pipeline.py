import os
import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.properties import PropertyFile
from sagemaker.pytorch import PyTorch, PyTorchProcessor
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.model import Model

def get_pipeline(
    region,
    role_arn,
    default_bucket,
    pipeline_name="Detectron2-TrainingPipeline",
    base_job_prefix="Detectron2"
):
    sagemaker_session = sagemaker.Session(default_bucket=default_bucket)
    
    # Parameters
    input_data_uri = ParameterString(
        name="InputDataS3Uri",
        default_value=f"s3://{default_bucket}/uploads/dataset.tar.gz"
    )
    metric_threshold = ParameterFloat(
        name="mAPThreshold",
        default_value=50.0
    )
    
    # Define SageMaker Estimators and Processors
    # We use PyTorch DLC as the base and install detectron2 at runtime via requirements.txt
    framework_version = "2.0.0"
    py_version = "py310"
    
    # 1. Validation Step (Processing)
    processor = PyTorchProcessor(
        framework_version=framework_version,
        py_version=py_version,
        role=role_arn,
        instance_count=1,
        instance_type="ml.m5.xlarge",
        base_job_name=f"{base_job_prefix}-validation",
        sagemaker_session=sagemaker_session,
    )
    
    validation_step = ProcessingStep(
        name="DataValidation",
        processor=processor,
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=input_data_uri,
                destination="/opt/ml/processing/input",
                s3_data_type="S3Prefix"
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="validation_output",
                destination=f"s3://{default_bucket}/{base_job_prefix}/validation",
                source="/opt/ml/processing/output"
            )
        ],
        code="pipeline/scripts/validate.py"
    )
    
    # 2. Training Step
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="pipeline/scripts",
        role=role_arn,
        framework_version=framework_version,
        py_version=py_version,
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        base_job_name=f"{base_job_prefix}-training",
        sagemaker_session=sagemaker_session,
        # Give more volume for datasets and checkpointer
        volume_size=60,
    )
    
    training_step = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": sagemaker.inputs.TrainingInput(
                s3_data=input_data_uri,
                content_type="application/x-tar"
            )
        },
        depends_on=[validation_step]
    )
    
    # 3. Evaluation Step
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json"
    )
    
    eval_step = ProcessingStep(
        name="EvaluateModel",
        processor=processor, # reuse processor
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model"
            ),
            sagemaker.processing.ProcessingInput(
                source=input_data_uri,
                destination="/opt/ml/processing/input"
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/evaluation"
            )
        ],
        code="pipeline/scripts/evaluate.py",
        property_files=[evaluation_report],
    )
    
    from sagemaker.workflow.functions import Join
    
    # 4. Register Model Step (Conditional)
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri=Join(
                on="/",
                values=[
                    eval_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                    "evaluation.json"
                ]
            ).to_string(),
            content_type="application/json"
        )
    )

    model = Model(
        image_uri=estimator.training_image_uri(),
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=role_arn,
    )

    register_step = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["image/png", "image/jpeg"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge", "ml.g4dn.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            model_package_group_name="Detectron2PackageGroup",
            model_metrics=model_metrics,
            approval_status="Approved",
        )
    )
    
    # Condition: Is mAP50 >= threshold?
    cond_lte = ConditionGreaterThanOrEqualTo(
        left=evaluation_report.read_dict("metrics.mAP50.value"),
        right=metric_threshold,
    )
    
    condition_step = ConditionStep(
        name="CheckEvaluationCondition",
        conditions=[cond_lte],
        if_steps=[register_step],
        else_steps=[],
    )
    
    # Create Pipeline
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[input_data_uri, metric_threshold],
        steps=[validation_step, training_step, eval_step, condition_step],
        sagemaker_session=sagemaker_session
    )
    
    return pipeline
