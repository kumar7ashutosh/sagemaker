import os
import boto3
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.pipeline import Pipeline

from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import ModelMetrics, MetricsSource
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker import image_uris


# ======================================================================================
# 🔧 1. Session + Role + Region Setup
# ======================================================================================

region = os.environ.get("AWS_REGION", boto3.Session().region_name)
boto_sess = boto3.Session(region_name=region)
pipeline_session = PipelineSession(boto_session=boto_sess)

# CI/CD variable (GitHub Actions)
role = os.environ.get("SAGEMAKER_ROLE_ARN")
if role is None:
    raise ValueError("❌ SAGEMAKER_ROLE_ARN environment variable not provided.")

bucket = os.environ.get("S3_BUCKET")
if bucket is None:
    raise ValueError("❌ S3_BUCKET env variable is missing.")

prefix = "sagemaker-churn-mlops"


# ======================================================================================
# 🔧 2. Pipeline Parameters
# ======================================================================================

input_data = ParameterString(
    name="InputData",
    default_value=f"s3://{bucket}/{prefix}/raw/churn.csv"
)

model_approval = ParameterString(
    name="ModelApprovalStatus",
    default_value="PendingManualApproval"
)

accuracy_threshold = ParameterFloat(
    name="AccuracyThreshold",
    default_value=0.50
)


# ======================================================================================
# 🔧 3. Step 1 — Processing Step
# ======================================================================================

sklearn_processor = SKLearnProcessor(
    framework_version="1.2-1",
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="churn-preprocess",
    sagemaker_session=pipeline_session,
)

processing_step = ProcessingStep(
    name="PreprocessStep",
    processor=sklearn_processor,
    inputs=[
        ProcessingInput(
            source=input_data,
            destination="/opt/ml/processing/input"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="train", source="/opt/ml/processing/train"),
        ProcessingOutput(output_name="test", source="/opt/ml/processing/test")
    ],
    code="src/preprocessing.py",
)


# ======================================================================================
# 🔧 4. Step 2 — Training Step (Script Mode XGBoost)
# ======================================================================================

xgb_image = image_uris.retrieve(
    framework="xgboost",
    region=region,
    version="1.5-1",
)

estimator = Estimator(
    entry_point="src/train.py",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    image_uri=xgb_image,
    sagemaker_session=pipeline_session,
    base_job_name="churn-train",
)

training_step = TrainingStep(
    name="TrainStep",
    estimator=estimator,
    inputs={
        "train": TrainingInput(
            s3_data=processing_step.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
            content_type="text/csv"
        )
    }
)


# ======================================================================================
# 🔧 5. Step 3 — Evaluation Step
# ======================================================================================

evaluation_processor = ScriptProcessor(
    role=role,
    image_uri=xgb_image,
    command=["python3"],
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="churn-eval",
    sagemaker_session=pipeline_session,
)

evaluation_report = PropertyFile(
    name="evaluation",
    output_name="evaluation",
    path="evaluation.json"
)

eval_step = ProcessingStep(
    name="EvaluateStep",
    processor=evaluation_processor,
    code="src/evaluate.py",
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        )
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ],
    property_files=[evaluation_report]
)


# ======================================================================================
# 🔧 6. Step 4 — Register Model Step
# ======================================================================================

model = Model(
    image_uri=xgb_image,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=pipeline_session,
)

model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=eval_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
        content_type="application/json"
    )
)

register_step = RegisterModel(
    name="RegisterModel",
    model=model,
    content_types=["text/csv"],
    response_types=["text/csv"],
    model_package_group_name="ChurnXGBModelGroup",
    approval_status=model_approval,
    model_metrics=model_metrics,
)


# ======================================================================================
# 🔧 7. Condition Step (Check Accuracy >= Threshold)
# ======================================================================================

cond_step = ConditionStep(
    name="CheckAccuracy",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=eval_step.name,
                property_file="evaluation",
                json_path="accuracy"
            ),
            right=accuracy_threshold,
        )
    ],
    if_steps=[register_step],
    else_steps=[],
)


# ======================================================================================
# 🔧 8. Build Pipeline
# ======================================================================================

pipeline = Pipeline(
    name="ChurnXGBPipeline",
    parameters=[input_data, model_approval, accuracy_threshold],
    steps=[processing_step, training_step, eval_step, cond_step],
    sagemaker_session=pipeline_session,
)


if __name__ == "__main__":
    print("💡 Creating/Updating SageMaker Pipeline...")
    pipeline.upsert(role_arn=role)
    print("🚀 Executing SageMaker Pipeline...")
    execution = pipeline.start()
    print("⏳ Waiting for completion...")
    execution.wait()
    print("✅ Pipeline execution completed successfully!")
