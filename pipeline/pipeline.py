import os
import boto3
from sagemaker.model import Model
from sagemaker.workflow.parameters import ParameterString, ParameterFloat
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, CacheConfig
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.properties import PropertyFile  # Added for evaluation.json par
# for sagemaker execution
region = boto3.Session().region_name
pipeline_session = PipelineSession()


# for CICD execution
# region = os.environ.get("AWS_REGION", "us-east-1")
# boto_session = boto3.Session(region_name=region)
# pipeline_session = PipelineSession(boto_session=boto_session)


role = "sagemaker-execution-role"  # Please enter your role here


# -----------------------------
# Pipeline Parameters
# -----------------------------
input_data = ParameterString(name="InputData", default_value="s3://sagemaker-churn-mlops-ka/sagemaker-churn-mlops/raw/churn.csv")
model_approval = ParameterString(name="ModelApprovalStatus", default_value="PendingManualApproval")
accuracy_threshold = ParameterFloat(name="AccuracyThreshold", default_value=0.50)

# -----------------------------
# Step 1: Preprocessing
# -----------------------------
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
        ProcessingOutput(
            output_name="train",
            source="/opt/ml/processing/train"
        ),
        ProcessingOutput(
            output_name="test",
            source="/opt/ml/processing/test"
        )
    ],
    code="src/preprocessing.py"
)

# -----------------------------
# Step 2: Training
# -----------------------------
estimator = Estimator(
    entry_point="src/train.py",
    role=role,
    instance_type="ml.m5.large",
    instance_count=1,
    base_job_name="churn-train",
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.3-1",
    py_version="py3",
    sagemaker_session=pipeline_session
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

model = Model(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1",
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    role=role,
    sagemaker_session=pipeline_session
)

# -----------------------------
# Step 3: Evaluation
# -----------------------------
script_eval = ScriptProcessor(
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-xgboost:1.5-1",
    command=["python3"],
    role=role,
    instance_type="ml.t3.medium",
    instance_count=1,
    base_job_name="churn-eval",
    sagemaker_session=pipeline_session,
)

# ✅ PropertyFile to extract accuracy
evaluation_report = PropertyFile(
    name="evaluation",
    output_name="evaluation",
    path="evaluation.json"
)

eval_step = ProcessingStep(
    name="EvaluateStep",
    processor=script_eval,
    code="src/evaluate.py",
    inputs=[
        ProcessingInput(
            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
            destination="/opt/ml/processing/model"
        ),
        ProcessingInput(
            source=processing_step.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
            destination="/opt/ml/processing/test"
        ),
    ],
    outputs=[
        ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")
    ],
    property_files=[evaluation_report]  # ✅ Add this line
)

# -----------------------------
# Step 4: Conditional Model Registration
# -----------------------------
model_metrics = ModelMetrics(
    model_statistics=MetricsSource(
        s3_uri=eval_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
        content_type="application/json"
    )
)

register_step = RegisterModel(
    name="RegisterModel",
    model=model,
    model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
    content_types=["text/csv"],
    response_types=["text/csv"],
    model_package_group_name="ChurnXGBModelGroup",
    approval_status=model_approval,
    model_metrics=model_metrics,
    sagemaker_session=pipeline_session,
)

# ✅ Fix ConditionStep using JsonGet
cond_step = ConditionStep(
    name="CheckAccuracy",
    conditions=[
        ConditionGreaterThanOrEqualTo(
            left=JsonGet(
                step_name=eval_step.name,
                property_file="evaluation",
                json_path="accuracy"
            ),
            right=accuracy_threshold
        )
    ],
    if_steps=[register_step],
    else_steps=[]
)

# -----------------------------
# Create Pipeline
# -----------------------------
pipeline = Pipeline(
    name="ChurnXGBPipeline",
    parameters=[input_data, model_approval, accuracy_threshold],
    steps=[processing_step, training_step, eval_step, cond_step],
    sagemaker_session=pipeline_session,
)
