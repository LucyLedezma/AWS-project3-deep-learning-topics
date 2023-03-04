import os

## local
os.environ['SM_CHANNEL_TRAINING'] = "./dogImages/train"
os.environ['SM_CHANNEL_VALIDATION'] = "./dogImages/valid"
os.environ['SM_CHANNEL_TEST'] = "./dogImages/test"
os.environ['SM_MODEL_DIR'] = "./model"

### ! python hpo.py
### Second option
from sagemaker.pytorch import PyTorchModel
from sagemaker.predictor import Predictor

estimator.model_data
model_location = "s3://sagemaker-us-east-1-996862753977/smdebugger-project3-img-classification-2023-03-03-14-43-48-681/output/model.tar.gz"
inference_model = PyTorchModel(
                entry_point="train_model.py",
                role=role,
                model_data=model_location,
                framework_version="1.8",
                py_version="py36"
)
predictor=inference_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)

### by attaching the training job

from sagemaker.pytorch import PyTorch
estimator = PyTorch.attach('smdebugger-project3-img-classification-2023-03-03-16-06-15-316')

predictor= estimator.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)