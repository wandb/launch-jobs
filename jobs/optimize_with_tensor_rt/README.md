# NVIDIA TensorRT Model Optimization Job

Optimize a model from W&B Artifacts with NVIDIA TensorRT and save the optimized model to W&B Artifacts.

Note: This job currently supports TensorFlow 2. We are working on adding support for PyTorch.

## 1. Build and run the TensorRT optimization container

- This uses the [NVIDIA container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) and an image from NGC. If you don't have an NGC account, you can create one [here](https://ngc.nvidia.com/signin).

```
docker build -t $WANDB_NAME jobs/tensor-rt && \
docker run \
    --gpus all \
    --runtime=nvidia \
   -e WANDB_API_KEY=$WANDB_API_KEY \
   -e WANDB_ENTITY=$WANDB_ENTITY \
   -e WANDB_PROJECT=$WANDB_PROJECT \
   -e WANDB_NAME=$WANDB_NAME \
    $WANDB_NAME
```
