# NVIDIA Triton Inference Server Deploy Job

Deploy a model from W&B Artifacts to NVIDIA Triton Inference Server

## 1. Build and run the Triton server container

```
docker build -t tritonserver-wandb jobs/deploy_to_nvidia_triton/server && \
docker run \
  -v $HOME/.aws:/root/.aws:ro \
  -p 8000:8000 \
  --rm --net=host -d \
  tritonserver-wandb
```

## 2. Build and run the deployer container, which loads a model from W&B Artifacts into NVIDIA Triton Inference Server

```
docker build -t $WANDB_NAME jobs/deploy_to_nvidia_triton/deployer && \
docker run \
   -v $HOME/.aws:/root/.aws:ro \
   -e WANDB_API_KEY=$WANDB_API_KEY \
   -e WANDB_ENTITY=$WANDB_ENTITY \
   -e WANDB_PROJECT=$WANDB_PROJECT \
   -e WANDB_NAME=$WANDB_NAME \
   --rm --net=host \
   $WANDB_NAME
```

## Note on testing

### Keras

- Tested using keras savedmodel and torchscript on CPU

### Pytorch

- Note: For pytorch, your model needs to have already been converted to torchscript and saved to Artifacts before uploading -- currently investigating if we can do the conversion automatically.

### Ensemble

- Note: For ensemble, you must package the component models and define the config.pbtxt ahead of time. The component models can be referenced from other artifacts. See [this artifact](https://wandb.ai/megatruong/tritonserver-ensemble-testing10/artifacts/ensemble_model/ensemble_model/v0/files) for an example of how to structure your artifact, and the [ensemble directory](https://wandb.ai/megatruong/tritonserver-ensemble-testing10/artifacts/ensemble_model/ensemble_model/v0/files/ensemble) to see an example `config.pbtxt`.
