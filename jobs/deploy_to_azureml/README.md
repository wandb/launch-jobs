# AzureML Online Endpoints Deploy Job

Deploy a model from W&B Artifacts to AzureML Online Endpoints

## 1. Build and run the deployer container, which loads a model from W&B Artifacts into AzureML Online Endpoints

NOTE: You will need to pass Azure creds to the container. There are a few ways of doing this, but the most straightforward is to directly pass `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` and `AZURE_TENANT_ID` as environment variables to the container.

```
docker buildx build -t $WANDB_NAME jobs/deploy_to_azureml --platform linux/amd64,linux/arm64 --push
docker run \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_ENTITY=$WANDB_ENTITY \
    -e WANDB_PROJECT=$WANDB_PROJECT \
    -e WANDB_NAME=$WANDB_NAME \
    -e AZURE_CLIENT_ID=$AZURE_CLIENT_ID \
    -e AZURE_CLIENT_SECRET=$AZURE_CLIENT_SECRET \
    -e AZURE_TENANT_ID=$AZURE_TENANT_ID \
    --rm --net=host \
    $WANDB_NAME
```

## Note on supported models

1. **Tensorflow**: We assume SavedModel format -- the artifact should look like a SavedModel directory. The endpoint assumes data in this shape: `{"data": input_tensor}`
2. **PyTorch**: We look for any `.pt` or `.pth` files and load the first one as the model. The endpoint assumes data in this shape: `{"data": input_tensor}`
3. **ONNX**: We look for any `.onnx` files and load the first one as the model. We use the ONNX metadata to determine input and output shapes. The endpoint assumes data in this shape: `{"onnx_input_name1": input_tensor1, "onnx_input_name2": input_tensor2, ...}`
