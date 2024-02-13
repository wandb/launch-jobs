# AzureML Online Endpoints Deploy Job

Deploy a model from W&B Artifacts to AzureML Online Endpoints.

This job accepts a model artifact from W&B and deploys it to an AzureML Online Endpoint. It infers supported model types from the artifact and auto-generates the required `main.py/score.py` files, and spins up both the Endpoint (if it doesn't exist) and the Deployment. It also adds logging for each request to the endpoint back to W&B, tracking the inputs, outputs, and any error messages.

## 1. Build and run the deployer container, which loads a model from W&B Artifacts into AzureML Online Endpoints

NOTE: You will need to pass Azure creds to the container. There are a few ways of doing this, but the most straightforward is to directly pass `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET` and `AZURE_TENANT_ID` as environment variables to the container.

```
docker buildx build -t $WANDB_NAME jobs/deploy_to_azureml

docker run \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_ENTITY=$WANDB_ENTITY \
    -e WANDB_PROJECT=$WANDB_PROJECT \
    -e WANDB_NAME=$WANDB_NAME \
    -e AZURE_CLIENT_ID=$AZURE_CLIENT_ID \
    -e AZURE_CLIENT_SECRET=$AZURE_CLIENT_SECRET \
    -e AZURE_TENANT_ID=$AZURE_TENANT_ID \
    --rm --net=host \

    # if you're running from scratch, you may need to pass these env vars (which correspond to the config):
    -e AZURE_SUBSCRIPTION_ID=$AZURE_SUBSCRIPTION_ID \
    -e AZURE_RESOURCE_GROUP=$AZURE_RESOURCE_GROUP \
    -e AZURE_WORKSPACE=$AZURE_WORKSPACE \
    -e AZURE_KEYVAULT_NAME=$AZURE_KEYVAULT_NAME \
    -e AZURE_ENDPOINT_NAME=$AZURE_ENDPOINT_NAME \
    -e AZURE_DEPLOYMENT_NAME=$AZURE_DEPLOYMENT_NAME \
    -e WANDB_ARTIFACT_PATH=$WANDB_ARTIFACT_PATH \

    $WANDB_NAME
```

## Note on permissions

1. The generated `main.py/score.py` uses `ManagedIdentityCredential` to authenticate with AzureML. The identity will also need secrets read access to the specified `AZURE_KEYVAULT_NAME`. If this is not specified, the endpoint will be created but the deployment will fail. You can use the following command to grant the access to the keyvault:
   ```
   az role assignment create --assignee $ENDPOINT_APP_ID --role "Key Vault Secrets User" --scope /$KEYVAULT_SCOPE
   ```

## Note on supported models

1. **Tensorflow**: We assume SavedModel format -- the artifact should look like a SavedModel directory. The endpoint assumes data in this shape: `{"data": input_tensor}`
2. **PyTorch**: We look for any `.pt` or `.pth` files and load the first one as the model. The endpoint assumes data in this shape: `{"data": input_tensor}`
3. **ONNX**: We look for any `.onnx` files and load the first one as the model. We use the ONNX metadata to determine input and output shapes. The endpoint assumes data in this shape: `{"onnx_input_name1": input_tensor1, "onnx_input_name2": input_tensor2, ...}`
