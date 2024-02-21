# AzureML Online Endpoints Deploy Job

Deploy a model from W&B Artifacts to AzureML Online Endpoints.

This job accepts a model artifact from W&B and deploys it to an AzureML Online Endpoint. It infers supported model types from the artifact and auto-generates the required `main.py/score.py` files, and spins up both the Endpoint (if it doesn't exist) and the Deployment. It also adds logging for each request to the endpoint back to W&B, tracking the inputs, outputs, and any error messages.

## Prerequisites

### Azure

1. Ensure your AzureML workspace is set up. If you haven't already, [create an AzureML workspace](https://learn.microsoft.com/en-us/azure/machine-learning/quickstart-create-resources?view=azureml-api-2).
2. Ensure the client creds you're passing in has the basic AzureML permissions to spin up and deploy endpoints. The `Contributor` role is sufficient, but you might want tighter permissions. (See below "Note on permissions" if you run into other auth issues.)
   ```
   az role assignment create --assignee $AZURE_CLIENT_ID --role Contributor --scope /subscriptions/$AZURE_SUBSCRIPTION_ID
   ```

### W&B

1. The job requires a supported model saved as an artifact in W&B. Currently, the job supports:
   a. **Tensorflow** - We assume SavedModel format. The artifact should look like a SavedModel directory. - The endpoint accepts json with this shape: `{"data": input_tensor}`
   b. **PyTorch** - We look for any `.pt` or `.pth` files and load the first one as the model. - The endpoint accepts json with this shape: `{"data": input_tensor}`
   c. **ONNX** - We look for any `.onnx` files and load the first one as the model. - We use the ONNX metadata to determine input and output shapes. The endpoint accepts json with this pattern: `{"onnx_input_name1": input_tensor1, "onnx_input_name2": input_tensor2, ...}` (replace `onnx_input_name1` with whatever the actual input name is in the ONNX model).

## Usage

1. If running from scratch from this repo, build and run the deployer container. This creates and runs the Launch job to load a model from W&B Artifacts into AzureML Online Endpoints.

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

2. If the Launch job already exists in W&B, you can configure and run the job from the W&B UI or CLI.

## Note on permissions

1. The generated `main.py/score.py` uses `ManagedIdentityCredential` to authenticate with AzureML. The identity needs read access to `AZURE_KEYVAULT_NAME`. If not specified, the endpoint will be created but deployment will fail. You can use the following commands to grant access to the keyvault:
   ```
   az role assignment create --assignee $ENDPOINT_APP_ID --role "Key Vault Secrets User" --scope /$KEYVAULT_SCOPE
   az keyvault set-policy --name $AZURE_KEYVAULT_NAME --spn $ENDPOINT_APP_ID --secret-permissions get list --key-permissions get list
   ```
