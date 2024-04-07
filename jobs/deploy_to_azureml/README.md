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

   1. **Tensorflow** - We assume SavedModel format. The artifact should look like a SavedModel directory.
   2. **PyTorch** - We look for any `.pt` or `.pth` files and load the first one as the model.
   3. **ONNX** - We look for any `.onnx` files and load the first one as the model.

2. You will also need to [set up a launch queue](https://docs.wandb.ai/guides/launch/setup-launch) with an env file that contains `AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`, `AZURE_TENANT_ID`

## Usage

1. If the container doesn't already exist, build:

   ```shell
   docker buildx build -t $IMAGE_NAME:$IMAGE_TAG -f Dockerfile.wandb .
   ```

2. Add to launch agent queue by passing a valid launch config

   ```shell
      wandb launch -d $IMAGE_NAME:$IMAGE_TAG -q $YOUR_QUEUE -p $YOUR_PROJECT -c $YOUR_CONFIG
   ```

   For sample configs, see `configs/`. You can convert one of the configs with `yq` and `jq`. You'll need to update the azure configs to your own!

   ```
   YOUR_CONFIG="example.json" \
   EXAMPLE_CONFIG_YML="configs/pytorch.yml" \
   temp=$(yq eval -o=json $EXAMPLE_CONFIG_YML > $YOUR_CONFIG) \
   echo $temp | jq '{overrides: {run_config: .}}' > "$YOUR_CONFIG"
   ```

   If you want to see what the job will run through without actually deploying, set the config `dry_run: true`

3. If the Launch job already exists in W&B, you can configure and run the job from the W&B UI or CLI.

## Note on permissions

1. The generated `main.py/score.py` uses `ManagedIdentityCredential` to authenticate with AzureML. The identity needs read access to `AZURE_KEYVAULT_NAME`. If not specified, the endpoint will be created but deployment will fail. You can use the following commands to grant access to the keyvault:
   ```
   az role assignment create --assignee $ENDPOINT_APP_ID --role "Key Vault Secrets User" --scope /$KEYVAULT_SCOPE
   az keyvault set-policy --name $AZURE_KEYVAULT_NAME --spn $ENDPOINT_APP_ID --secret-permissions get list --key-permissions get list
   ```
