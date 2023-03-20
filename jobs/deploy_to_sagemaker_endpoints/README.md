# Sagemaker Endpoints Deploy Job

Deploy a model from W&B Artifacts to Sagemaker Endpoints

## 1. Build and run the sagemaker deploy container, which loads a model from W&B Artifacts into Sagemaker Endpoints

- This requires a valid AWS config located at `.aws` in your home directory.

```
docker build -t $WANDB_NAME jobs/sagemaker-endpoints && \
docker run \
   -v $HOME/.aws:/root/.aws:ro \
   -e WANDB_API_KEY=$WANDB_API_KEY \
   -e WANDB_ENTITY=$WANDB_ENTITY \
   -e WANDB_PROJECT=$WANDB_PROJECT \
   -e WANDB_NAME=$WANDB_NAME \
   $WANDB_NAME
```
