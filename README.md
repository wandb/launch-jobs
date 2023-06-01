# Weights & Biases Launch Jobs

[W&B Launch](https://docs.wandb.ai/guides/launch) introduces a connective layer between machine learning practitioners and the high-scale, specialized hardware that powers modern machine learning workflows. Easily scale training runs from your desktop to your GPUs, quickly spin up intensive model evaluation suites, and prepare models for production inference, all without the friction of complex infrastructure.

[A Launch Job](https://docs.wandb.ai/guides/launch/create-job) is a complete blueprint of how to perform a step in your ML workflow, like training a model, running an evaluation, or deploying a model to an inference server.


In this repository, we have curated a set of popular jobs for common tasks, such as deploying a model to serve inference.

# Deploy models from W&B
- [Deploy to Triton](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_nvidia_triton): Deploy a model to NVIDIA Triton Inference Server
- [Deploy to SageMaker](https://github.com/wandb/launch-jobs/tree/main/jobs/deploy_to_sagemaker_endpoints): Deploy a model to SageMaker Endpoints

## Evaluate and optimize models
- [Run OpenAI Evals](https://github.com/wandb/launch-jobs/tree/main/jobs/openai_evals) to evaluate LLMs, and follow along with the [companion guide](https://wandb.ai/wandb_fc/openai-evals/reports/OpenAI-Evals-Demo-Using-W-B-Prompts-to-Run-Evaluations--Vmlldzo0MTI4ODA3).
- [Use NVIDIA TensorRT](https://github.com/wandb/launch-jobs/tree/main/jobs/gpu_optimize_with_tensor_rt) to quantize and optimize a model.

## Connect to external services
- [Dispatch GitHub Actions](https://github.com/wandb/launch-jobs/tree/main/jobs/github_actions_workflow_dispatch): Trigger a workflow to run in GitHub Actions
- [Microsoft Teams Webhook](https://github.com/wandb/launch-jobs/tree/main/jobs/msft_teams_webhook): Send a Teams alert when a new model has been deployed

## Contribute a job
Have you created a job that you think other W&B users would like to try? For example, perhaps you're deploying models to another service. If you'd like to share your work wtih the community, see CONTRIBUTING.md for instructions on how to add new jobs to this repo.
