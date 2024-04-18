# NVIDIA NeMo Inference Microservice Deploy Job

Deploy a model from W&B Artifacts to the NVIDIA NeMo Inference Microservice.

This job accepts a compatible model artifact from W&B and deploys to an running NIM/Triton server. It converts supported models to the `.nemo` format

Deployment time varies by model and machine type. The base Llama2-7b config takes about 1 minute on GCP's `a2-ultragpu-1g`.

## Compatible model types

1. Llama2
2. StarCoder
3. NV-GPT (coming soon)

## User Quickstart

1. Create a queue if you don't have one already. See an example queue config below.

   ```yaml
   net: host
   gpus: all # can be a specific set of GPUs or `all` to use everything
   runtime: nvidia # also requires nvidia container runtime
   volume:
     - model-store:/model-store/
   ```

   ![image](https://github.com/wandb/launch-jobs/assets/15385696/d349e37a-ce1d-48b3-992f-1b4b617efa19)

2. Create this job in your project:

   ```bash
   wandb job create -n "deploy-to-nvidia-nemo-inference-microservice" \
      -e $ENTITY \
      -p $PROJECT \
      -E jobs/deploy_to_nvidia_nemo_inference_microservice/job.py \
      -g andrew/nim-updates \
      git https://github.com/wandb/launch-jobs
   ```

3. Launch an agent on your GPU machine:
   ```bash
   wandb launch-agent -e $ENTITY -p $PROJECT -q $QUEUE
   ```
4. Submit the deployment job with your desired configs from the [Launch UI](https://wandb.ai/launch). See `configs/` for examples.

   1. You can also submit via the CLI:

      ```bash
      CONFIG_JSON_FNAME="configs/example.json"

      wandb launch -j $ENTITY/$PROJECT/deploy-to-nvidia-nemo-inference-microservice:latest \
        -e $ENTITY \
        -p $PROJECT \
        -q $QUEUE \
        -c $CONFIG_JSON_FNAME
      ```

      ![image](https://github.com/wandb/launch-jobs/assets/15385696/8bc95b7a-94a6-453e-9c87-f6b25a567604)

5. You can track the deployment process in the Launch UI.
   ![image](https://github.com/wandb/launch-jobs/assets/15385696/49ca8391-689e-4cb7-9ba9-b5691f2cc7aa)
6. Once complete, you can immediately curl the endpoint to test the model. The model name is always `ensemble`.
   ```bash
    #!/bin/bash
    curl -X POST "http://0.0.0.0:9999/v1/completions" \
        -H "accept: application/json" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "ensemble",
            "prompt": "Tell me a joke",
            "max_tokens": 4096,
            "temperature": 0.5,
            "n": 1,
            "stream": false,
            "stop": "string",
            "frequency_penalty": 0.0
            }'
   ```
