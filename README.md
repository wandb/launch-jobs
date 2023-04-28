# launch-jobs

This repo contains a collection of jobs that can be run on W&B Launch.

## Notes

Internal users: Run the notebook `loader.ipynb` to load jobs into the `wandb/jobs` project. Some jobs may take a while to run. Notably, the deploy jobs including Sagemaker and Triton take a while because they verify that the endpoint is running before exiting.

## Adding new jobs

Add a new directory in `jobs/` with the following structure:

```
jobs/
└── new_job_name
    ├── configs/config.yml
    ├── Dockerfile
    ├── job.py
    ├── requirements.txt
    └── README.md
```
