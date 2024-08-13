import asyncio
import os
import random
import submitit
import time
import wandb
from wandb.sdk import launch

def slow_multiplication(x, y):
    time.sleep(x*y)
    return x*y

async def main():
    print(f"Running in conda env: {os.getenv("CONDA_DEFAULT_ENV")}")
    print(f"From directory: {os.getcwd()}")

    executor = submitit.AutoExecutor(folder="logs")
    wandb.init(project="submitit-test", config={
        "submitit": {"timeout_min": None, "partition": None}
    })
    launch.manage_wandb_config(include="submitit")
    executor.update_parameters(timeout_min=1)

    # await a single result
    job = executor.submit(slow_multiplication, 10, 2)
    await job.awaitable().result()

    # print results as they become available
    jobs = [executor.submit(slow_multiplication, k, random.randint(1, 4)) for k in range(1, 5)]
    for aws in asyncio.as_completed([j.awaitable().result() for j in jobs]):
        result = await aws
        print(result)

if __name__ == "__main__":
    asyncio.run(main())