import asyncio
import random
import submitit
import time

def slow_multiplication(x, y):
    time.sleep(x*y)
    return x*y

async def main():
    executor = submitit.AutoExecutor(folder="logs")
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