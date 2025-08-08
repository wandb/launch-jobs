import wandb
import random
from wandb.sdk import launch
import re
import requests

job_input_schema = {
    "type": "object",
    "properties": {
        "lower_bound": {
            "type": "number",
            "description": "Lower bound of the range",
        },
        "upper_bound": {
            "type": "string",
            "description": "Upper bound of the range",
        },
        "log_a_message": {
            "type": "boolean",
            "description": "Log a message to the console",
        },
        "api_key": {
            "type": "string",
            "description": "API key to use",
        },
    },
}

settings = wandb.Settings(
    disable_git=True,
    job_source="image",
    disable_job_creation=False
)

def make_name_dns_safe(name: str) -> str:
    resp = name.replace("_", "-").lower()
    resp = re.sub(r"[^a-z\.\-]", "", resp)
    # Actual length limit is 253, but we want to leave room for the generated suffix
    resp = resp[:200]
    return resp

with wandb.init(settings=settings) as run:
    
    launch.manage_wandb_config(
        include=["lower_bound", "upper_bound", "log_a_message", "api_key"],
        schema=job_input_schema,
    )
    
    lower_bound = run.config.get("lower_bound", 0)
    upper_bound = run.config.get("upper_bound", 10)
    log_a_message = run.config.get("log_a_message", False)
    api_key = run.config.get("api_key", "1234567890")
    
    for i in range(lower_bound, upper_bound):
        run.log({
            "step": i, 
            "a": random.random(), 
            "b": random.random(), 
            "c": random.random(),
        })
        
    service_name = make_name_dns_safe(f"ping-service-{run.entity}-{run.project}-{run.id}")
    url = f"http://{service_name}:8080"
    response = requests.get(f"{url}/ping")
    
    if log_a_message:
        print(response.text)
        
    run.log({
        "api_key": api_key,
        "response": response.text
    })  
    
    
    
