import json
import os

import pymsteams
from tenacity import Retrying, stop_after_attempt, wait_random_exponential

import wandb

settings = wandb.Settings(disable_git=True)

with wandb.init(settings=settings) as run:
    msg = pymsteams.connectorcard(run.config["webhook_url"])
    msg.color(run.config["color"])
    msg.text(" ")

    banner = pymsteams.cardsection()
    banner.activityTitle(f"W&B Trigger: {run.config['title']}")
    banner.activitySubtitle("A new model has been deployed!")
    banner.activityImage(
        "https://assets.website-files.com/5ac6b7f2924c656f2b13a88c/6376969889260ee072decf4c_Models-Icon.png"
    )
    msg.addSection(banner)

    info = pymsteams.cardsection()
    info_dict = {
        "artifact": run.config["artifact"].name,
        "version": run.config["artifact"].version,
        "description": run.config["artifact"].description,
        "aliases": run.config["artifact"].aliases,
        "commit": run.config["artifact"].commit_hash,
    }
    info.activityTitle("Deploy Details")
    for k, v in info_dict.items():
        info.addFact(k, json.dumps(v))
    msg.addSection(info)

    debug = pymsteams.cardsection()
    debug.title("DEBUG INFO")
    for k, v in run.config.items():
        if k == "artifact":
            debug.addFact("artifact", run.config["artifact"].name)
        else:
            debug.addFact(k, json.dumps(v))
    msg.addSection(debug)

    base_url = "https://wandb.ai"
    entity = run.config["artifact"].entity
    project = run.config["artifact"].project
    typ = run.config["artifact"].type
    name = run.config["artifact"].name
    msg.addLinkButton(
        "View artifact",
        os.path.join(
            base_url,
            entity,
            project,
            "artifacts",
            typ,
            name.replace(":v", "/v"),
        ),
    )

    for attempt in Retrying(
        stop=stop_after_attempt(run.config["retry_settings"]["attempts"]),
        wait=wait_random_exponential(**run.config["retry_settings"]["backoff"]),
    ):
        with attempt:
            msg.send()
