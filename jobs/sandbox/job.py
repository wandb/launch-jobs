"""Sandbox executor launch job.

Reads run config from wandb launch, resolves ``secret://`` references from
the container's environment (where launch injected them), and starts an
aviato sandbox.

The run_config should contain only sandbox primitives — the CLI is
responsible for putting everything the sandbox image needs into
``command``/``args`` and ``env_vars``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from typing import Any

import wandb
from aviato import Sandbox
from pydantic import BaseModel, ConfigDict, Field, model_validator
from wandb.sdk import launch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_SECRET_PREFIX = "secret://"


class MountedFile(BaseModel):
    path: str
    content: str


class SandboxConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image: str
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env_vars: dict[str, str] = Field(default_factory=dict)
    resources: dict[str, Any] | None = None
    timeout: float | None = None
    files_artifact: str | None = None
    tags: list[str] | None = None
    tower_ids: list[str] | None = None

    @model_validator(mode="after")
    def _check_args_require_command(self) -> SandboxConfig:
        if self.command is None and self.args:
            raise ValueError(
                "'args' were provided but 'command' is not set — "
                "set 'command' or remove 'args'"
            )
        return self


def _resolve_secrets(config: dict) -> dict[str, str]:
    """Resolve ``secret://`` references from the container environment.

    Returns a dict keyed by the config field name, with the resolved
    secret value.  e.g. ``{"model_api_key": "<actual-value>"}``.
    """
    resolved = {}
    for field, ref in config.items():
        if not isinstance(ref, str) or not ref.startswith(_SECRET_PREFIX):
            continue
        env_name = ref[len(_SECRET_PREFIX):].upper()
        env_value = os.environ.get(env_name)
        if env_value is not None:
            resolved[field.upper()] = env_value
        else:
            logger.warning(
                "secret ref %s for field %s could not be resolved — "
                "%s not in container env",
                ref, field, env_name,
            )
    return resolved


config = launch.load_wandb_config()

logger.info("Loaded run config: %s", json.dumps(dict(config), indent=2))

sandbox_config = SandboxConfig(**config)
logger.info("Parsed env_vars from config: %s", sandbox_config.env_vars)

resolved_secrets = _resolve_secrets(config)
logger.info("Resolved secret keys: %s", list(resolved_secrets.keys()))

env = {**sandbox_config.env_vars, **resolved_secrets}
wandb_api_key = os.environ.get('WANDB_API_KEY')
if wandb_api_key:
    env['WANDB_API_KEY'] = wandb_api_key
else:
    logger.warning("WANDB_API_KEY is not set — the sandbox will not be able to authenticate with W&B")

logger.info("Final sandbox env keys: %s", sorted(env.keys()))

# Write env vars into the pod's own environment so that wandb.init(),
# weave.init(), and any other SDK calls in this process can pick them up.
overwritten = [k for k, v in env.items() if os.environ.get(k) not in (None, v)]
if overwritten:
    logger.warning("Overwriting existing env vars: %s", ", ".join(sorted(overwritten)))
for k, v in env.items():
    os.environ[k] = v

mounted_files: list[MountedFile] | None = None
if sandbox_config.files_artifact:
    api = wandb.Api()
    artifact = api.artifact(sandbox_config.files_artifact)
    logger.info("Resolving files artifact %s", sandbox_config.files_artifact)

    # Entry names are relative (leading / stripped by CLI).
    # Download all files into a temp dir, then reconstruct absolute
    # mount paths by prepending /.
    _MAX_MOUNTED_FILE_SIZE = 1 * 1024 * 1024  # 1 MiB

    mounted_files = []
    with tempfile.TemporaryDirectory() as tmpdir:
        artifact.download(root=tmpdir)
        real_tmpdir = os.path.realpath(tmpdir)
        for entry in artifact.manifest.entries.values():
            norm_path = os.path.normpath(entry.path)
            if os.path.isabs(norm_path) or norm_path.startswith(".."):
                raise ValueError(
                    f"Artifact entry has unsafe path: {entry.path}"
                )
            file_path = os.path.join(tmpdir, norm_path)
            if not os.path.realpath(file_path).startswith(real_tmpdir + os.sep):
                raise ValueError(
                    f"Artifact entry escapes temp directory: {entry.path}"
                )
            file_size = os.path.getsize(file_path)
            if file_size > _MAX_MOUNTED_FILE_SIZE:
                logger.warning(
                    "File %s is %d bytes which exceeds the 1 MiB mounted "
                    "file size limit — sandbox creation may fail.",
                    entry.path, file_size,
                )
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                raise ValueError(
                    f"Artifact file {entry.path} is not valid UTF-8 — "
                    "mounted files only support text content"
                )
            mounted_files.append(MountedFile(path="/" + norm_path, content=content))

if sandbox_config.command is not None:
    cmd_args = [sandbox_config.command, *sandbox_config.args]
else:
    cmd_args = []
logger.info("Starting sandbox: image=%s command=%s args=%s", sandbox_config.image, sandbox_config.command, sandbox_config.args)

mounted_files_dicts = [f.model_dump() for f in mounted_files] if mounted_files else None

with Sandbox.run(
    *cmd_args,
    container_image=sandbox_config.image,
    environment_variables=env,
    resources=sandbox_config.resources,
    max_lifetime_seconds=sandbox_config.timeout,
    mounted_files=mounted_files_dicts,
    tags=sandbox_config.tags,
    tower_ids=sandbox_config.tower_ids,
) as sandbox:
    logger.info("Sandbox running: id=%s tower=%s", sandbox.sandbox_id, sandbox.tower_id)
    sandbox.wait_until_complete()

logger.info("Sandbox finished: id=%s status=%s exit_code=%s", sandbox.sandbox_id, sandbox.status, sandbox.returncode)
sys.exit(sandbox.returncode if sandbox.returncode is not None else (0 if sandbox.status == "success" else 1))
