import os
from typing import Optional

SECRET_REF_PREFIX = "secret://"


def get_launch_secret_from_env(secret_key: str, config_dict: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Get a secret from the environment.
    """
    secret_ref = config_dict.get(secret_key)
    if secret_ref is None:
        return None, None
    env_key = secret_ref.replace(SECRET_REF_PREFIX, "").upper()
    secret_value = os.getenv(env_key)
    if secret_value is None:
        return None, None
    return env_key, secret_value