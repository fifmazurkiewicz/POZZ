import json
import os
from typing import Optional

try:
    from dotenv import load_dotenv  # type: ignore
except Exception:  # Fallback if python-dotenv is unavailable in lint envs
    def load_dotenv(*args, **kwargs):  # type: ignore
        return False

try:
    import boto3  # type: ignore
    from botocore.exceptions import BotoCoreError, ClientError  # type: ignore
except Exception:  # boto3 may not be available in local minimal setups
    boto3 = None
    BotoCoreError = Exception
    ClientError = Exception


def get_environment() -> str:
    """Return the current environment: 'local' or non-local (e.g., 'prod', 'staging')."""
    return os.getenv("ENVIRONMENT", "local").strip().lower()


def ensure_local_env_loaded() -> None:
    """Load .env for local development."""
    if get_environment() == "local":
        load_dotenv(override=False)


def _fetch_secret_from_aws(secret_name: str, region_name: Optional[str]) -> Optional[str]:
    """Fetch a secret value from AWS Secrets Manager.

    The secret can be either a plain string (raw key) or a JSON blob containing
    the field 'OPENROUTER_API_KEY'.
    """
    if boto3 is None:
        return None

    region = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-central-1"
    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
    except (BotoCoreError, ClientError):
        return None

    secret_str = response.get("SecretString")
    if not secret_str:
        return None

    # Try JSON first
    try:
        data = json.loads(secret_str)
        if isinstance(data, dict) and "OPENROUTER_API_KEY" in data:
            return str(data["OPENROUTER_API_KEY"]).strip()
    except json.JSONDecodeError:
        pass

    # Fallback: treat as raw key
    return secret_str.strip()


def get_openrouter_api_key() -> Optional[str]:
    """Return OpenRouter API key using source appropriate for the environment.

    - local: from .env / env var OPENROUTER_API_KEY
    - non-local: from AWS Secrets Manager (OPENROUTER_SECRET_NAME) or env var
    """
    ensure_local_env_loaded()

    if get_environment() == "local":
        return os.getenv("OPENROUTER_API_KEY")

    # Non-local: try Secrets Manager first
    secret_name = os.getenv("OPENROUTER_SECRET_NAME", "med-sim/openrouter")
    region = os.getenv("AWS_REGION")
    key = _fetch_secret_from_aws(secret_name=secret_name, region_name=region)
    if key:
        return key

    # Fallback to env var if secret missing
    return os.getenv("OPENROUTER_API_KEY")


def get_database_url() -> Optional[str]:
    """Return PostgreSQL connection URL.

    - local: from .env / env var DATABASE_URL
    - non-local: from AWS Secrets Manager (POSTGRES_SECRET_NAME) or env var
    Secret can be raw DSN or JSON with field DATABASE_URL.
    """
    ensure_local_env_loaded()

    if get_environment() == "local":
        return os.getenv("DATABASE_URL")

    secret_name = os.getenv("POSTGRES_SECRET_NAME", "POZZ")
    region = os.getenv("AWS_REGION")
    value = _fetch_secret_from_aws(secret_name=secret_name, region_name=region)
    if value:
        # Try JSON wrapper
        try:
            data = json.loads(value)
            if isinstance(data, dict) and "DATABASE_URL" in data:
                return str(data["DATABASE_URL"]).strip()
        except json.JSONDecodeError:
            pass
        return value

    return os.getenv("DATABASE_URL")


