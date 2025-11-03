import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

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
        logger.warning("boto3 is not available - cannot fetch secrets from AWS Secrets Manager")
        return None

    region = region_name or os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "eu-central-1"
    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
    except ClientError as e:
        response = getattr(e, "response", None)  # type: ignore
        if response and isinstance(response, dict):
            error_code = response.get("Error", {}).get("Code", "Unknown")
            if error_code == "ResourceNotFoundException":
                logger.warning(f"AWS Secret not found: {secret_name} in region {region}")
            elif error_code == "AccessDeniedException":
                logger.warning(f"Access denied to AWS Secret: {secret_name} in region {region}")
            else:
                logger.warning(f"Error fetching AWS Secret {secret_name} from region {region}: {error_code}")
        else:
            logger.warning(f"Error fetching AWS Secret {secret_name} from region {region}: {type(e).__name__}")
        return None
    except (BotoCoreError, Exception) as e:
        logger.warning(f"Error connecting to AWS Secrets Manager (secret={secret_name}, region={region}): {type(e).__name__}")
        return None

    secret_str = response.get("SecretString")
    if not secret_str:
        logger.warning(f"AWS Secret {secret_name} has no SecretString")
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


def get_openai_api_key() -> Optional[str]:
    """Return OpenAI API key (optional, for direct Whisper access).
    
    Checks OPENAI_API_KEY env var or Secrets Manager (OPENAI_SECRET_NAME).
    """
    ensure_local_env_loaded()
    
    if get_environment() == "local":
        return os.getenv("OPENAI_API_KEY")
    
    secret_name = os.getenv("OPENAI_SECRET_NAME", "med-sim/openai")
    region = os.getenv("AWS_REGION")
    key = _fetch_secret_from_aws(secret_name=secret_name, region_name=region)
    if key:
        try:
            data = json.loads(key)
            if isinstance(data, dict) and "OPENAI_API_KEY" in data:
                return str(data["OPENAI_API_KEY"]).strip()
        except json.JSONDecodeError:
            pass
        return key
    
    return os.getenv("OPENAI_API_KEY")


def get_database_url() -> Optional[str]:
    """Return PostgreSQL connection URL.

    - local: from .env / env var DATABASE_URL
    - non-local: from AWS Secrets Manager (POSTGRES_SECRET_NAME) or env var
    Secret can be raw DSN or JSON with field DATABASE_URL.
    """
    ensure_local_env_loaded()

    env = get_environment()
    if env == "local":
        db_url = os.getenv("DATABASE_URL")
        if not db_url:
            logger.warning("ENVIRONMENT=local but DATABASE_URL not found in .env or environment variables")
        return db_url

    # Non-local environment: try Secrets Manager
    secret_name = os.getenv("POSTGRES_SECRET_NAME", "POZZ")
    region = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION")
    
    logger.info(f"Fetching database URL from AWS Secrets Manager: secret={secret_name}, region={region}, environment={env}")
    
    value = _fetch_secret_from_aws(secret_name=secret_name, region_name=region)
    if value:
        # Try JSON wrapper
        try:
            data = json.loads(value)
            if isinstance(data, dict) and "DATABASE_URL" in data:
                logger.info("Found DATABASE_URL in JSON secret")
                return str(data["DATABASE_URL"]).strip()
        except json.JSONDecodeError:
            # Not JSON, treat as raw DSN
            logger.info("Using raw secret value as DATABASE_URL")
            return value.strip()
        return value.strip()

    # Fallback to env var
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        logger.info("Using DATABASE_URL from environment variable")
        return db_url
    
    logger.error(
        f"DATABASE_URL not found. Environment={env}, "
        f"SecretName={secret_name}, Region={region}, "
        f"EnvVar={'set' if os.getenv('DATABASE_URL') else 'not set'}"
    )
    return None


