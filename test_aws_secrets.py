#!/usr/bin/env python3
"""Test script to verify AWS Secrets Manager configuration.

Usage:
    ENVIRONMENT=prod AWS_REGION=eu-central-1 uv run python test_aws_secrets.py
"""

import os
import sys

# Set environment before importing modules
os.environ.setdefault("ENVIRONMENT", "prod")
os.environ.setdefault("AWS_REGION", "eu-central-1")
os.environ.setdefault("OPENROUTER_SECRET_NAME", "POZZ")
os.environ.setdefault("POSTGRES_SECRET_NAME", "POZZ")

print("=" * 70)
print("AWS Secrets Manager Configuration Test")
print("=" * 70)
print()

# Check environment
env = os.getenv("ENVIRONMENT", "local")
region = os.getenv("AWS_REGION", "eu-central-1")
print(f"Environment: {env}")
print(f"AWS Region: {region}")
print()

# Check if boto3 is available
try:
    import boto3
    from botocore.exceptions import ClientError
    print("✓ boto3 is available")
except ImportError:
    print("✗ boto3 is not installed. Install it with: uv add boto3")
    sys.exit(1)

# Check AWS credentials
try:
    session = boto3.Session()
    credentials = session.get_credentials()
    if credentials:
        access_key = credentials.access_key
        print(f"✓ AWS credentials found (Access Key: {access_key[:8]}...)")
    else:
        print("✗ No AWS credentials found")
        print("  Set up AWS credentials using one of:")
        print("  - AWS credentials file: ~/.aws/credentials")
        print("  - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
        print("  - IAM role (if running on EC2/Lightsail)")
        sys.exit(1)
except Exception as e:
    print(f"✗ Error checking AWS credentials: {e}")
    sys.exit(1)

print()

# Test Secrets Manager access
secrets_to_check = [
    ("OpenRouter API Key", "OPENROUTER_SECRET_NAME", "med-sim/openrouter"),
    ("PostgreSQL URL", "POSTGRES_SECRET_NAME", "med-sim/postgres"),
]

client = boto3.client("secretsmanager", region_name=region)
all_ok = True

for secret_label, env_var, default_name in secrets_to_check:
    secret_name = os.getenv(env_var, default_name)
    print(f"Checking {secret_label}:")
    print(f"  Secret name: {secret_name}")
    
    try:
        response = client.get_secret_value(SecretId=secret_name)
        secret_value = response.get("SecretString", "")
        
        # Mask sensitive data
        if secret_label == "OpenRouter API Key":
            if secret_value:
                masked = secret_value[:10] + "..." + secret_value[-4:] if len(secret_value) > 14 else "***"
                print(f"  ✓ Secret found (value: {masked})")
            else:
                print(f"  ⚠ Secret found but empty")
        elif secret_label == "PostgreSQL URL":
            if secret_value:
                # Extract just the host for display
                if "@" in secret_value:
                    parts = secret_value.split("@")[1].split("/")
                    host = parts[0] if parts else "unknown"
                    print(f"  ✓ Secret found (host: {host})")
                else:
                    print(f"  ✓ Secret found (length: {len(secret_value)} chars)")
            else:
                print(f"  ⚠ Secret found but empty")
        
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "Unknown")
        all_ok = False
        
        if error_code == "ResourceNotFoundException":
            print(f"  ✗ Secret NOT FOUND in AWS Secrets Manager")
            print(f"     Create it in AWS Console:")
            print(f"     - Go to: https://console.aws.amazon.com/secretsmanager/home?region={region}")
            print(f"     - Create secret with name: {secret_name}")
        elif error_code == "AccessDeniedException":
            print(f"  ✗ ACCESS DENIED to secret")
            print(f"     Check IAM permissions for Secrets Manager")
            print(f"     Required permission: secretsmanager:GetSecretValue")
        else:
            print(f"  ✗ Error: {error_code}")
            print(f"     {e}")
    
    except Exception as e:
        all_ok = False
        print(f"  ✗ Unexpected error: {e}")
    
    print()

# Test using modules.config
print("=" * 70)
print("Testing with modules.config")
print("=" * 70)
print()

try:
    from modules.config import get_openrouter_api_key, get_database_url, get_environment
    
    env_check = get_environment()
    print(f"Environment detected: {env_check}")
    print()
    
    # Test OpenRouter key
    print("Testing get_openrouter_api_key():")
    try:
        key = get_openrouter_api_key()
        if key:
            masked = key[:10] + "..." + key[-4:] if len(key) > 14 else "***"
            print(f"  ✓ Retrieved (value: {masked})")
        else:
            print(f"  ✗ Returned None")
            all_ok = False
    except ValueError as e:
        print(f"  ✗ ValueError: {e}")
        all_ok = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_ok = False
    
    print()
    
    # Test Database URL
    print("Testing get_database_url():")
    try:
        db_url = get_database_url()
        if db_url:
            if "@" in db_url:
                host = db_url.split("@")[1].split("/")[0].split(":")[0]
                print(f"  ✓ Retrieved (host: {host})")
            else:
                print(f"  ✓ Retrieved (length: {len(db_url)} chars)")
        else:
            print(f"  ✗ Returned None")
            all_ok = False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        all_ok = False
    
except ImportError as e:
    print(f"✗ Could not import modules.config: {e}")
    all_ok = False

print()
print("=" * 70)
if all_ok:
    print("✓ All checks passed!")
    sys.exit(0)
else:
    print("✗ Some checks failed. Please fix the issues above.")
    sys.exit(1)

