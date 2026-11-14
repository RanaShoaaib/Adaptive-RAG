import hashlib
import json
import os
import weaviate
from contextlib import contextmanager
from dotenv import load_dotenv
from weaviate.classes.init import Auth, AdditionalConfig, Timeout
from adaptive_rag.config import ROOT_DIR


def _load_env() -> None:
    """Load ROOT_DIR/.env without overriding variables that are already set."""
    load_dotenv(ROOT_DIR / ".env")


def _env_port(name: str, default: int) -> int:
    try:
        port = int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer between 1 and 65535") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{name} must be an integer between 1 and 65535")
    return port


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return True
    if value in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _connection_settings() -> dict[str, str | int | bool]:
    _load_env()
    settings = {}
    for protocol in ("http", "grpc"):
        prefix = f"WEAVIATE_{protocol.upper()}"
        host = os.getenv(f"{prefix}_HOST", "").strip()
        if not host:
            raise ValueError(f"{prefix}_HOST is not set. Add it to your .env file.")
        settings[f"{protocol}_host"] = host
        settings[f"{protocol}_port"] = _env_port(f"{prefix}_PORT", 443)
        settings[f"{protocol}_secure"] = _env_bool(f"{prefix}_SECURE", True)
    return settings


def endpoint_fingerprint() -> str:
    """Identify the configured endpoints without storing hosts or credentials."""
    payload = json.dumps(_connection_settings(), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def get_weaviate_client() -> weaviate.WeaviateClient:
    """
        Create and return an authenticated Weaviate client.

        Returns:
            A connected Weaviate client configured with the project endpoints
            and timeout settings.

        Raises:
            ValueError: If credentials or endpoint settings are missing or invalid.
    """
    _load_env()
    api_key = os.getenv("WEAVIATE_API_KEY")

    if not api_key:
        raise ValueError(
            "WEAVIATE_API_KEY is not set. Add it to your .env file."
        )

    client = weaviate.connect_to_custom(
        auth_credentials=Auth.api_key(api_key),
        additional_config=AdditionalConfig(timeout=Timeout(init=30, query=60, insert=120)),
        **_connection_settings(),
    )

    return client

@contextmanager
def weaviate_session():
    """
        Provide a connected Weaviate client and ensure it is closed afterward.

        Yields:
            A connected Weaviate client.
    """
    client = get_weaviate_client()
    try:
        yield client
    finally:
        client.close()
