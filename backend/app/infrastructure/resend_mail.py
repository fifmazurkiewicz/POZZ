import httpx

from app.settings import get_settings


def notify_pending_account(account_email: str | None) -> None:
    settings = get_settings()
    if not settings.resend_api_key:
        raise RuntimeError("RESEND_API_KEY is not configured")
    response = httpx.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {settings.resend_api_key}"},
        json={"from": settings.resend_from, "to": [settings.resend_mail], "subject": "POZZ: new account awaiting approval", "text": f"Account: {account_email or 'unknown'}\nOpen: /menu/admin"},
        timeout=5,
    )
    response.raise_for_status()
