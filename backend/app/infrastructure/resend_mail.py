from html import escape

import httpx

from app.settings import get_settings


def notify_pending_account(account_email: str | None) -> None:
    settings = get_settings()
    if not settings.resend_api_key:
        raise RuntimeError("RESEND_API_KEY is not configured")
    response = httpx.post(
        "https://api.resend.com/emails",
        headers={"Authorization": f"Bearer {settings.resend_api_key}"},
        json={
            "from": settings.resend_from,
            "to": [settings.resend_mail],
            "subject": "POZZ: nowe konto do akceptacji",
            "text": f"Nowe konto: {account_email or 'nieznany adres'}\n\nOtwórz panel administratora: https://pozz.fmazurkiewicz.dev/menu/admin",
            "html": f"""<p>Nowe konto oczekuje na akceptację w <strong>POZZ</strong>.</p>
<p><strong>Konto:</strong> {escape(account_email or 'nieznany adres')}</p>
<p><a href=\"https://pozz.fmazurkiewicz.dev/menu/admin\">Otwórz panel administratora</a></p>""",
        },
        timeout=5,
    )
    response.raise_for_status()
