"""
Notification tasks for Water.

Webhook, email, and messaging notifications.
"""

import json
import logging
import urllib.request
import urllib.error
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel

from water.core.task import Task


class WebhookInput(BaseModel):
    url: str = ""
    data: Dict[str, Any] = {}


class WebhookOutput(BaseModel):
    status_code: int = 0
    success: bool = True


def webhook_task(
    id: str,
    url: str = "",
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 10.0,
    description: Optional[str] = None,
) -> Task:
    """
    Create a webhook notification task (POST).

    Sends the entire input data as JSON to the specified URL.

    Args:
        id: Task identifier.
        url: Webhook URL template.
        headers: Additional request headers.
        timeout: Request timeout in seconds.
        description: Task description.

    Returns:
        A Task instance.
    """
    default_headers = {"Content-Type": "application/json"}
    if headers:
        default_headers.update(headers)

    def execute(params: dict, context: Any) -> dict:
        data = params.get("input_data", params)
        webhook_url = url.format(**data) if url else data.get("url", "")

        if not webhook_url:
            return {"status_code": 0, "success": False, "error": "No URL provided"}

        body = json.dumps(data).encode()
        req = urllib.request.Request(
            webhook_url,
            data=body,
            headers=default_headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return {"status_code": resp.status, "success": True}
        except urllib.error.HTTPError as e:
            logger.warning("Webhook POST to '%s' returned HTTP %d: %s", webhook_url, e.code, e)
            return {"status_code": e.code, "success": False, "error": str(e)}
        except urllib.error.URLError as e:
            logger.warning("Webhook POST to '%s' failed with URL error: %s", webhook_url, e)
            return {"status_code": 0, "success": False, "error": str(e)}

    return Task(
        id=id,
        description=description or f"Webhook POST: {url}",
        input_schema=WebhookInput,
        output_schema=WebhookOutput,
        execute=execute,
    )
