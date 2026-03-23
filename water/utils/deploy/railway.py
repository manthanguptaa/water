"""Railway deployment support for Water flows."""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_railway_config(
    app_module: str,
    app_variable: str = "app",
    start_command: Optional[str] = None,
) -> str:
    """
    Generate a railway.toml configuration file.

    Args:
        app_module: Python module containing the FlowServer app.
        app_variable: Variable name of the ASGI app.
        start_command: Custom start command (optional).

    Returns:
        The railway.toml content as a string.
    """
    cmd = start_command or f"uvicorn {app_module}:{app_variable} --host 0.0.0.0 --port ${{PORT:-8000}}"
    return f"""[build]
builder = "nixpacks"

[deploy]
startCommand = "{cmd}"
healthcheckPath = "/health"
healthcheckTimeout = 30
restartPolicyType = "on_failure"
restartPolicyMaxRetries = 3
"""


def cmd_flow_prod_railway(args) -> None:
    """Handle 'water flow prod:railway' command."""
    from water.utils.cli import _find_app_module, _ensure_requirements_txt

    app_module = args.app
    if not app_module:
        app_module = _find_app_module()
        if not app_module:
            logger.error(
                "Could not auto-detect your FlowServer app module. "
                "Use --app <module_name> to specify it."
            )
            sys.exit(1)

    app_variable = args.var or "app"
    logger.info("Detected app: %s:%s", app_module, app_variable)

    _ensure_requirements_txt()

    start_command = getattr(args, "start_command", None)
    config = generate_railway_config(app_module, app_variable, start_command)

    config_path = Path.cwd() / "railway.toml"
    config_path.write_text(config)
    logger.info("Generated %s", config_path)

    if getattr(args, "config_only", False):
        return

    logger.info("To deploy to Railway:")
    logger.info("  1. Install Railway CLI: npm install -g @railway/cli")
    logger.info("  2. Login: railway login")
    logger.info("  3. Deploy: railway up")
    logger.info("Or connect your GitHub repo at https://railway.app")
