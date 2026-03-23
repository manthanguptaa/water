"""Fly.io deployment support for Water flows."""

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_fly_config(
    app_module: str,
    app_variable: str = "app",
    app_name: Optional[str] = None,
    start_command: Optional[str] = None,
    region: str = "iad",
) -> str:
    """
    Generate a fly.toml configuration file.

    Args:
        app_module: Python module containing the FlowServer app.
        app_variable: Variable name of the ASGI app.
        app_name: Fly.io app name.
        start_command: Custom start command (optional).
        region: Deployment region (default: iad).

    Returns:
        The fly.toml content as a string.
    """
    name = app_name or "water-flow-server"
    cmd = start_command or f"uvicorn {app_module}:{app_variable} --host 0.0.0.0 --port 8080"
    return f"""app = "{name}"
primary_region = "{region}"

[build]
  builder = "paketobuildpacks/builder:base"

[env]
  PORT = "8080"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true
  min_machines_running = 0

  [http_service.concurrency]
    type = "connections"
    hard_limit = 25
    soft_limit = 20

[[http_service.checks]]
  grace_period = "10s"
  interval = "30s"
  method = "GET"
  timeout = "5s"
  path = "/health"

[processes]
  app = "{cmd}"
"""


def cmd_flow_prod_fly(args) -> None:
    """Handle 'water flow prod:fly' command."""
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

    app_name = getattr(args, "name", None) or "water-flow-server"
    region = getattr(args, "region", None) or "iad"
    start_command = getattr(args, "start_command", None)
    config = generate_fly_config(app_module, app_variable, app_name, start_command, region)

    config_path = Path.cwd() / "fly.toml"
    config_path.write_text(config)
    logger.info("Generated %s", config_path)

    if getattr(args, "config_only", False):
        return

    logger.info("To deploy to Fly.io:")
    logger.info("  1. Install Fly CLI: curl -L https://fly.io/install.sh | sh")
    logger.info("  2. Login: fly auth login")
    logger.info("  3. Launch: fly launch --name %s", app_name)
    logger.info("  4. Deploy: fly deploy")
