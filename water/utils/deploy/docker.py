"""Docker deployment support for Water flows."""

import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def generate_dockerfile(
    app_module: str,
    app_variable: str = "app",
    python_version: str = "3.11",
    start_command: Optional[str] = None,
) -> str:
    """
    Generate a multi-stage Dockerfile for Water flows.

    Args:
        app_module: Python module containing the FlowServer app.
        app_variable: Variable name of the ASGI app.
        python_version: Python version to use.
        start_command: Custom start command (optional).

    Returns:
        Dockerfile content as a string.
    """
    cmd = start_command or f"uvicorn {app_module}:{app_variable} --host 0.0.0.0 --port 8000"
    return f"""# --- Build stage ---
FROM python:{python_version}-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# --- Runtime stage ---
FROM python:{python_version}-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY . .

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["{cmd.split()[0]}", {', '.join(f'"{a}"' for a in cmd.split()[1:])}]
"""


def generate_docker_compose(
    app_module: str,
    app_variable: str = "app",
    include_redis: bool = False,
    include_postgres: bool = False,
) -> str:
    """
    Generate a docker-compose.yml for Water flows.

    Args:
        app_module: Python module containing the FlowServer app.
        app_variable: Variable name of the ASGI app.
        include_redis: Include Redis service.
        include_postgres: Include PostgreSQL service.

    Returns:
        docker-compose.yml content as a string.
    """
    services = [
        '  app:',
        '    build: .',
        '    ports:',
        '      - "8000:8000"',
        '    environment:',
        '      - PORT=8000',
    ]

    depends = []
    if include_redis:
        services[0:0] = []
        depends.append("redis")
        services.append('      - REDIS_URL=redis://redis:6379')
    if include_postgres:
        depends.append("postgres")
        services.append('      - DATABASE_URL=postgresql://water:water@postgres:5432/water')

    if depends:
        services.append('    depends_on:')
        for dep in depends:
            services.append(f'      - {dep}')

    services.append('    restart: unless-stopped')

    extra_services = []
    if include_redis:
        extra_services.extend([
            '',
            '  redis:',
            '    image: redis:7-alpine',
            '    ports:',
            '      - "6379:6379"',
            '    restart: unless-stopped',
        ])
    if include_postgres:
        extra_services.extend([
            '',
            '  postgres:',
            '    image: postgres:16-alpine',
            '    ports:',
            '      - "5432:5432"',
            '    environment:',
            '      - POSTGRES_USER=water',
            '      - POSTGRES_PASSWORD=water',
            '      - POSTGRES_DB=water',
            '    volumes:',
            '      - pgdata:/var/lib/postgresql/data',
            '    restart: unless-stopped',
        ])

    volumes = []
    if include_postgres:
        volumes = ['', 'volumes:', '  pgdata:']

    all_lines = ['services:'] + services + extra_services + volumes
    return '\n'.join(all_lines) + '\n'


def generate_docker_config(
    app_module: str,
    app_variable: str = "app",
    python_version: str = "3.11",
    start_command: Optional[str] = None,
    include_redis: bool = False,
    include_postgres: bool = False,
) -> dict:
    """Generate both Dockerfile and docker-compose.yml."""
    return {
        "dockerfile": generate_dockerfile(app_module, app_variable, python_version, start_command),
        "compose": generate_docker_compose(app_module, app_variable, include_redis, include_postgres),
    }


def cmd_flow_prod_docker(args) -> None:
    """Handle 'water flow prod:docker' command."""
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

    include_redis = getattr(args, "redis", False)
    include_postgres = getattr(args, "postgres", False)
    start_command = getattr(args, "start_command", None)

    configs = generate_docker_config(
        app_module, app_variable,
        start_command=start_command,
        include_redis=include_redis,
        include_postgres=include_postgres,
    )

    dockerfile_path = Path.cwd() / "Dockerfile"
    dockerfile_path.write_text(configs["dockerfile"])
    logger.info("Generated %s", dockerfile_path)

    compose_path = Path.cwd() / "docker-compose.yml"
    compose_path.write_text(configs["compose"])
    logger.info("Generated %s", compose_path)

    if getattr(args, "config_only", False):
        return

    logger.info("To deploy with Docker: docker compose up --build")
    logger.info("Your Water flows will be available at http://localhost:8000")
