import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

logger = logging.getLogger(__name__)


RENDER_API_BASE = "https://api.render.com/v1"


def _import_flow(spec: str):
    """
    Import a Flow object from a 'module:var' specification.

    Args:
        spec: A string in the format "module:variable_name"
              (e.g., "cookbook.sequential_flow:registration_flow")

    Returns:
        The Flow object referenced by the spec.

    Raises:
        ValueError: If the spec format is invalid.
        ImportError: If the module cannot be imported.
        AttributeError: If the variable is not found in the module.
        TypeError: If the variable is not a Flow instance.
    """
    from water.core.flow import Flow

    if ":" not in spec:
        raise ValueError(
            f"Invalid spec '{spec}'. Expected format 'module:variable' "
            f"(e.g., 'cookbook.sequential_flow:registration_flow')"
        )

    module_path, var_name = spec.rsplit(":", 1)
    module = importlib.import_module(module_path)
    obj = getattr(module, var_name)

    if not isinstance(obj, Flow):
        raise TypeError(
            f"'{var_name}' in '{module_path}' is not a Flow instance "
            f"(got {type(obj).__name__})"
        )

    return obj


def _find_flows_in_module(module_path: str):
    """
    Import a module and find all Flow instances defined in it.

    Args:
        module_path: Dotted module path (e.g., "cookbook.sequential_flow").

    Returns:
        List of (variable_name, flow_object) tuples.
    """
    from water.core.flow import Flow

    module = importlib.import_module(module_path)
    flows = []
    for name in dir(module):
        obj = getattr(module, name)
        if isinstance(obj, Flow):
            flows.append((name, obj))
    return flows


def cmd_run(args):
    """Handle 'water run <module:flow_var>' command."""
    try:
        flow = _import_flow(args.flow)
    except (ValueError, ImportError, AttributeError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    input_data = {}
    if args.input:
        try:
            input_data = json.loads(args.input)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        result = asyncio.run(flow.run(input_data))
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"Error running flow: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_visualize(args):
    """Handle 'water visualize <module:flow_var>' command."""
    try:
        flow = _import_flow(args.flow)
    except (ValueError, ImportError, AttributeError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        diagram = flow.visualize()
    except Exception as e:
        print(f"Error generating visualization: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(diagram + "\n")
        print(f"Diagram saved to {output_path}")
    else:
        print(diagram)


def cmd_dry_run(args):
    """Handle 'water dry-run <module:flow_var>' command."""
    try:
        flow = _import_flow(args.flow)
    except (ValueError, ImportError, AttributeError, TypeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    input_data = {}
    if args.input:
        try:
            input_data = json.loads(args.input)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON input: {e}", file=sys.stderr)
            sys.exit(1)

    try:
        report = asyncio.run(flow.dry_run(input_data))
        print(json.dumps(report, indent=2, default=str))
    except Exception as e:
        print(f"Error during dry run: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_list(args):
    """Handle 'water list <module>' command."""
    try:
        flows = _find_flows_in_module(args.module)
    except ImportError as e:
        print(f"Error: Could not import module '{args.module}': {e}", file=sys.stderr)
        sys.exit(1)

    if not flows:
        print(f"No Flow instances found in '{args.module}'.")
        return

    # Print table header
    header = f"{'Variable':<30} {'Flow ID':<25} {'Description':<35} {'Tasks':<6} {'Version':<10}"
    print(header)
    print("-" * len(header))

    for var_name, flow in flows:
        task_count = len(flow._tasks)
        version = flow.version or "-"
        description = flow.description or "-"
        # Truncate long descriptions
        if len(description) > 33:
            description = description[:30] + "..."
        print(f"{var_name:<30} {flow.id:<25} {description:<35} {task_count:<6} {version:<10}")


def _find_app_module():
    """Auto-detect the module containing a FlowServer .get_app() call."""
    cwd = Path.cwd()
    for py_file in cwd.glob("*.py"):
        try:
            content = py_file.read_text()
            if "FlowServer" in content and "get_app()" in content:
                return py_file.stem
        except Exception:
            logger.warning("Failed to read file '%s' while searching for FlowServer", py_file, exc_info=True)
            continue
    return None


def _render_api_request(path, method="GET", data=None, api_key=None):
    """Make a request to the Render API."""
    url = f"{RENDER_API_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(url, data=body, headers=headers, method=method)

    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode()
        print(f"Error: Render API returned {e.code}: {error_body}", file=sys.stderr)
        sys.exit(1)


def _get_repo_url():
    """Get the git remote origin URL."""
    import subprocess

    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True,
        )
        url = result.stdout.strip()
        # Convert SSH URL to HTTPS if needed
        if url.startswith("git@github.com:"):
            url = url.replace("git@github.com:", "https://github.com/")
        if url.endswith(".git"):
            url = url[:-4]
        return url
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _generate_render_yaml(app_module, app_variable="app", start_command=None):
    """Generate a render.yaml blueprint file."""
    if start_command is None:
        start_command = f"uvicorn {app_module}:{app_variable} --host 0.0.0.0 --port $PORT"

    render_config = {
        "services": [
            {
                "type": "web",
                "name": "water-flow-server",
                "runtime": "python",
                "buildCommand": "pip install -r requirements.txt",
                "startCommand": start_command,
                "envVars": [
                    {"key": "PYTHON_VERSION", "value": "3.11.6"},
                ],
            }
        ]
    }

    return render_config


def _ensure_requirements_txt():
    """Ensure requirements.txt exists with water-ai dependency."""
    req_path = Path.cwd() / "requirements.txt"
    if req_path.exists():
        content = req_path.read_text()
        if "water-ai" not in content:
            print("Warning: requirements.txt exists but doesn't include 'water-ai'.")
            print("  Add 'water-ai' to your requirements.txt for deployment.")
        return

    # Create a basic requirements.txt
    req_path.write_text("water-ai\n")
    print("Created requirements.txt with water-ai dependency.")


def cmd_flow_prod_render(args):
    """Handle 'water flow prod:render' command."""
    api_key = os.environ.get("RENDER_API_KEY")

    # Determine the app module
    app_module = args.app
    if not app_module:
        app_module = _find_app_module()
        if not app_module:
            print(
                "Error: Could not auto-detect your FlowServer app module.",
                file=sys.stderr,
            )
            print(
                "  Use --app <module_name> to specify it (e.g., --app playground).",
                file=sys.stderr,
            )
            sys.exit(1)

    app_variable = args.var or "app"
    print(f"Detected app: {app_module}:{app_variable}")

    # Ensure requirements.txt
    _ensure_requirements_txt()

    # Generate render.yaml
    start_command = args.start_command
    render_config = _generate_render_yaml(app_module, app_variable, start_command)

    render_yaml_path = Path.cwd() / "render.yaml"

    # Write render.yaml as YAML-like format (simple enough to avoid PyYAML dependency)
    _write_render_yaml(render_yaml_path, render_config)
    print(f"Generated {render_yaml_path}")

    if not api_key:
        print()
        print("No RENDER_API_KEY found. To deploy automatically:")
        print("  1. Get your API key from https://dashboard.render.com/settings#api-keys")
        print("  2. Set it: export RENDER_API_KEY=<your-key>")
        print("  3. Re-run: water flow prod:render")
        print()
        print("Or deploy manually:")
        print("  1. Push your code (with render.yaml) to GitHub")
        print("  2. Go to https://dashboard.render.com/select-repo?type=blueprint")
        print("  3. Connect your repo and deploy")
        return

    # Deploy via Render API
    repo_url = _get_repo_url()
    if not repo_url:
        print(
            "Error: Could not detect git remote URL. Ensure you're in a git repo with an origin remote.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Deploying from repo: {repo_url}")

    service_name = args.name or "water-flow-server"
    start_cmd = start_command or f"uvicorn {app_module}:{app_variable} --host 0.0.0.0 --port $PORT"

    service_payload = {
        "type": "web_service",
        "name": service_name,
        "repo": repo_url,
        "autoDeploy": "yes",
        "branch": args.branch or "main",
        "runtime": "python",
        "buildCommand": "pip install -r requirements.txt",
        "startCommand": start_cmd,
        "plan": args.plan or "free",
        "region": args.region or "oregon",
        "envVars": [
            {"key": "PYTHON_VERSION", "value": "3.11.6"},
        ],
    }

    print("Creating Render web service...")
    result = _render_api_request("/services", method="POST", data=service_payload, api_key=api_key)

    service = result.get("service", result)
    service_id = service.get("id", "unknown")
    service_url = service.get("serviceDetails", {}).get("url", "")

    print()
    print("Deployment initiated!")
    print(f"  Service ID: {service_id}")
    if service_url:
        print(f"  URL: {service_url}")
    print(f"  Dashboard: https://dashboard.render.com")
    print()
    print("Your Water flows will be available once the build completes.")


def _write_render_yaml(path, config):
    """Write render.yaml without requiring PyYAML."""
    lines = []
    lines.append("services:")
    for svc in config["services"]:
        lines.append(f"  - type: {svc['type']}")
        lines.append(f"    name: {svc['name']}")
        lines.append(f"    runtime: {svc['runtime']}")
        lines.append(f"    buildCommand: \"{svc['buildCommand']}\"")
        lines.append(f"    startCommand: \"{svc['startCommand']}\"")
        if svc.get("envVars"):
            lines.append("    envVars:")
            for env in svc["envVars"]:
                lines.append(f"      - key: {env['key']}")
                lines.append(f"        value: {env['value']}")

    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        prog="water",
        description="Water - Multi-agent orchestration framework CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # water run <module:flow_var> --input '{...}'
    run_parser = subparsers.add_parser(
        "run",
        help="Run a flow from the terminal",
    )
    run_parser.add_argument(
        "flow",
        help="Flow spec in 'module:variable' format (e.g., 'cookbook.sequential_flow:registration_flow')",
    )
    run_parser.add_argument(
        "--input",
        default=None,
        help="Input data as a JSON string (e.g., '{\"key\": \"value\"}')",
    )

    # water visualize <module:flow_var> [--output file.md]
    viz_parser = subparsers.add_parser(
        "visualize",
        help="Generate a Mermaid diagram of a flow",
    )
    viz_parser.add_argument(
        "flow",
        help="Flow spec in 'module:variable' format",
    )
    viz_parser.add_argument(
        "--output",
        default=None,
        help="Save diagram to a file instead of printing to stdout",
    )

    # water dry-run <module:flow_var> --input '{...}'
    dryrun_parser = subparsers.add_parser(
        "dry-run",
        help="Validate a flow without executing tasks",
    )
    dryrun_parser.add_argument(
        "flow",
        help="Flow spec in 'module:variable' format",
    )
    dryrun_parser.add_argument(
        "--input",
        default=None,
        help="Input data as a JSON string",
    )

    # water list <module>
    list_parser = subparsers.add_parser(
        "list",
        help="List all Flow instances in a module",
    )
    list_parser.add_argument(
        "module",
        help="Dotted module path (e.g., 'cookbook.sequential_flow')",
    )

    # water eval
    eval_parser = subparsers.add_parser("eval", help="Evaluation commands")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", help="Eval subcommands")

    # water eval run <config>
    eval_run_parser = eval_subparsers.add_parser(
        "run",
        help="Run an eval suite from a config file",
    )
    eval_run_parser.add_argument(
        "config",
        help="Path to eval config file (YAML or JSON)",
    )
    eval_run_parser.add_argument(
        "--flow",
        default=None,
        help="Override flow spec (module:variable format)",
    )
    eval_run_parser.add_argument(
        "--output",
        default=None,
        help="Save report to file instead of printing",
    )
    eval_run_parser.add_argument(
        "--format",
        default="text",
        choices=["text", "json"],
        help="Output format (default: text)",
    )

    # water eval compare <baseline> <current>
    eval_compare_parser = eval_subparsers.add_parser(
        "compare",
        help="Compare two eval reports for regressions",
    )
    eval_compare_parser.add_argument(
        "baseline",
        help="Path to baseline eval report (JSON)",
    )
    eval_compare_parser.add_argument(
        "current",
        help="Path to current eval report (JSON)",
    )

    # water eval list [directory]
    eval_list_parser = eval_subparsers.add_parser(
        "list",
        help="List eval config files in a directory",
    )
    eval_list_parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to search (default: current directory)",
    )

    # water flow
    flow_parser = subparsers.add_parser("flow", help="Flow management commands")
    flow_subparsers = flow_parser.add_subparsers(dest="flow_command", help="Flow subcommands")

    # water flow prod:render
    render_parser = flow_subparsers.add_parser(
        "prod:render",
        help="Deploy flows to Render",
    )
    render_parser.add_argument(
        "--app",
        help="Python module containing FlowServer app (e.g., 'playground')",
    )
    render_parser.add_argument(
        "--var",
        default="app",
        help="Variable name of the ASGI app (default: 'app')",
    )
    render_parser.add_argument(
        "--name",
        help="Render service name (default: 'water-flow-server')",
    )
    render_parser.add_argument(
        "--branch",
        default="main",
        help="Git branch to deploy (default: 'main')",
    )
    render_parser.add_argument(
        "--plan",
        default="free",
        help="Render plan type (default: 'free')",
    )
    render_parser.add_argument(
        "--region",
        default="oregon",
        help="Render region (default: 'oregon')",
    )
    render_parser.add_argument(
        "--start-command",
        help="Custom start command (overrides auto-detected)",
    )

    # water flow prod:railway
    railway_parser = flow_subparsers.add_parser(
        "prod:railway",
        help="Deploy flows to Railway",
    )
    railway_parser.add_argument("--app", help="Python module containing FlowServer app")
    railway_parser.add_argument("--var", default="app", help="Variable name of the ASGI app")
    railway_parser.add_argument("--start-command", help="Custom start command")
    railway_parser.add_argument("--config-only", action="store_true", help="Generate config only, no deploy")

    # water flow prod:fly
    fly_parser = flow_subparsers.add_parser(
        "prod:fly",
        help="Deploy flows to Fly.io",
    )
    fly_parser.add_argument("--app", help="Python module containing FlowServer app")
    fly_parser.add_argument("--var", default="app", help="Variable name of the ASGI app")
    fly_parser.add_argument("--name", help="Fly.io app name")
    fly_parser.add_argument("--region", default="iad", help="Fly.io region (default: iad)")
    fly_parser.add_argument("--start-command", help="Custom start command")
    fly_parser.add_argument("--config-only", action="store_true", help="Generate config only, no deploy")

    # water flow prod:docker
    docker_parser = flow_subparsers.add_parser(
        "prod:docker",
        help="Generate Dockerfile + docker-compose.yml",
    )
    docker_parser.add_argument("--app", help="Python module containing FlowServer app")
    docker_parser.add_argument("--var", default="app", help="Variable name of the ASGI app")
    docker_parser.add_argument("--start-command", help="Custom start command")
    docker_parser.add_argument("--redis", action="store_true", help="Include Redis in docker-compose")
    docker_parser.add_argument("--postgres", action="store_true", help="Include PostgreSQL in docker-compose")
    docker_parser.add_argument("--config-only", action="store_true", help="Generate config only")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        cmd_run(args)
    elif args.command == "visualize":
        cmd_visualize(args)
    elif args.command == "dry-run":
        cmd_dry_run(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "eval":
        from water.eval.cli import cmd_eval_run, cmd_eval_compare, cmd_eval_list
        if args.eval_command == "run":
            cmd_eval_run(args)
        elif args.eval_command == "compare":
            cmd_eval_compare(args)
        elif args.eval_command == "list":
            cmd_eval_list(args)
        else:
            eval_parser.print_help()
            sys.exit(0)
    elif args.command == "flow":
        if args.flow_command == "prod:render":
            cmd_flow_prod_render(args)
        elif args.flow_command == "prod:railway":
            from water.utils.deploy.railway import cmd_flow_prod_railway
            cmd_flow_prod_railway(args)
        elif args.flow_command == "prod:fly":
            from water.utils.deploy.fly import cmd_flow_prod_fly
            cmd_flow_prod_fly(args)
        elif args.flow_command == "prod:docker":
            from water.utils.deploy.docker import cmd_flow_prod_docker
            cmd_flow_prod_docker(args)
        else:
            flow_parser.print_help()
            sys.exit(0)


if __name__ == "__main__":
    main()
