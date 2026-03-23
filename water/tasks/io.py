"""
File I/O tasks for Water.

Read and write files as part of flow execution.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

from pydantic import BaseModel

from water.core.task import Task


class FileInput(BaseModel):
    path: str = ""
    content: str = ""


class FileOutput(BaseModel):
    path: str = ""
    content: str = ""
    success: bool = True


def file_read(
    id: str,
    path: str = "",
    encoding: str = "utf-8",
    parse_json: bool = False,
    description: Optional[str] = None,
) -> Task:
    """
    Create a file read task.

    Args:
        id: Task identifier.
        path: File path template (supports {variable} substitution).
        encoding: File encoding.
        parse_json: If True, parse content as JSON.
        description: Task description.

    Returns:
        A Task instance.
    """
    def execute(params: dict, context: Any) -> dict:
        data = params.get("input_data", params)
        file_path = path.format(**data) if path else data.get("path", "")

        p = Path(file_path)
        if not p.exists():
            return {"path": file_path, "content": "", "success": False, "error": "File not found"}

        content = p.read_text(encoding=encoding)
        result = {"path": file_path, "content": content, "success": True}

        if parse_json:
            try:
                result["json_data"] = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse JSON content from '%s': %s", file_path, e)
                result["error"] = f"JSON parse error: {e}"

        return result

    return Task(
        id=id,
        description=description or f"Read file: {path}",
        input_schema=FileInput,
        output_schema=FileOutput,
        execute=execute,
    )


def file_write(
    id: str,
    path: str = "",
    content: str = "",
    encoding: str = "utf-8",
    description: Optional[str] = None,
) -> Task:
    """
    Create a file write task.

    Args:
        id: Task identifier.
        path: File path template.
        content: Content template.
        encoding: File encoding.
        description: Task description.

    Returns:
        A Task instance.
    """
    def execute(params: dict, context: Any) -> dict:
        data = params.get("input_data", params)
        file_path = path.format(**data) if path else data.get("path", "")
        file_content = content.format(**data) if content else data.get("content", "")

        p = Path(file_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(file_content, encoding=encoding)

        return {"path": file_path, "content": file_content, "success": True}

    return Task(
        id=id,
        description=description or f"Write file: {path}",
        input_schema=FileInput,
        output_schema=FileOutput,
        execute=execute,
    )
