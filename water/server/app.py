import logging
from typing import List, Dict, Any, Optional, Type
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

from water.core.flow import Flow
from water.observability.dashboard import FlowDashboard


class RunFlowRequest(BaseModel):
    """Request model for flow execution."""
    input_data: Dict[str, Any]

class RunFlowResponse(BaseModel):
    """Response model for flow execution results."""
    flow_id: str
    status: str
    result: Dict[str, Any]
    execution_time_ms: float
    timestamp: datetime

class TaskInfo(BaseModel):
    """Information about a task within a flow."""
    id: str
    description: str
    type: str  # "sequential", "parallel", "branch", "loop"
    input_schema: Optional[Dict[str, str]] = None
    output_schema: Optional[Dict[str, str]] = None

class FlowSummary(BaseModel):
    """Summary information about a flow."""
    id: str
    description: str
    tasks: List[TaskInfo]

class FlowDetail(BaseModel):
    """Detailed information about a flow."""
    id: str
    description: str
    metadata: Dict[str, Any]
    tasks: List[TaskInfo]

class FlowsListResponse(BaseModel):
    """Response model for listing flows."""
    flows: List[FlowSummary]

class FlowServer:
    """
    FastAPI server for hosting Water flows.

    Provides REST endpoints for discovering and executing flows.

    Example:
        flows = [flow1, flow2, flow3]
        app = FlowServer(flows=flows).get_app()

        if __name__ == "__main__":
            import uvicorn
            uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
    """

    def __init__(self, flows: List[Flow], storage: Optional[Any] = None) -> None:
        """
        Initialize FlowServer with a list of flows.

        Args:
            flows: List of registered Flow instances
            storage: Optional storage backend for dashboard functionality

        Raises:
            ValueError: If flows contain duplicates or unregistered flows
        """
        self.flows: Dict[str, Flow] = {}
        for flow in flows:
            if flow.id in self.flows:
                raise ValueError(f"Duplicate flow ID: {flow.id}")
            if not flow._registered:
                raise ValueError(f"Flow {flow.id} must be registered before adding to server")
            self.flows[flow.id] = flow

        self.dashboard = FlowDashboard(storage) if storage else None

    def _serialize_schema(self, schema_class: Type[BaseModel]) -> Optional[Dict[str, str]]:
        """
        Convert Pydantic model to a simple field:type mapping.

        Args:
            schema_class: Pydantic BaseModel class

        Returns:
            Dictionary mapping field names to simplified type strings, or None
        """
        if not schema_class:
            return None

        try:
            schema_dict = {}
            for field_name, field_info in schema_class.model_fields.items():
                # Simple type mapping - much simpler than before
                field_type = field_info.annotation
                type_name = getattr(field_type, '__name__', str(field_type))

                # Basic type cleanup
                if 'int' in type_name.lower():
                    schema_dict[field_name] = "int"
                elif 'float' in type_name.lower():
                    schema_dict[field_name] = "float"
                elif 'str' in type_name.lower():
                    schema_dict[field_name] = "string"
                elif 'bool' in type_name.lower():
                    schema_dict[field_name] = "boolean"
                elif 'list' in type_name.lower():
                    schema_dict[field_name] = "array"
                elif 'dict' in type_name.lower():
                    schema_dict[field_name] = "object"
                else:
                    schema_dict[field_name] = type_name

            return schema_dict

        except Exception:
            logger.warning("Failed to parse schema for input model", exc_info=True)
            return {"error": "Could not parse schema"}

    def _extract_task_info(self, execution_nodes: List[Dict[str, Any]]) -> List[TaskInfo]:
        """
        Extract task information from execution nodes.

        Args:
            execution_nodes: List of execution node dictionaries

        Returns:
            List of TaskInfo objects
        """
        task_infos = []

        for node in execution_nodes:
            node_type = node["type"]

            if node_type == "sequential":
                task = node["task"]
                task_infos.append(TaskInfo(
                    id=task.id,
                    description=task.description,
                    type="sequential",
                    input_schema=self._serialize_schema(task.input_schema),
                    output_schema=self._serialize_schema(task.output_schema)
                ))

            elif node_type == "parallel":
                for task in node["tasks"]:
                    task_infos.append(TaskInfo(
                        id=task.id,
                        description=task.description,
                        type="parallel",
                        input_schema=self._serialize_schema(task.input_schema),
                        output_schema=self._serialize_schema(task.output_schema)
                    ))

            elif node_type == "branch":
                for branch in node["branches"]:
                    task = branch["task"]
                    task_infos.append(TaskInfo(
                        id=task.id,
                        description=task.description,
                        type="branch",
                        input_schema=self._serialize_schema(task.input_schema),
                        output_schema=self._serialize_schema(task.output_schema)
                    ))

            elif node_type == "loop":
                task = node["task"]
                task_infos.append(TaskInfo(
                    id=task.id,
                    description=task.description,
                    type="loop",
                    input_schema=self._serialize_schema(task.input_schema),
                    output_schema=self._serialize_schema(task.output_schema)
                ))

        return task_infos

    def get_app(self) -> FastAPI:
        """
        Create and configure the FastAPI application.

        Returns:
            Configured FastAPI application instance
        """
        app = FastAPI(
            title="Water Flows API",
            description="REST API for executing Water framework workflows",
            version="1.0.0"
        )

        # Add CORS middleware for development
        from fastapi.middleware.cors import CORSMiddleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "flows_count": len(self.flows),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

        @app.get("/flows", response_model=FlowsListResponse)
        async def list_flows():
            """Get list of all available flows."""
            flows_summary = []
            for flow in self.flows.values():
                task_infos = self._extract_task_info(flow._tasks)
                flows_summary.append(FlowSummary(
                    id=flow.id,
                    description=flow.description,
                    tasks=task_infos,
                ))

            return FlowsListResponse(flows=flows_summary)

        @app.get("/flows/{flow_id}", response_model=FlowDetail)
        async def get_flow_details(flow_id: str):
            """Get detailed information about a specific flow."""
            if flow_id not in self.flows:
                raise HTTPException(status_code=404, detail=f"Flow '{flow_id}' not found")

            flow = self.flows[flow_id]
            task_infos = self._extract_task_info(flow._tasks)

            return FlowDetail(
                id=flow.id,
                description=flow.description,
                metadata=flow.metadata,
                tasks=task_infos,
            )

        @app.post("/flows/{flow_id}/run", response_model=RunFlowResponse)
        async def run_flow(flow_id: str, request: RunFlowRequest):
            """Execute a specific flow with input data."""
            if flow_id not in self.flows:
                raise HTTPException(status_code=404, detail=f"Flow '{flow_id}' not found")

            flow = self.flows[flow_id]

            try:
                start_time = datetime.now(timezone.utc)
                result = await flow.run(request.input_data)
                end_time = datetime.now(timezone.utc)

                execution_time_ms = round((end_time - start_time).total_seconds() * 1000, 4)

                return RunFlowResponse(
                    flow_id=flow_id,
                    status="success",
                    result=result,
                    execution_time_ms=execution_time_ms,
                    timestamp=end_time
                )

            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail={
                        "error": str(e),
                        "flow_id": flow_id
                    }
                )

        # Dashboard routes (only if storage was provided)
        if self.dashboard:
            @app.get("/dashboard", response_class=HTMLResponse)
            async def serve_dashboard():
                return self.dashboard.get_spa_html()

            @app.get("/api/dashboard/stats")
            async def dashboard_stats():
                return await self.dashboard.get_stats()

            @app.get("/api/dashboard/sessions")
            async def dashboard_sessions(flow_id: str = None, limit: int = 50, offset: int = 0):
                return await self.dashboard.get_sessions_list(flow_id=flow_id, limit=limit, offset=offset)

            @app.get("/api/dashboard/sessions/{execution_id}")
            async def dashboard_session_detail(execution_id: str):
                detail = await self.dashboard.get_session_detail(execution_id)
                if detail is None:
                    raise HTTPException(status_code=404, detail="Session not found")
                return detail

            @app.get("/api/dashboard/flows")
            async def dashboard_flows():
                return self.dashboard.get_flows_summary(self.flows)

        return app
