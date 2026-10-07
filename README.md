# Water

**The production-ready agent harness framework for Python.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI](https://img.shields.io/pypi/v/water-ai)](https://pypi.org/project/water-ai/)
[![Python](https://img.shields.io/pypi/pyversions/water-ai)](https://pypi.org/project/water-ai/)

## Overview

Water is an agent harness framework — it provides the infrastructure *around* your AI agents, not the agents themselves. Orchestration, resilience, observability, guardrails, approval gates, sandboxing, and deployment tooling so you can focus on what your agents actually do.

Works with any agent framework: LangChain, CrewAI, Agno, OpenAI, Anthropic, or your own custom agents.

## Installation

```bash
pip install water-ai
```

## Quick Start

```python
import asyncio
from water import Flow, create_task
from pydantic import BaseModel

class NumberInput(BaseModel):
    value: int

class NumberOutput(BaseModel):
    result: int

def add_five(params, context):
    return {"result": params["input_data"]["value"] + 5}

task = create_task(
    id="add",
    description="Add five",
    input_schema=NumberInput,
    output_schema=NumberOutput,
    execute=add_five,
)

flow = Flow(id="math", description="Math flow").then(task).register()

async def main():
    result = await flow.run({"value": 10})
    print(result)  # {"result": 15}

asyncio.run(main())
```

## Flow Patterns

Water supports composable flow patterns that chain together with a fluent API:

```python
flow = Flow(id="pipeline", description="Example pipeline")

# Sequential — tasks run one after another
flow.then(task_a).then(task_b).then(task_c)

# Parallel — tasks run concurrently, results are merged
flow.parallel([task_a, task_b, task_c])

# Conditional branching — route to different tasks based on data
flow.branch([
    (lambda data: data["type"] == "email", email_task),
    (lambda data: data["type"] == "sms", sms_task),
])

# Loop — repeat a task while a condition holds
flow.loop(lambda data: data["retries"] < 3, retry_task, max_iterations=5)

# Map — run a task for each item in a list (parallel)
flow.map(process_task, over="items")

# DAG — define tasks with explicit dependencies
flow.dag(
    [task_a, task_b, task_c],
    dependencies={"task_c": ["task_a", "task_b"]},
)

# SubFlow composition — nest flows with input/output mapping
from water import SubFlow, compose_flows
sub = SubFlow(inner_flow, input_mapping={"text": "raw_input"}, output_mapping={"clean": "text"})
flow.then(sub.as_task())

# Compose multiple flows sequentially
pipeline = compose_flows(flow_a, flow_b, flow_c, id="full_pipeline")

# Try-catch-finally — structured error handling
flow.try_catch(
    try_tasks=[risky_task, process_task],
    catch_task=error_handler,
    finally_task=cleanup_task,
)

# Conditional execution & fallbacks
flow.then(task, when=lambda data: data["enabled"])
flow.then(task, fallback=fallback_task)
```

## Agent Harness

Water provides infrastructure around your AI agents — not the agents themselves.

### LLM Tasks

Use any LLM provider through a unified interface:

```python
from water.agents import create_agent_task, OpenAIProvider, AnthropicProvider

agent = create_agent_task(
    id="writer",
    description="Write copy",
    prompt_template="Write about: {topic}",
    provider_instance=OpenAIProvider(model="gpt-4o"),
    system_prompt="You are a copywriter.",
)
```

### Streaming LLM Agents

Stream responses token-by-token with real-time callbacks:

```python
from water.agents import create_streaming_agent_task, OpenAIStreamProvider

agent = create_streaming_agent_task(
    id="stream_writer",
    prompt_template="Write about: {topic}",
    provider_instance=OpenAIStreamProvider(model="gpt-4o"),
    on_chunk=lambda chunk: print(chunk.delta, end="", flush=True),
)
```

### Multi-Agent Orchestration

Coordinate multiple agents with shared context:

```python
from water.agents import create_agent_team, AgentRole

team = create_agent_team(
    team_id="research_team",
    roles=[
        AgentRole(id="researcher", provider=OpenAIProvider(model="gpt-4o"),
                  system_prompt="Research the topic thoroughly."),
        AgentRole(id="writer", provider=AnthropicProvider(model="claude-sonnet-4-20250514"),
                  system_prompt="Write a clear article."),
    ],
    strategy="sequential",  # or "round_robin", "dynamic"
)
```

### Tool Use

Give agents the ability to call tools:

```python
from water.agents import Tool, Toolkit, ToolExecutor

search = Tool(name="search", description="Search the web",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}},
    execute=lambda args: {"results": search_web(args["query"])})

executor = ToolExecutor(provider=OpenAIProvider(model="gpt-4o"),
    tools=Toolkit(name="tools", tools=[search]), max_rounds=5)
result = await executor.run(messages=[{"role": "user", "content": "Search for AI news"}])
```

### Prompt Templates

Reusable templates with variable interpolation and composition:

```python
from water.agents import PromptTemplate, PromptLibrary

template = PromptTemplate("You are a {{role}}. {{action}}: {{content}}", defaults={"role": "assistant"})
result = template.render(action="Summarize", content="...")

library = PromptLibrary()
library.register("system", "You are a {{role}}.")
library.register("task", "{{action}}: {{input}}")
combined = library.compose("system", "task", separator="\n\n")
```

### Fallback Chains

Automatically failover between LLM providers:

```python
from water.agents import FallbackChain

chain = FallbackChain(
    providers=[OpenAIProvider(model="gpt-4o"), AnthropicProvider(model="claude-sonnet-4-20250514")],
    strategy="first_success",  # also: "round_robin", "lowest_latency"
)
```

### Batch Processing

Process many inputs concurrently with controlled parallelism:

```python
from water.agents import BatchProcessor

processor = BatchProcessor(max_concurrency=5, retry_failed=True, max_retries=2)
result = await processor.run_batch(task=summarize_task, inputs=[{"text": doc} for doc in docs])
print(f"Success rate: {result.success_rate:.0%}")
```

### Dynamic Planning

Let an LLM decompose goals into steps and execute them:

```python
from water.agents import PlannerAgent, TaskRegistry

registry = TaskRegistry()
registry.register("search", search_task, "Search the web")
registry.register("summarize", summarize_task, "Summarize text")

planner = PlannerAgent(provider=OpenAIProvider(model="gpt-4o"), task_registry=registry)
result = await planner.plan_and_execute("Find and summarize AI news")
```

### Context Management

Manage conversation context windows with automatic truncation:

```python
from water.agents import ContextManager, TruncationStrategy

ctx = ContextManager(max_tokens=4096, strategy=TruncationStrategy.SLIDING_WINDOW, reserve_tokens=500)
trimmed = ctx.prepare_messages(long_conversation)
```

### Approval Gates

Add human approval checkpoints for high-risk operations:

```python
from water.agents import create_approval_task, ApprovalGate, ApprovalPolicy, RiskLevel

gate = ApprovalGate(policy=ApprovalPolicy(auto_approve_below=RiskLevel.MEDIUM, timeout=300.0))
approval = create_approval_task(id="prod_gate", action_description="Deploy to production",
    risk_level=RiskLevel.CRITICAL, gate=gate)
```

### Sandboxed Execution

Run untrusted code in isolated environments:

```python
from water.agents import create_sandboxed_task, SandboxConfig, SubprocessSandbox

sandboxed = create_sandboxed_task(
    id="run_code",
    sandbox=SubprocessSandbox(),  # also: InMemorySandbox(), DockerSandbox()
    config=SandboxConfig(timeout=10.0, max_memory_mb=128),
)
```

### Agentic Loop (ReAct)

The model controls the loop. `create_agentic_task` runs a Think-Act-Observe-Repeat cycle where the LLM decides which tools to call and when to stop:

```python
from water.agents import create_agentic_task, Tool

search = Tool(name="search", description="Search the web",
    input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    execute=lambda query: f"Results for {query}")

agent = create_agentic_task(
    id="researcher",
    provider=OpenAIProvider(model="gpt-4o"),
    tools=[search],
    system_prompt="You are a research assistant.",
    max_iterations=10,
    stop_tool=True,  # Inject __done__ tool for explicit stop signaling
    on_step=lambda i, step: print(f"Step {i}: {step['think'][:50]}"),
    on_tool_call=lambda name, args: False if name == "dangerous" else True,
    stop_condition=lambda steps, history: len(history) >= 5,
)
```

### Sub-Agent Isolation

Create child agents that run their own isolated ReAct loops with separate context windows:

```python
from water.agents import SubAgentConfig, create_sub_agent_tool

researcher = create_sub_agent_tool(SubAgentConfig(
    id="researcher",
    provider=OpenAIProvider(model="gpt-4o"),
    tools=[search_tool, read_file_tool],
    system_prompt="You are a research specialist.",
    max_iterations=5,
))

# Parent agent uses the sub-agent as a regular tool
parent = create_agentic_task(
    id="orchestrator", provider=provider,
    tools=[researcher, write_tool, test_tool],
    system_prompt="Delegate research to your researcher.",
)
```

### Layered Memory

Priority-ordered memory (ORG > PROJECT > USER > SESSION > AUTO_LEARNED) with automatic resolution:

```python
from water.agents import MemoryManager, MemoryLayer, create_memory_tools

memory = MemoryManager()
await memory.add("timeout", "30s", MemoryLayer.ORG)
await memory.add("timeout", "5s", MemoryLayer.SESSION)
entry = await memory.get("timeout")  # "30s" — ORG wins

# Give agents tools to manage their own memory
memory_tools = create_memory_tools(memory)  # memory_store, memory_recall, memory_list
```

### Semantic Tool Search

TF-IDF based tool selection for large toolkits — no external dependencies:

```python
from water.agents import create_tool_selector

selector = create_tool_selector(tools=all_tools, top_k=5, always_include=["bash"])

agent = create_agentic_task(
    id="smart-agent", provider=provider, tools=all_tools,
    tool_selector=selector,  # Narrows tools per iteration automatically
)
```

## Guardrails

Validate, filter, and constrain agent outputs:

```python
from water.guardrails import GuardrailChain, ContentFilter, SchemaGuardrail, CostGuardrail, TopicGuardrail

chain = GuardrailChain()
chain.add(ContentFilter(block_pii=True, block_injection=True))
chain.add(SchemaGuardrail(schema=OutputModel))
chain.add(CostGuardrail(max_tokens=4000, max_cost_usd=0.50))
chain.add(TopicGuardrail(allowed_topics=["python", "data science"]))

results = chain.check(output_data)
```

### Retry with Feedback

Automatically retry failed guardrails with LLM feedback:

```python
from water.guardrails import RetryWithFeedback

retry = RetryWithFeedback(max_retries=3,
    feedback_template="Failed: {violations}. Fix and retry.")
result = await retry.execute_with_retry(execute_fn, check_fn, params, context)
```

## Evaluation

Test and benchmark agent flows:

```python
from water.eval import EvalSuite, EvalCase, ExactMatch, LLMJudge

suite = EvalSuite(
    flow=math_flow,
    evaluators=[ExactMatch(key="answer"), LLMJudge(provider=provider, rubric="Is this correct?")],
    cases=[EvalCase(input={"q": "2+2"}, expected={"answer": "4"})],
)
report = await suite.run()
```

```bash
# CLI-based evaluation
water eval run eval_config.yaml
water eval compare run_001.json run_002.json
```

## Resilience

Built-in patterns for production reliability:

```python
from water.resilience import CircuitBreaker, RateLimiter, InMemoryCheckpoint, InMemoryDLQ
from water.resilience import FlowCache, InMemoryFlowCache, ProviderRateLimiter, ProviderLimits

# Circuit breaker — stop calling failing services
cb = CircuitBreaker(failure_threshold=5, recovery_timeout=30)

# Rate limiter — control throughput
limiter = RateLimiter(max_calls=100, period=60)

# Provider rate limiting — per-model RPM/TPM controls
provider_limiter = ProviderRateLimiter(limits={
    "gpt-4o": {"rpm": 60, "tpm": 150_000},
    "claude-sonnet-4-20250514": {"rpm": 40, "tpm": 100_000},
})
wait = await provider_limiter.acquire("gpt-4o", estimated_tokens=500)

# Flow caching — cache entire flow results
cache = FlowCache(backend=InMemoryFlowCache(), ttl=600)

# Checkpoint — resume flows after crashes
flow.checkpoint = InMemoryCheckpoint()

# Dead-letter queue — capture failed tasks
flow.dlq = InMemoryDLQ()
```

Tasks also support retry and timeout out of the box:

```python
task = create_task(
    id="flaky", execute=call_api,
    retry_count=3, retry_delay=1.0, retry_backoff=2.0, timeout=30.0,
)
```

## Integrations

### MCP (Model Context Protocol)

Expose flows as MCP tools or call external MCP servers:

```python
from water.integrations import MCPServer, MCPClient, create_mcp_task

server = MCPServer(flows=[my_flow])
client = MCPClient(server_url="http://localhost:3000")
mcp_task = create_mcp_task(id="search", client=client, tool_name="web_search")
```

For a live search and fetch example, see
[`parallel_search_flow.py`](cookbook/integrations/parallel_search_flow.py).
It uses the official MCP SDK's Streamable HTTP transport and Water's MCP task
factory with the anonymous [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
endpoint. From the repository root, in a Python 3.10+ virtual environment:

```bash
pip install . 'mcp>=1.30,<2'
python -m cookbook.integrations.parallel_search_flow
```

No Parallel API key or LLM is needed. Anonymous access has lower rate limits;
this example is intended for exploration and light use.

### A2A (Agent-to-Agent Protocol)

Expose flows as discoverable A2A agents or call remote agents:

```python
from water.integrations import A2AServer, A2AClient, AgentSkill, create_a2a_task

# Serve
server = A2AServer(flow=my_flow, name="Research Agent",
    skills=[AgentSkill(id="research", name="Research", description="Research topics")])
server.add_routes(app)  # serves /.well-known/agent.json + /a2a

# Consume
client = A2AClient(agent_url="https://remote-agent.example.com")
task = await client.send_task(input_data={"topic": "quantum computing"})
```

### Chat Adapters

Connect flows to Slack, Discord, or Telegram:

```python
from water.integrations import ChatBot, SlackAdapter

bot = ChatBot(adapter=SlackAdapter(token="xoxb-..."), flows=[support_flow])
```

### SSE Streaming & Event Triggers

```python
from water.integrations import StreamingFlow, StreamManager, add_streaming_routes
from water.triggers import WebhookTrigger, CronTrigger, QueueTrigger, TriggerRegistry

# Stream flow events via SSE
streaming = StreamingFlow(flow, StreamManager())

# Trigger flows from webhooks (with HMAC verification), cron, or queues
webhook = WebhookTrigger("my_flow", path="/hooks", secret="shared-secret")
cron = CronTrigger("report_flow", schedule="0 9 * * 1-5", input_data={"type": "daily"})
queue = QueueTrigger("process_flow", max_size=1000)

registry = TriggerRegistry()
registry.register(webhook)
registry.register(cron)
await registry.start_all()
```

## Observability

```python
from water.observability import (TelemetryManager, FlowDashboard, CostTracker, TokenUsage,
    StructuredLogger, auto_instrument)

# OpenTelemetry integration
telemetry = TelemetryManager(service_name="my-service")

# Built-in dashboard (served at /dashboard)
dashboard = FlowDashboard(storage=my_storage)

# Cost tracking with budget enforcement
tracker = CostTracker(budget_limit=10.0, on_budget_exceeded="warn")
flow.use(tracker)
summary = tracker.get_summary()

# Structured JSON logging with context
logger = StructuredLogger(level="info", format="json", redact_fields=["api_key"])
logger.set_context(flow_id="my_flow", execution_id="exec_001")
logger.info("Processing started", step="validation")

# Auto-instrumentation — zero-code tracing
instrumentor = auto_instrument(service_name="my-service", capture_input=True, capture_output=True)
flow.use(instrumentor)

# Execution replay — reproduce and debug past runs
from water.core.replay import ReplayEngine, ReplayConfig
engine = ReplayEngine(storage=my_storage)
result = await engine.replay(flow, session_id="exec_abc123",
    config=ReplayConfig(from_task="transform", override_inputs={"transform": {"mode": "v2"}}))
```

## Middleware, Hooks & Events

```python
from water.middleware import HookManager, EventEmitter

# Hooks — register callbacks for lifecycle events
hooks = HookManager()
hooks.on("on_task_start", lambda task_id, **kw: print(f"Starting: {task_id}"))
hooks.on("on_task_error", lambda task_id, error, **kw: alert(error))

# Events — subscribe to real-time flow events
emitter = EventEmitter()
flow.events = emitter
subscription = emitter.subscribe()
async for event in subscription:
    print(f"[{event.event_type}] {event.task_id}")
```

## Plugins

Extend Water with custom storage, providers, middleware, guardrails, and integrations:

```python
from water.plugins import PluginRegistry, WaterPlugin, PluginType

class MyPlugin(WaterPlugin):
    name = "my_plugin"
    plugin_type = PluginType.STORAGE
    def register(self, app):
        app.register_storage("custom", MyStorage())

registry = PluginRegistry()
registry.register(MyPlugin())
# Or auto-discover via entry points: registry.discover()
```

## Flow Versioning

Track schema changes with compatibility checking and data migration:

```python
from water import SchemaRegistry, snapshot_flow_schemas

registry = SchemaRegistry()
registry.register_version("my_flow", "1.0.0", snapshot_flow_schemas(flow_v1))
registry.register_version("my_flow", "2.0.0", snapshot_flow_schemas(flow_v2))

changes = registry.check_compatibility("my_flow", "1.0.0", "2.0.0")
migrated = registry.migrate_data("my_flow", old_data, "1.0.0", "2.0.0")
```

## Server

Serve your flows as a REST API with one line:

```python
from water.server import FlowServer

server = FlowServer(flows=[flow_a, flow_b])
app = server.get_app()

# Routes:
# GET  /flows              — list all flows
# GET  /flows/{id}         — flow details
# POST /flows/{id}/run     — execute a flow
# GET  /health             — health check
# GET  /dashboard          — observability UI
```

```bash
uvicorn app:app --reload
```

## CLI

```bash
# Run a flow
water run cookbook.core.sequential_flow:registration_flow --input '{"email": "a@b.com", "password": "secret", "first_name": "Water"}'

# Visualize as Mermaid diagram
water visualize cookbook.core.dag_flow:pipeline_flow

# Validate without executing
water dry-run cookbook.core.sequential_flow:registration_flow --input '{"email": "a@b.com"}'

# List all flows in a module
water list cookbook.core.sequential_flow

# Run evaluations
water eval run eval_config.yaml
water eval compare run_001.json run_002.json
water eval list ./evals/

# Deploy to Render
water flow prod:render --app playground
```

## Architecture

```
water/
├── core/           # Flow, Task, ExecutionEngine, Context, SubFlow, Replay, Versioning
├── agents/         # LLM tasks, streaming, multi-agent, tools, context, prompts,
│                   #   fallback, batch, planner, approval, human-in-the-loop, sandbox,
│                   #   agentic loop (ReAct), sub-agents, layered memory, tool search
├── guardrails/     # Content filter, schema, cost, topic guardrails, retry-with-feedback
├── eval/           # EvalSuite, evaluators, CLI, YAML/JSON config
├── storage/        # InMemory, SQLite, Redis, Postgres backends
├── resilience/     # Circuit breaker, rate limiter, cache, checkpoint, DLQ,
│                   #   flow cache, provider rate limiter
├── middleware/     # Middleware, hooks, events
├── integrations/   # MCP, A2A protocol, chat adapters, SSE streaming
├── triggers/       # Webhook, cron, queue triggers with registry
├── observability/  # Telemetry, dashboard, cost tracking, structured logging,
│                   #   auto-instrumentation
├── plugins/        # Plugin registry with entry-point discovery
├── server/         # FlowServer (FastAPI)
├── tasks/          # Built-in task library (HTTP, JSON transform, file I/O, etc.)
└── utils/          # Testing, scheduler, declarative loader, secrets, CLI
```

## Cookbook

The [`cookbook/`](cookbook/) directory has 73 runnable examples organized by category:

| Category | Examples |
|----------|----------|
| [**core/**](cookbook/core/) | Sequential, parallel, branching, loops, map, DAG, subflow, try-catch, replay, versioning, validation, contracts |
| [**agents/**](cookbook/agents/) | LLM tasks, streaming, multi-agent, tools, fallback chains, prompts, batch, planner, approval, human-in-the-loop, sandbox, agentic loop, sub-agents, memory, tool search |
| [**real_world/**](cookbook/real_world/) | Claude Code-style coding agent |
| [**resilience/**](cookbook/resilience/) | Circuit breaker, rate limiting, provider rate limits, flow cache, checkpointing, DLQ, caching, retry/timeout |
| [**observability/**](cookbook/observability/) | Cost tracking, auto-instrumentation, structured logging, tracing, telemetry, dashboard |
| [**integrations/**](cookbook/integrations/) | MCP, A2A protocol, chat bots, SSE streaming, triggers |
| [**server/**](cookbook/server/) | REST server, playground, deployment |
| [**utils/**](cookbook/utils/) | Testing, secrets, plugins, declarative flows, scheduler |
| [**storage/**](cookbook/storage/) | Storage backends |
| [**middleware/**](cookbook/middleware/) | Hooks, events, middleware |

## Contributing

We welcome contributions — bug reports, feature requests, code, docs, and testing.

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.
