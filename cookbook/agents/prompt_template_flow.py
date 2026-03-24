"""
Prompt Template Flow Example

Demonstrates the Water Prompt Templating Engine:
  - Basic template usage with {{variable}} interpolation
  - A PromptLibrary with persona templates
  - Composing multiple templates into one
  - Integrating templates with create_agent_task

NOTE: This example uses OpenAIProvider and requires a valid OPENAI_API_KEY.
"""

import asyncio
from water.agents import create_agent_task
from water.agents.llm import OpenAIProvider
from water.agents.prompts import PromptTemplate, PromptLibrary
from water.core import Flow, create_task


# ---------------------------------------------------------------------------
# 1. Basic template usage
# ---------------------------------------------------------------------------

def basic_template_demo():
    """Show simple variable interpolation and defaults."""
    print("=== Basic Template Usage ===\n")

    # Simple rendering
    greeting = PromptTemplate("Hello, {{name}}! Welcome to {{project}}.")
    print(greeting.render(name="Alice", project="Water"))

    # Using defaults
    with_defaults = PromptTemplate(
        "You are a {{role}} specializing in {{domain}}.",
        defaults={"role": "helpful assistant", "domain": "general tasks"},
    )
    print(with_defaults.render())                        # all defaults
    print(with_defaults.render(domain="data science"))   # partial override

    # Inspecting variables
    print(f"Variables: {with_defaults.get_variables()}")
    print(f"Missing if we supply nothing: {with_defaults.validate([])}")
    print()


# ---------------------------------------------------------------------------
# 2. PromptLibrary with persona templates
# ---------------------------------------------------------------------------

def library_persona_demo():
    """Register several persona templates and render them."""
    print("=== Prompt Library — Personas ===\n")

    lib = PromptLibrary()

    lib.register(
        "analyst",
        "You are a senior data analyst. Analyze the following: {{topic}}.",
    )
    lib.register(
        "creative_writer",
        "You are a creative writer with a {{tone}} tone. Write about: {{subject}}.",
        defaults={"tone": "witty"},
    )
    lib.register(
        "code_reviewer",
        "You are a code reviewer. Review the {{language}} code below and provide feedback.\n\n{{code}}",
    )

    print(lib.render("analyst", topic="Q3 revenue trends"))
    print()
    print(lib.render("creative_writer", subject="a rainy afternoon"))
    print()
    print(lib.render(
        "code_reviewer",
        language="Python",
        code="def add(a, b): return a + b",
    ))
    print()
    print(f"Registered templates: {lib.list_templates()}")
    print()


# ---------------------------------------------------------------------------
# 3. Composing templates
# ---------------------------------------------------------------------------

def composition_demo():
    """Compose multiple templates into a single prompt."""
    print("=== Template Composition ===\n")

    lib = PromptLibrary()

    lib.register("system", "You are {{role}}. Always be {{style}}.")
    lib.register("context", "Here is background context:\n{{context}}")
    lib.register("instruction", "Now, {{action}} based on the above.")

    composed = lib.compose("system", "context", "instruction")
    prompt = composed.render(
        role="a research assistant",
        style="concise and accurate",
        context="The user is studying climate change impacts on agriculture.",
        action="provide a summary of key findings",
    )
    print(prompt)
    print()


# ---------------------------------------------------------------------------
# 4. Using templates with create_agent_task
# ---------------------------------------------------------------------------

async def agent_task_demo():
    """Render a prompt template and feed it to an agent task inside a Flow."""
    print("=== Prompt Template + Agent Task ===\n")

    lib = PromptLibrary()
    lib.register(
        "summarizer",
        "Summarize the following {{format}}:\n\n{{content}}",
        defaults={"format": "text"},
    )

    # Render the prompt
    prompt_text = lib.render(
        "summarizer",
        format="article",
        content="Water is an open-source agent harness framework for building "
                "reliable AI pipelines. It supports flows, tasks, retries, "
                "checkpoints, and much more.",
    )

    # Build a flow with an agent task that uses the rendered prompt
    provider = OpenAIProvider(model="gpt-4o-mini", temperature=0.3)

    summarize = create_agent_task(
        id="summarize",
        provider_instance=provider,
        system_prompt="You are a summarization assistant.",
        prompt_template=prompt_text,   # the rendered prompt used as the template
    )

    flow = Flow("summarize_flow")
    flow.then(summarize).register()

    result = await flow.run({"prompt": prompt_text})
    print(f"Agent response: {result.get('response', result)}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    basic_template_demo()
    library_persona_demo()
    composition_demo()
    asyncio.run(agent_task_demo())


if __name__ == "__main__":
    main()
