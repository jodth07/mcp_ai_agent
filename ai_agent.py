import json
from typing import Any, List, Optional

# LangChain Core imports
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

from clients import OllamaMistralClient
from tools import all_tools


class LangChainCompatibleLLM(BaseChatModel):
    model_client: Any = Field()
    tool_schemas: Optional[Any] = None

    def _call(self, messages: List[HumanMessage], **kwargs) -> str:
        prompt = "\n".join([f"{m.type.upper()}: {m.content}" for m in messages])
        return self.model_client.generate(prompt)

    def _generate(self, messages: List[HumanMessage], stop=None, **kwargs) -> ChatResult:
        text = self._call(messages, **kwargs)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    @property
    def _llm_type(self) -> str:
        return "local_mixtral"

    def bind_tools(self, tools):
        tool_schemas = [convert_to_openai_tool(tool) for tool in tools]
        self.tool_schemas = tool_schemas
        if hasattr(self.model_client, "tool_schemas"):
            self.model_client.tool_schemas = tool_schemas
        return self


def build_system_prompt(tools):
    tool_descriptions = "\n".join(
        [f"- {tool.name}: {tool.description}" for tool in tools]
    )
    return (
        "You are an AI agent. You must ONLY use the provided tools.\n"
        "Here are the available tools:\n"
        f"{tool_descriptions}\n\n"
        "When answering, respond strictly in JSON format like this:\n"
        "{{\"tool_name\": \"toolname\", \"parameters\": {{\"param1\": \"value\"}}}}\n"
        "Do not guess the tool's output. Only provide the tool_name and parameters."
    )

def create_agent(tools):
    local_llm_client = OllamaMistralClient()
    wrapped_llm = LangChainCompatibleLLM(model_client=local_llm_client)

    system_prompt_text = build_system_prompt(tools)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=wrapped_llm, tools=tools, prompt=prompt)

    return AgentExecutor(agent=agent, tools=tools, verbose=True)

def process_input(user_input: str, tool_lookup, agent_executor) -> str:
    result = agent_executor.invoke({"input": user_input})

    if "output" in result:
        try:
            tool_call = json.loads(result["output"])
            tool_name = tool_call.get("tool_name")
            parameters = tool_call.get("parameters", {})

            print(f"\nParsed Tool Call: {tool_call}")

            if tool_name in tool_lookup:
                tool = tool_lookup[tool_name]
                tool_result = tool.invoke(parameters)
                print(f"\nAgent Tool Output:\n{tool_result}")
            else:
                print(f"\nUnknown tool requested: {tool_name}")
        except (json.JSONDecodeError, TypeError, KeyError):
            print(f"\nAgent Raw Output:\n{result['output']}")
    else:
        print("\nAgent did not produce any output.")

def main():

    tools = all_tools
    tool_lookup = {tool.name: tool for tool in tools}
    executor = create_agent(tools)

    while True:
        user_input = input("> ")
        if len(user_input.strip()) > 1:
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting the agent.")
                break

            process_input(user_input, tool_lookup, executor)


if __name__ == "__main__":
    main()
