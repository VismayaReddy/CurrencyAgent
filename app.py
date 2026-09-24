import os
import uvicorn
import requests
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda


# --- 1. Define Tools ---

@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
    """Convert an amount from one currency to another using the FastFOREX API."""

    api_key = os.environ.get("FASTFOREX_API_KEY")

    url = "https://api.fastforex.io/convert"

    params = {
        "from": from_currency.upper(),
        "to": to_currency.upper(),
        "amount": amount
    }

    headers = {
        "X-API-Key": api_key
    }

    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        return "Unable to fetch currency conversion data."

    data = response.json()

    result = data.get("result", {})

    converted_amount = result.get(to_currency.upper())

    if converted_amount is None:
        return "Unable to convert the given currencies."

    return (
        f"{amount} {from_currency.upper()} = "
        f"{converted_amount} {to_currency.upper()}"
    )


@tool
def summarize_text(text: str) -> str:
    """Summarize the given text in a short and simple way."""

    response = llm_flash.invoke(
        f"Summarize the following text in simple and concise points:\n\n{text}"
    )

    return response.content


tools = [currency_converter, summarize_text]


# --- 2. Initialize Model & Agent ---

GOOGLE_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a Currency and Text Summarization Agent. "
        "You can convert currencies and summarize text. "
        "For currency conversion, always use the currency_converter tool. "
        "For text summarization, always use the summarize_text tool. "
        "For questions unrelated to currency conversion or text summarization, "
        "say: 'I am not authorized to answer questions outside of currency conversion and text summarization.'"
    )
)


class AgentInput(BaseModel):
    input: str = Field(description="Your message to the agent")


def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input

    return {
        "messages": [("user", user_input)]
    }


def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")

    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        content = getattr(last, "content", str(last))

        if isinstance(content, list):
            return "".join(
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )

        return str(content)

    return str(agent_output)


formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(
    input_type=AgentInput,
    output_type=str
)


# --- 3. FastAPI App ---

app = FastAPI(
    title="Currency & Text Summarization Agent",
    version="1.0",
    description=(
        "A LangChain agent using Gemini with currency conversion "
        "and text summarization tools, served via LangServe."
    ),
)


@app.get("/")
def root():
    return {
        "message": "Server is running. Visit /agent/playground/ to chat, or /docs for the API."
    }


add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )