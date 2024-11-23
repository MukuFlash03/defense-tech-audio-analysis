from restack_ai.function import function, log
from openai import OpenAI
from dataclasses import dataclass
import os


@dataclass
class FunctionInputParams:
    user_prompt: str


@function.defn()
async def translate(input: FunctionInputParams):
    try:
        log.info("translate function started", input=input)

        # Verify environment variable exists
        api_url = os.environ.get("OPENBABYLON_API_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_url:
            raise ValueError("OPENBABYLON_API_URL environment variable is not set")

        # Add timeout and better client configuration
        client = OpenAI(
            api_key=api_key,
            # base_url=api_url,
            timeout=30.0,  # Add timeout in seconds
        )

        messages = []
        if input.user_prompt:
            messages.append({"role": "user", "content": input.user_prompt})

        response = client.chat.completions.create(
            # model="orpo-mistral-v0.3-ua-tokV2-focus-10B-low-lr-1epoch-aux-merged-1ep",
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
        )
        log.info("translate function completed", response=response)
        return response.choices[0].message
    except ValueError as ve:
        log.error("Configuration error", error=str(ve))
        raise
    except Exception as e:
        # Add more context to the error logging
        log.error(
            "translate function failed",
            error=str(e),
            error_type=type(e).__name__,
            api_url=os.environ.get("OPENBABYLON_API_URL"),
        )
        raise
