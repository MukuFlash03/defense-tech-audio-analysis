from restack_ai.function import function, log
from openai import OpenAI
from dataclasses import dataclass
import os
import sys
from typing import List, Type, TypeVar, Any
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)

from .custom_types import ( 
  ConversationAnalysis
)

from .workflow import (
    get_conversation_info
)

# Define a generic type variable
T = TypeVar("T", bound=BaseModel)

print("Inside extract_info.py")
print("OpenAI API key:", os.getenv("OPENAI_API_KEY"))

# client: OpenAI = OpenAI()
# async_client: AsyncOpenAI = AsyncOpenAI()


@dataclass
class FunctionInputParams:
    user_prompt: str

@function.defn()
async def extract_info(input: FunctionInputParams):
    try:
        log.info("extract_info function started", input=input)

        log.info("Before calling get_conversation_info")
        conversation_analysis = (
            await get_conversation_info(input)
        )
        log.info("After calling get_conversation_info")

        log.info("Before returning json_data in extract_info")
        json_data = conversation_analysis.model_dump_json(indent=4)
        log.info("After returning json_data in extract_info") 

        return json_data
  
    except ValueError as ve:
        log.error("Inside extract_info: Configuration error", error=str(ve))
        raise
    except Exception as e:
        log.error(
            "extract_info function failed",
            error=str(e),
            error_type=type(e).__name__,
            api_url=os.environ.get("OPENBABYLON_API_URL"),
        )
        raise
