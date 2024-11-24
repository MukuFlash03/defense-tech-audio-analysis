from typing import List, Type, TypeVar, Any
from pydantic import BaseModel
import sys
import os
from dataclasses import dataclass
from restack_ai.function import function, log
from openai import OpenAI, AsyncOpenAI

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)


from custom_types import ( 
  ConversationAnalysis
)

@dataclass
class FunctionInputParams:
    user_prompt: str

# Define a generic type variable
T = TypeVar("T", bound=BaseModel)

print("Inside basic_agent.py")
print("OpenAI API key:", os.getenv("OPENAI_API_KEY"))

# client: OpenAI = OpenAI()
# async_client: AsyncOpenAI = AsyncOpenAI()


# def parse_input(
#     system_content: str,
#     user_content: str,
#     response_format: Type[T],
#     # model: str = "gpt-4o-2024-08-06",
#     model: str = "gpt-4o-mini",
# ) -> T:
#     """
#     Generates a response from OpenAI based on the given inputs and model.

#     Args:
#         model (str): The OpenAI model to use for the completion.
#         system_content (str): Content for the system role.
#         user_content (str): Content for the user query.
#         response_format (Type[T]): The class type of the response format (a Pydantic model).

#     Returns:
#         T: Parsed response from the completion in the type specified by response_format.
#     """
#     completion = client.beta.chat.completions.parse(
#         model=model,
#         messages=[
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": user_content},
#         ],
#         response_format=response_format,
#     )

#     if completion.choices[0].message.parsed is None:
#         raise ValueError("Failed to parse response.")

#     return completion.choices[0].message.parsed

# async def parse_input_async(
#     system_content: str,
#     user_content: str,
#     response_format: Type[T],
#     # model: str = "gpt-4o-2024-08-06",
#     model: str = "gpt-4o-mini",
# ) -> T:
#     """
#     Generates a response from OpenAI based on the given inputs and model.

#     Args:
#         model (str): The OpenAI model to use for the completion.
#         system_content (str): Content for the system role.
#         user_content (str): Content for the user query.
#         response_format (Type[T]): The class type of the response format (a Pydantic model).

#     Returns:
#         T: Parsed response from the completion in the type specified by response_format.
#     """
#     completion = await async_client.beta.chat.completions.parse(
#         model=model,
#         messages=[
#             {"role": "system", "content": system_content},
#             {"role": "user", "content": user_content},
#         ],
#         response_format=response_format,
#     )

#     if completion.choices[0].message.parsed is None:
#         raise ValueError("Failed to parse response.")

#     return completion.choices[0].message.parsed


async def parse_info_async(input: FunctionInputParams):
    try:
        log.info("parse_info_async function started", input=input)

        # Verify environment variable exists
        api_url = os.environ.get("OPENBABYLON_API_URL")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_url:
            raise ValueError("OPENBABYLON_API_URL environment variable is not set")

        # Add timeout and better client configuration
        # client = OpenAI(
        #     api_key=api_key,
        #     # base_url=api_url,
        #     timeout=30.0,  # Add timeout in seconds
        # )

        async_client = AsyncOpenAI(
        # client = OpenAI(
            api_key=api_key,
            # base_url=api_url,
            timeout=30.0,  # Add timeout in seconds
        )

        log.info("About to call OpenAI API")

        response = await async_client.beta.chat.completions.parse(
        # response = await client.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": f"""
                        Extract the required detailed analysis from the conversation.

                        Reliability of Source
                        For source_reliability key, a source is assessed for reliability based on a technical assessment of its capability, 
                        or in the case of Human Intelligence sources their history. 
                        Notation uses Alpha coding, A-F:

                        A - Completely reliable: No doubt of authenticity, trustworthiness, or competency; has a history of complete reliability
                        B - Usually reliable: Minor doubt about authenticity, trustworthiness, or competency; has a history of valid information most of the time
                        C - Fairly reliable: Doubt of authenticity, trustworthiness, or competency but has provided valid information in the past
                        D - Not usually reliable: Significant doubt about authenticity, trustworthiness, or competency but has provided valid information in the past
                        E - Unreliable: Lacking in authenticity, trustworthiness, and competency; history of invalid information
                        F - Reliability cannot be judged: No basis exists for evaluating the reliability of the source
                      
                      Credibility 
                      For information_credibility key, an item is assessed for credibility based on likelihood and levels of corroboration by other sources
                      which measures accuracy of data.
                      Notation uses a numeric code, 1-6.

                      1 - Confirmed by other sources: Confirmed by other independent sources; logical in itself; Consistent with other information on the subject
                      2 - Probably True: Not confirmed; logical in itself; consistent with other information on the subject
                      3 - Possibly True: Not confirmed; reasonably logical in itself; agrees with some other information on the subject
                      4 - Doubtful: Not confirmed; possible but not logical; no other information on the subject
                      5 - Improbable: Not confirmed; not logical in itself; contradicted by other information on the subject
                      6 - Truth cannot be judged: No basis exists for evaluating the validity of the information
                      """
                },
                {
                    "role": "user", 
                    "content": "Extract the required detailed analysis from the conversation."
                    # "content": input.user_prompt
                },
            ],
            temperature=0.0,
            response_format=ConversationAnalysis,
        )

        log.info("OpenAI API response received")

        if response.choices[0].message.parsed is None:
            raise ValueError("Failed to parse response.")
        
        log.info("parse_info_async function completed", response=response)
        return response.choices[0].message.parsed
    except ValueError as ve:
        log.error("Inside parse_info_async: Configuration error", error=str(ve))
        raise
    except Exception as e:
        log.error(
            "parse_info_async function failed",
            error=str(e),
            error_type=type(e).__name__,
            api_url=os.environ.get("OPENBABYLON_API_URL"),
        )
        raise

# async def main() -> str:
#     personal_details: PersonalDetails = await parse_input_async(
#         system_content="""Extract the personal details of name, \
# email, location and phone from the resume.""",
#         user_content="Email : dheerajpai@fgmail.com<Dheeraj>, Palo Alto, 268-987-DPAI",
#         response_format=PersonalDetails,
#     )
#     # return personal_details
#     return ""


# if __name__ == "__main__":
#     import asyncio

#     asyncio.run(main())

#     # Example to test parse_input_async function

#     pass
