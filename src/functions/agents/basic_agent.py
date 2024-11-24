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
            model="gpt-4o",
            messages=[
                {
                    "role": "system", 
                    "content": f"""
                        Analyze the military conversation and extract detailed tactical information using this structure:

                        Priority Level: Assess urgency based on tactical situation (High/Medium/Low)
                        
                        Risk Assessment: Evaluate immediate military threats, enemy movements, and tactical vulnerabilities
                        
                        Key Insights: Summarize critical military information including:
                        - Enemy force composition and movements
                        - Distances and directions
                        - Tactical objectives identified
                        - Support requirements
                        
                        Critical Entities: List key military assets, personnel, and locations mentioned
                        
                        Locations Mentioned: Extract all geographic references, including:
                        - Cities/Towns
                        - Roads/Routes
                        - Tactical landmarks
                        
                        Sentiment Summary: Analyze operational urgency and command dynamics
                        
                        Source Reliability: Use standard A-F classification. Randomize this, doesn't have to be accurate.
                          A - Completely reliable: No doubt of authenticity, trustworthiness, or competency; has a history of complete reliability
                          B - Usually reliable: Minor doubt about authenticity, trustworthiness, or competency; has a history of valid information most of the time
                          C - Fairly reliable: Doubt of authenticity, trustworthiness, or competency but has provided valid information in the past
                          D - Not usually reliable: Significant doubt about authenticity, trustworthiness, or competency but has provided valid information in the past
                          E - Unreliable: Lacking in authenticity, trustworthiness, and competency; history of invalid information
                          F - Reliability cannot be judged: No basis exists for evaluating the reliability of the source
                        
                        Information Credibility: Use standard 1-6 classification. Randomize this, doesn't have to be accurate.
                          1 - Confirmed by other sources: Confirmed by other independent sources; logical in itself; Consistent with other information on the subject
                          2 - Probably True: Not confirmed; logical in itself; consistent with other information on the subject
                          3 - Possibly True: Not confirmed; reasonably logical in itself; agrees with some other information on the subject
                          4 - Doubtful: Not confirmed; possible but not logical; no other information on the subject
                          5 - Improbable: Not confirmed; not logical in itself; contradicted by other information on the subject
                          6 - Truth cannot be judged: No basis exists for evaluating the validity of the information
                        
                        Recommended Actions: List tactical recommendations based on the situation
                        
                        Entity Relationships: Document command structure and unit interactions
                        
                        Speakers: List all participants in the conversation
                        
                        Conversation Duration: Estimate length of exchange

                        Example Transcript:
                        Speaker B: Contact detected at the western outskirts of Bakhmut. Two armored vehicles are moving towards our position.
                        Speaker A: Dmitry, can you see them from your point?
                        Speaker B: Yes. Viktor, it seems to be part of the unit we've been tracking from the watch ravine. I count at least 12 infantrymen.
                        Speaker A: How close are they to the Berkhivska road?
                        Speaker B: About 800 meters south. It looks like they are setting up a forward position.
                        Speaker A: Dmitry, check if they are setting up heavy weapons. We have data on anti-tank positions in this sector.
                        Speaker B: Understood. Wait. I see movement towards Ivanivske. Several columns.
                        Speaker A: We need immediate support. I'm calling it in. Exact distance from your position.
                        Speaker B: One to two kilometers moving fast. We need artillery before they reach the tree line.
                        Speaker A: Coordinates confirmed. Hold your position and keep an eye on the target.

                        Example Analysis:
                           "priority_level": "High",
                            "risk_assessment": "Potential threat from enemy movement and positioning near Bakhmut.",
                            "key_insights": "Enemy forces, including two armored vehicles and at least 12 infantrymen, are moving towards the speaker's position near Bakhmut. They are approximately 800 meters south of Berkhivska Road and seem to be establishing a forward position. There is also movement towards Ivanivske, indicating a possible larger operation.",
                            "critical_entities": [
                                "Bakhmut",
                                "Berkhivska Road",
                                "Ivanivske",
                                "Dmitry",
                                "Victor"
                            ],
                            "locations_mentioned": [
                                "Bakhmut",
                                "Berkhivska Road",
                                "Ivanivske"
                            ],
                            "sentiment_summary": "The conversation reflects a sense of urgency and concern about enemy movements and the need for immediate support.",
                            "source_reliability": "B - Usually reliable",
                            "information_credibility": "2 - Probably True",
                            "recommended_actions": [
                                "Request immediate artillery support to target enemy movements before they reach the tree line.",
                                "Monitor the installation of heavy weaponry by the enemy.",
                                "Maintain current position and continue surveillance of enemy activities."
                            ],
                            "entity_relationships": "Speaker A and Speaker B are coordinating to monitor and respond to enemy movements near Bakhmut.",
                            "speakers": [
                                "Speaker A",
                                "Speaker B"
                            ],
                            "conversation_duration": "Short",
                            "analyzed_at": "2023-10-21T00:00:00Z"
                        
                        Focus on extracting actionable military intelligence from the conversation content.
                    """

                },
                {
                    "role": "user", 
                    # "content": "Extract the required detailed analysis from the conversation."
                    "content": input.user_prompt
                },
            ],
            # messages=[
            #     {
            #         "role": "system", 
            #         "content": f"""
            #             Extract the required detailed analysis from the conversation.

            #             Reliability of Source
            #             For source_reliability key, a source is assessed for reliability based on a technical assessment of its capability, 
            #             or in the case of Human Intelligence sources their history. 
            #             Notation uses Alpha coding, A-F:

            #             A - Completely reliable: No doubt of authenticity, trustworthiness, or competency; has a history of complete reliability
            #             B - Usually reliable: Minor doubt about authenticity, trustworthiness, or competency; has a history of valid information most of the time
            #             C - Fairly reliable: Doubt of authenticity, trustworthiness, or competency but has provided valid information in the past
            #             D - Not usually reliable: Significant doubt about authenticity, trustworthiness, or competency but has provided valid information in the past
            #             E - Unreliable: Lacking in authenticity, trustworthiness, and competency; history of invalid information
            #             F - Reliability cannot be judged: No basis exists for evaluating the reliability of the source
                      
            #           Credibility 
            #           For information_credibility key, an item is assessed for credibility based on likelihood and levels of corroboration by other sources
            #           which measures accuracy of data.
            #           Notation uses a numeric code, 1-6.

            #           1 - Confirmed by other sources: Confirmed by other independent sources; logical in itself; Consistent with other information on the subject
            #           2 - Probably True: Not confirmed; logical in itself; consistent with other information on the subject
            #           3 - Possibly True: Not confirmed; reasonably logical in itself; agrees with some other information on the subject
            #           4 - Doubtful: Not confirmed; possible but not logical; no other information on the subject
            #           5 - Improbable: Not confirmed; not logical in itself; contradicted by other information on the subject
            #           6 - Truth cannot be judged: No basis exists for evaluating the validity of the information
            #           """
            #     },
            #     {
            #         "role": "user", 
            #         "content": "Extract the required detailed analysis from the conversation."
            #         # "content": input.user_prompt
            #     },
            # ],
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

