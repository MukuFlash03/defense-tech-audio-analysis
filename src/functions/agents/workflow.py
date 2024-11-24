from typing import List, Type, TypeVar, Any
from pydantic import BaseModel
import asyncio
import os
import sys
from pathlib import Path
import logging
from dataclasses import dataclass
from restack_ai.function import function, log

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup path resolution
API_DIR = Path(__file__).parent
ROOT_DIR = API_DIR.parent
sys.path.append(str(API_DIR))

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(os.path.dirname(current_file_path))
sys.path.append(parent_directory)

from custom_types import (
    ConversationAnalysis
)

@dataclass
class FunctionInputParams:
    user_prompt: str

sample_json_db = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "sample_json_db")
conversation_analysis_file = os.path.join(sample_json_db, "conversation_analysis.json")

from basic_agent import (
    parse_info_async,
)

async def get_conversation_info(conversationInput: FunctionInputParams) -> Any:
    base_folder = os.path.join(parent_directory, "data")

    log.info("Inside get_conversation_info in workflow.py")

    task_list: Any = []
    
    log.info("Before calling parse_info_async")
    conversation_analysis_task = parse_info_async(conversationInput)
    log.info("After calling parse_info_async")

    task_list = [
        asyncio.create_task(task)
        for task in [
            conversation_analysis_task
        ]
    ]

    conversation_analysis_response = (
        await asyncio.gather(*task_list)
    )

    import json

    log.info("Received conversation_analysis: ")
    log.info(conversation_analysis_response)

    conversation_analysis = conversation_analysis_response[0]

    log.info("Before trying to save conversation_analysis")
    with open(conversation_analysis_file, "w") as f:
        json.dump(conversation_analysis.model_dump(), f, indent=4)
    log.info("After trying to save conversation_analysis")

    # with open(education_file, "w") as f:
    #     json.dump(educations.model_dump(), f, indent=4)

    # with open(skill_file, "w") as f:
    #     json.dump(skills.model_dump(), f, indent=4)

    # with open(project_file, "w") as f:
    #     json.dump(projects.model_dump(), f, indent=4)

    # with open(achievement_file, "w") as f:
    #     json.dump(achievements.model_dump(), f, indent=4)

    # with open(personal_details_file, "w") as f:
    #     json.dump(personal_details.model_dump(), f, indent=4)

    # with open(question_answer_file, "w") as f:
    #     json.dump(qa_list.model_dump(), f, indent=4)
    
    """
    Uncomment above
    """

    # user = User(
    #     # personal_details=personal_details,
    #     # experiences=experiences,
    #     # educations=educations,
    #     # skills=skills,
    #     projects=projects.model_dump(),
    #     achievements=achievements,
    #     questionAnswer=qa_list,
    # )

    # for i, experience in enumerate(experiences.experiences):
    #     print("Company", i + 1, experience.company)
    #     print("Title", i + 1, experience.title)
    #     print("Location", i + 1, experience.location)
    #     print("Start Date", i + 1, experience.start_date)
    #     print("End Date", i + 1, experience.end_date)


    return (conversation_analysis)
