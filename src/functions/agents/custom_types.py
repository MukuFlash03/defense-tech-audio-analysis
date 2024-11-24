from pydantic import BaseModel
from typing import Dict, List

class ConversationAnalysis(BaseModel):
    priority_level: str
    risk_assessment: str
    key_insights: str
    critical_entities: List[str]
    locations_mentioned: List[str]
    sentiment_summary: str
    source_reliability: str
    information_credibility: str
    recommended_actions: List[str]
    entity_relationships: str
    # emotions_per_speaker: Dict[str, List[str]]
    speakers: List[str]
    # speaker_roles: Dict[str, str]
    conversation_duration: str
    analyzed_at: str


# # DO NOT USE THIS FOR NOW. Directly output each stuff as model_dump()
# class User(BaseModel):
#     personal_details: PersonalDetails
#     experience: ExperienceList
#     skills: SkillList
#     achievements: AchievementList
#     projects: ProjectList
#     education: EducationList
#     questionAnswer: QuestionAnswerList
