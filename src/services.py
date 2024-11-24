import asyncio
from src.client import client
from src.functions.transcribe import transcribe
from src.functions.translate import translate
from src.functions.speaker_identification import identify_speakers
from src.workflows.child import ChildWorkflow
from src.workflows.parent import ParentWorkflow
from src.functions.agents.extract_info import extract_info
from src.functions.db_audio_analysis import read_from_audio_table, write_to_audio_table

async def main():
    await asyncio.gather(
        client.start_service(
            workflows=[ParentWorkflow, ChildWorkflow],
            functions=[transcribe, translate, identify_speakers, extract_info, read_from_audio_table]
        )
    )

def run_services():
    asyncio.run(main())

if __name__ == "__main__":
    run_services()
