from datetime import timedelta
from dataclasses import dataclass
from restack_ai.workflow import workflow, import_functions, log
from typing import Any
# from ..utils.util import format_translated_conversation

with import_functions():
    from src.functions.transcribe import (
        transcribe,
        FunctionInputParams as TranscribeFunctionInputParams,
    )
    from src.functions.translate import (
        translate,
        FunctionInputParams as TranslationFunctionInputParams,
    )
    from src.functions.speaker_identification import (
        identify_speakers,
        FunctionInputParams as IdentifySpeakerFunctionInputParams,
    )
    from src.functions.agents.extract_info import (
        extract_info,
        FunctionInputParams as ExtractInfoFunctionInputParams,
    )
    from src.functions.db_audio_analysis import (
        read_from_audio_table, write_to_audio_table,
        FunctionInputParams as WriteDataFunctionInputParams,
    )


@dataclass
class WorkflowInputParams:
    file_data: tuple[str, str]

@dataclass
class WorkflowOutputParams:
    # transcription: str
    # translation: str
    translation_2: str
    conversation_analysis: str
    db_write_audio: Any
    # db_read_audio: str

@dataclass
class WorkflowOutputTestParams:
    db_read_audio: str

@workflow.defn()
class ChildWorkflow:
    @workflow.run
    async def run(self, input: WorkflowInputParams):
        log.info("ChildWorkflow started", input=input)

        # transcription = await workflow.step(
        #     transcribe,
        #     TranscribeFunctionInputParams(file_data=input.file_data),
        #     start_to_close_timeout=timedelta(seconds=120),
        # )

        # translation_prompt = f"""
        # Instructions: Translate the following content to English. Output only the translated content.
        # Content: {transcription['text']}
        # """

        # translation = await workflow.step(
        #     translate,
        #     TranslationFunctionInputParams(user_prompt=translation_prompt),
        #     start_to_close_timeout=timedelta(seconds=120),
        # )

        log.info("Before fetching speaker_identification_transcript()")
        speaker_identification_transcript = await workflow.step(
            identify_speakers,
            IdentifySpeakerFunctionInputParams(file_data=input.file_data),
            start_to_close_timeout=timedelta(seconds=120),
        )
        log.info("After fetching speaker_identification_transcript()")

        # print("Speaker Identification: \n", speaker_identification_transcript)

        combined_text = "\n".join(
            f"Speaker {utterance['speaker']}: {utterance['text']}"
            for utterance in speaker_identification_transcript['utterances']
        )

        # Create translation prompt
        translation_prompt_2 = f"""
        Instructions: Translate the following content to English. Output only the translated content.
        Content: {combined_text}
        """

        log.info("Running translation_2....")

        # Pass to translation function
        translation_2 = await workflow.step(
            translate,
            TranslationFunctionInputParams(user_prompt=translation_prompt_2),
            start_to_close_timeout=timedelta(seconds=120),
        )

        log.info("Completed translation_2....")

        extract_info_prompt = f"""
        Instructions: Analyze the following content that is a military conversation transcript. Output only analyzed content as per format.
        Content: {combined_text}
        """

        log.info("Running extract_info....")

        # Pass to translation function
        extraction_json_data = await workflow.step(
            extract_info,
            ExtractInfoFunctionInputParams(user_prompt=extract_info_prompt),
            start_to_close_timeout=timedelta(seconds=120),
        )
        
        log.info("Extracted JSON data:")
        log.info(extraction_json_data)

        log.info("Completed extract_info....")


        log.info("Before writing to DB in child workflow")
        db_write_audio = await workflow.step(
              write_to_audio_table,
              WriteDataFunctionInputParams(conversation_analysis=extraction_json_data),
              start_to_close_timeout=timedelta(seconds=120),
          )
        log.info("After writing to DB in child workflow")

        # log.info("Before reading from DB in child workflow")
        # db_read_audio = await workflow.step(
        #       read_from_audio_table,
        #       start_to_close_timeout=timedelta(seconds=120),
        #   )
        # log.info("After reading from DB in child workflow")

        log.info(
            "ChildWorkflow completed",
            # transcription=transcription["text"],
            # translation=translation["content"],
            translation_2=translation_2["content"],
            conversation_analysis=extraction_json_data,
            db_write_audio=db_write_audio,
            # db_read_audio=db_read_audio
        )

        return WorkflowOutputParams(
            # transcription=transcription["text"],
            # translation=translation["content"],
            translation_2=translation_2["content"],
            conversation_analysis=extraction_json_data,
            db_write_audio=db_write_audio,
            # db_read_audio=db_read_audio
        )

        # return WorkflowOutputTestParams(
        #     db_read_audio=db_read_test
        # )
