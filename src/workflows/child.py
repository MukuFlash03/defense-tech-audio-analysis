from datetime import timedelta
from dataclasses import dataclass
from restack_ai.workflow import workflow, import_functions, log
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
    # from src.functions.identify_speaker import (
    #     process_command_audio,
    #     FunctionInputParams as IdentifySpeakerFunctionInputParams,
    # )


@dataclass
class WorkflowInputParams:
    file_data: tuple[str, str]


@dataclass
class WorkflowOutputParams:
    transcription: str
    translation: str
    translation_2: str


@workflow.defn()
class ChildWorkflow:
    @workflow.run
    async def run(self, input: WorkflowInputParams):
        log.info("ChildWorkflow started", input=input)

        transcription = await workflow.step(
            transcribe,
            TranscribeFunctionInputParams(file_data=input.file_data),
            start_to_close_timeout=timedelta(seconds=120),
        )

        translation_prompt = f"""
        Instructions: Translate the following content to English. Output only the translated content.
        Content: {transcription['text']}
        """

        translation = await workflow.step(
            translate,
            TranslationFunctionInputParams(user_prompt=translation_prompt),
            start_to_close_timeout=timedelta(seconds=120),
        )

        # speaker_analysis = await workflow.step(
        #     process_command_audio,
        #     IdentifySpeakerFunctionInputParams(
        #         audio_data=input.file_data,
        #         command_text=translation["content"],
        #     ),
        #     start_to_close_timeout=timedelta(seconds=120),
        # )

        speaker_identification_transcript = await workflow.step(
            identify_speakers,
            IdentifySpeakerFunctionInputParams(file_data=input.file_data),
            start_to_close_timeout=timedelta(seconds=120),
        )

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

        # formatted_translation = format_translated_conversation(translation_2)
        # log.info("Formatted Translation:")
        # log.info(formatted_translation)

        log.info(
            "ChildWorkflow completed",
            transcription=transcription["text"],
            translation=translation["content"],
            translation_2=translation_2["content"],
            # speaker_identification=speaker_identification["content"],
        )
        return WorkflowOutputParams(
            transcription=transcription["text"],
            translation=translation["content"],
            translation_2=translation_2["content"],
            # speaker_identification=speaker_identification["content"],
        )
