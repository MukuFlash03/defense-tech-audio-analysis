from restack_ai.function import function, log
from dataclasses import dataclass
from groq import Groq
import os
import base64
from dotenv import load_dotenv
import assemblyai as aai

load_dotenv()

@dataclass
class FunctionInputParams:
    file_data: tuple[str, str]

@function.defn()
async def identify_speakers(input: FunctionInputParams):
    try:
        log.info("speaker identification function started", input=input)
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

        aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")

        config = aai.TranscriptionConfig(
            speaker_labels=True,
            language_code="ru"
        )

        transcriber = aai.Transcriber(config=config)

        filename, base64_content = input.file_data
        print("Filename: ", filename)

        # FILE_URL = "https://assembly.ai/wildfires.mp3"

        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        FILE_URL = os.path.join(project_root, 'synthetic_audio', filename)

        print("FILE_URL: ", FILE_URL)

        transcript = transcriber.transcribe(FILE_URL)

        # extract all utterances from the response
        utterances = transcript.utterances

        # For each utterance, print its speaker and what was said
        for utterance in utterances:
          speaker = utterance.speaker
          text = utterance.text
          print(f"Speaker {speaker}: {text}")

        # filename, base64_content = input.file_data
        # file_bytes = base64.b64decode(base64_content)
        # transcription = client.audio.transcriptions.create(
        #     file=(filename, file_bytes), # Required audio file
        #     model="whisper-large-v3-turbo", # Required model to use for transcription
        #     # Best practice is to write the prompt in the language of the audio, use translate.google.com if needed
        #     prompt=f"Опиши о чем речь в аудио",  # Translation: Describe what the audio is about
        #     language="ru", # Original language of the audio
        #     response_format="json",  # Optional
        #     temperature=0.0  # Optional
        # )

        log.info("speaker identification function completed", transcription=utterances)

        formatted_transcript = {
            'utterances': [
                {
                    'speaker': utterance.speaker,
                    'text': utterance.text,
                    # Optionally include other metadata if needed later
                    'start': utterance.start,
                    'end': utterance.end,
                    'confidence': utterance.confidence
                }
                for utterance in transcript.utterances
            ]
        }
        
        log.info("speaker identification transcription formatting completed", transcription=utterances)
        
        return formatted_transcript
        # return utterances
        

    except Exception as e:
        log.error("speaker identification function failed", error=e)
        raise e
