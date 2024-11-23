architecture_explanation.md

# War Audio Transcription & Translation System Documentation

## Overview
This repository contains a system designed to transcribe war-related audio recordings (primarily in Russian) and translate them into English. It uses a modern tech stack combining Restack AI, Streamlit, FastAPI, Groq, and OpenBabylon.

## Architecture

### Components

1. **Frontend (Streamlit)**
   - Provides a web interface for users to upload audio files
   - Displays transcription and translation results
   - Located in:
   
```1:57:frontend.py
import streamlit as st
import requests
import base64
# Set page title and header
st.title("Defense Hackathon Quickstart: War Audio Transcription & Translation")



uploaded_files = st.file_uploader("Choose a files", accept_multiple_files=True)

if uploaded_files:
    file_data_list = []
    for uploaded_file in uploaded_files:
        audio_data = uploaded_file.read()
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        file_data_list.append((uploaded_file.name, audio_base64))

if "response_history" not in st.session_state:
    st.session_state.response_history = []

if st.button("Process Audio"):
    if uploaded_file:
        try:
            with st.spinner('Processing audio...'):
                response = requests.post(
                    "http://localhost:8000/api/process_audio",
                    json={"file_data": file_data_list}
                )

                if response.status_code == 200:
                    st.success("Processing audio was successful!")

                    results = response.json()["result"]
                    for idx, uploaded_file in enumerate(uploaded_files):
                        st.session_state.response_history.append({
                            "file_name": uploaded_file.name,
                            "file_type": uploaded_file.type,
                            "transcription": results[idx]['transcription'],
                            "translation": results[idx]['translation']
                    })
                else:
                    st.error(f"Error: {response.status_code}")

        except requests.exceptions.ConnectionError as e:
            st.error(f"Failed to connect to the server. Make sure the FastAPI server is running.")
    else:
        st.warning("Please upload a file before submitting.")
if st.session_state.response_history:
    st.subheader("Audio Processing History")
    for i, item in enumerate(st.session_state.response_history, 1):
        st.markdown(f"**Run {i}:**")
        st.markdown(f"**File Name:** {item['file_name']}")
        st.markdown(f"**File Type:** {item['file_type']}")
        st.markdown(f"**Transcription:** {item['transcription']}")
        st.markdown(f"**Translation:** {item['translation']}")
        st.markdown("---")
```


2. **Backend API (FastAPI)**
   - Handles file uploads and processing requests
   - Communicates with Restack workflows
   - Located in:
   
```1:58:src/app.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dataclasses import dataclass
from src.client import client
import time
import uvicorn


@dataclass
class QueryRequest:
    file_data: list[tuple[str, str]]

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def home():
    return "Welcome to the Quickstart: War Audio Transcription & Translation example!"

@app.post("/api/process_audio")
async def schedule_workflow(request: QueryRequest):
    try:
        workflow_id = f"{int(time.time() * 1000)}-parent_workflow"
        
        runId = await client.schedule_workflow(
            workflow_name="ParentWorkflow",
            workflow_id=workflow_id,
            input={"file_data": request.file_data}
        )
        print("Scheduled workflow", runId)
        
        result = await client.get_workflow_result(
            workflow_id=workflow_id,
            run_id=runId
        )
        
        return {
            "result": result,
            "workflow_id": workflow_id,
            "run_id": runId
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
# Remove Flask-specific run code since FastAPI uses uvicorn
def run_app():
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == '__main__':
    run_app()
```


3. **Workflow System (Restack)**
   - Consists of two main workflows:
     - Parent Workflow: Orchestrates multiple file processing
     - Child Workflow: Handles individual file processing
   - Located in:
   
```1:24:src/workflows/parent.py
from restack_ai.workflow import workflow, log, workflow_info
from dataclasses import dataclass
from .child import ChildWorkflow

@dataclass
class WorkflowInputParams:
    file_data: list[tuple[str, str]]

@workflow.defn()
class ParentWorkflow:
    @workflow.run
    async def run(self, input: WorkflowInputParams):
        parent_workflow_id = workflow_info().workflow_id

        log.info("ParentWorkflow started", input=input)

        child_workflow_results = []
        for file_data in input.file_data:
            result = await workflow.child_execute(ChildWorkflow, workflow_id=f"{parent_workflow_id}-child-execute-{file_data[0]}", input=WorkflowInputParams(file_data=file_data))
            child_workflow_results.append(result)

        log.info("ParentWorkflow completed", results=child_workflow_results)

        return child_workflow_results
```

   
```1:45:src/workflows/child.py
from datetime import timedelta
from dataclasses import dataclass
from restack_ai.workflow import workflow, import_functions, log

with import_functions():
    from src.functions.transcribe import transcribe, FunctionInputParams as TranscribeFunctionInputParams
    from src.functions.translate import translate, FunctionInputParams as TranslationFunctionInputParams
    
@dataclass
class WorkflowInputParams:
    file_data: tuple[str, str]

@dataclass
class WorkflowOutputParams:
    transcription: str
    translation: str

@workflow.defn()
class ChildWorkflow:
    @workflow.run
    async def run(self, input: WorkflowInputParams):
        log.info("ChildWorkflow started", input=input)

        transcription = await workflow.step(
            transcribe,
            TranscribeFunctionInputParams(file_data=input.file_data),
            start_to_close_timeout=timedelta(seconds=120)
        )

        translation_prompt = f"""
        Instructions: Translate the following content to English. Output only the translated content.
        Content: {transcription['text']}
        """

        translation = await workflow.step(
            translate,
            TranslationFunctionInputParams(user_prompt=translation_prompt),
            start_to_close_timeout=timedelta(seconds=120)
        )

        log.info("ChildWorkflow completed", transcription=transcription['text'], translation=translation['content'])
        return WorkflowOutputParams(
            transcription=transcription['text'],
            translation=translation['content']
        )
```


4. **Processing Functions**
   - Transcription (using Groq's Whisper model)
   - Translation (using OpenBabylon)
   - Located in:
   
```1:35:src/functions/transcribe.py
from restack_ai.function import function, log
from dataclasses import dataclass
from groq import Groq
import os
import base64
@dataclass
class FunctionInputParams:
    file_data: tuple[str, str]

@function.defn()
async def transcribe(input: FunctionInputParams):
    try:
        log.info("transcribe function started", input=input)
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


        filename, base64_content = input.file_data
        file_bytes = base64.b64decode(base64_content)
        transcription = client.audio.transcriptions.create(
            file=(filename, file_bytes), # Required audio file
            model="whisper-large-v3-turbo", # Required model to use for transcription
            # Best practice is to write the prompt in the language of the audio, use translate.google.com if needed
            prompt=f"Опиши о чем речь в аудио",  # Translation: Describe what the audio is about
            language="ru", # Original language of the audio
            response_format="json",  # Optional
            temperature=0.0  # Optional
        )

        log.info("transcribe function completed", transcription=transcription)
        return transcription
        

    except Exception as e:
        log.error("transcribe function failed", error=e)
        raise e
```

   
```1:51:src/functions/translate.py
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
        if not api_url:
            raise ValueError("OPENBABYLON_API_URL environment variable is not set")

        # Add timeout and better client configuration
        client = OpenAI(
            api_key="openbabylon",
            base_url=api_url,
            timeout=30.0,  # Add timeout in seconds
        )

        messages = []
        if input.user_prompt:
            messages.append({"role": "user", "content": input.user_prompt})

        response = client.chat.completions.create(
            model="orpo-mistral-v0.3-ua-tokV2-focus-10B-low-lr-1epoch-aux-merged-1ep",
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
```


## Processing Flow

1. User uploads audio file(s) through Streamlit interface
2. Frontend sends files to FastAPI backend
3. Backend initiates Parent Workflow
4. For each file, Parent Workflow:
   - Spawns Child Workflow
   - Child Workflow executes:
     - Transcription (Russian audio → Russian text)
     - Translation (Russian text → English text)
5. Results are returned to frontend and displayed

## Setup and Deployment

### Prerequisites
- Python 3.12+
- Poetry for dependency management
- Docker for Restack services
- Required API keys:
  - GROQ_API_KEY
  - OpenBabylon URL (provided during hackathon)
  - RESTACK credentials

### Installation Steps

1. **Docker Setup**
```bash
docker run -d --pull always --name restack -p 5233:5233 -p 6233:6233 -p 7233:7233 ghcr.io/restackio/restack:main
```

2. **Environment Setup**
```bash
poetry env use 3.12
poetry shell
poetry install
```

3. **Configuration**
- Copy `.env.example` to `.env`
- Set required environment variables:
  - OPENBABYLON_API_URL
  - GROQ_API_KEY
  - OPENAI_API_KEY (dummy value for OpenBabylon)

4. **Running the Application**
- Start Restack services: `poetry run services`
- Start FastAPI backend: `poetry run app`
- Start Streamlit frontend: `poetry run streamlit run frontend.py`

## Deployment Configuration

The system includes Nginx configuration for production deployment:

```1:43:nginx.conf
worker_processes 1;

events {
    worker_connections 1024;
}

http {
    server {
        listen 80;

        # Route to FastAPI on /api
        location /api {
            proxy_pass http://localhost:8000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # Route to Streamlit on /
        location / {
            proxy_pass http://localhost:8501;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        location /_stcore/stream {
            proxy_pass http://localhost:8501/_stcore/stream;
            proxy_http_version 1.1;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header Host $host;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_read_timeout 86400;
        }
    }
}
```


This configuration handles:
- Routing `/api` requests to FastAPI
- Routing root requests to Streamlit
- WebSocket support for Streamlit's live updates

## Error Handling

The system includes comprehensive error handling:
- Frontend connection errors
- API processing errors
- Workflow execution errors
- Transcription/translation service errors

## Limitations

1. Currently supports primarily Russian audio inputs
2. Requires stable internet connection for API services
3. Processing time depends on audio file size and service availability

## Security Considerations

1. CORS is configured to allow all origins (should be restricted in production)
2. API keys should be properly secured in production
3. File size limits should be implemented for production use

This documentation provides an overview of the system's architecture and functionality. For specific implementation details, refer to the individual code files in the repository.