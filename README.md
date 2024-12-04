# USS Hornet Defense Tech Hackathon Quickstart: War Audio Transcription & Translation

The main project is a multi-modal intelligence feed that integrates real-time data from several sources and classifies them according to NATO Admiralty code to analyze dynamic geopolitical events.

<img width="1440" alt="Screenshot 2024-11-24 at 10 55 00 PM" src="https://github.com/user-attachments/assets/6ebe351a-f3a1-4898-b9de-3a4e85311504">

Tech stack used: Python, Streamlit, FastApi, AssemblyAI, OpenAI, OpenBabylon, Restack AI

This is the audio transcription / translation microservice of the larger multimodal data feed project. <br>
The AI workflow will need an audio file as an input and will transcribe it & translate it to English.

-------

Screenshots

1. Data Feed Dashboard

<img width="1440" alt="Screenshot 2024-11-24 at 8 04 03 PM" src="https://github.com/user-attachments/assets/b9a794ed-bede-4858-8394-8be066a8d035">

2. Audio Microservice

<img width="1440" alt="Screenshot 2024-11-24 at 8 03 49 PM" src="https://github.com/user-attachments/assets/a8a2eb5d-05fe-444c-bb99-a5765e07cfd5">


--------------

## Datasets

Find audio samples at https://drive.google.com/drive/folders/1mbchTGfmhq2sc7sQEMfx-dQzd11kWIfO?usp=drive_link

## OpenBabylon credentials

During the hackathon, OpenBabylon provided a public url:

OPENBABYLON_API_URL=64.139.222.109:80
No api key is needed, although a dummy api_key="openbabylon" is necessary for openai sdk.

## Prerequisites

- Python 3.12 or higher
- Poetry (for dependency management)
- Docker (for running Restack services)

## Usage

1. Run Restack local engine with Docker:

   ```bash
   docker run -d --pull always --name restack -p 5233:5233 -p 6233:6233 -p 7233:7233 ghcr.io/restackio/restack:main
   ```

2. Open the Web UI to see the workflows:

   ```bash
   http://localhost:5233
   ```

3. Clone this repository:

   ```bash
   git clone https://github.com/restackio/examples-python.git
   cd examples/defense_quickstart_audio_transcription_translation
   ```

4. Setup virtual environment with Poetry:

   ```bash
   poetry env use 3.12
   poetry shell
   poetry install
   poetry env info # Optional: copy the interpreter path to use in your IDE (e.g. Cursor, VSCode, etc.)
   ```

5. Set up your environment variables:

   Copy `.env.example` to `.env` and add your OpenBabylon API URL:

   ```bash
   cp .env.example .env
   # Edit .env and add your:
   # OPENBABYLON_API_URL
   # GROQ_API_KEY
   # OPENAI_API_KEY - Set this to a random string as OpenBabylon uses OpenAI API
   ```

6. Run the services:

   ```bash
   poetry run services
   ```

   This will start the Restack service with the defined workflows and functions.

7. In a new terminal, run FastAPI app:

   ```bash
   poetry run app
   ```

8. In a new terminal, run the Streamlit frontend

   ```bash
   poetry run streamlit run frontend.py
   ```
