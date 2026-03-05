# VoxText

A Streamlit app to transcribe audio files with Whisper.

## Features

- Upload audio files (`mp3`, `wav`, `m4a`, `ogg`)
- Select the Whisper model from a dropdown menu:
  - `tiny`
  - `base`
  - `small`
  - `medium`
  - `large`
- Automatic transcription
- Download the transcription as a `.txt` file

## Requirements

- Python 3.10+
- `ffmpeg` installed on the system (required by Whisper)

Main Python dependencies (see `requirements.txt`):

- `streamlit`
- `openai-whisper`
- `torch`
- `ffmpeg-python`

## Installation

```bash
python -m venv trascrizione
source trascrizione/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Open the browser at the address shown by Streamlit (usually `http://localhost:8501`).

## Usage

1. Choose a Whisper model from the dropdown menu.
2. Upload an audio file.
3. Click `Transcribe Audio`.
4. View the transcribed text.
5. Download it with `Scarica Trascrizione (.txt)`.

## Notes

- Larger models (`medium`, `large`) are usually more accurate but slower.
- If available, Whisper can use the GPU through PyTorch.
