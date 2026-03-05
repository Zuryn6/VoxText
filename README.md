# VoxText

VoxText is a Streamlit app for audio transcription with OpenAI Whisper, with built-in audio preprocessing and waveform preview.

## Features

- Upload audio files (`mp3`, `wav`, `m4a`, `ogg`)
- Choose Whisper model: `tiny`, `base`, `small`, `medium`, `large`
- Set input language (`it`, `en`, `es`, `fr`, `de`, `auto`)
- Tune preprocessing with sidebar controls:
  - `Noise Reduction Intensity`
  - `Transcription Complexity (Beam Size)`
- Add optional topic keywords via `initial_prompt` to improve recognition of technical terms
- Automatic audio preprocessing:
  - spectral noise reduction
  - pre-emphasis (high-pass)
  - normalization
- Side-by-side original vs cleaned audio preview
- Waveform comparison chart (raw vs optimized)
- Download transcription as `.txt`

## Requirements

- Python 3.10+
- `ffmpeg` installed on the system (required by Whisper)

Main Python dependencies (see `requirements.txt`):

- `streamlit`
- `openai-whisper`
- `torch`
- `ffmpeg-python`
- `noisereduce`
- `librosa`
- `soundfile`
- `numpy`
- `matplotlib`

## Installation

```bash
python -m venv VoxText
```

Windows (PowerShell):

```powershell
.\VoxText\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
source VoxText/bin/activate
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open the URL shown by Streamlit (usually `http://localhost:8501`).

## Usage

1. Select the Whisper model.
2. Set language and sidebar options (`Noise Reduction Intensity`, `Beam Size`, optional keywords).
3. Upload an audio file.
4. Review original and cleaned audio plus waveform comparison.
5. Click `Start <MODEL> Transcription`.
6. Review the transcript and download it with `Download Transcript (.txt)`.

## Notes

- Higher `Beam Size` may improve difficult transcriptions but increases runtime.
- More aggressive noise reduction can remove background noise but may also affect voice clarity.
- Larger models (`medium`, `large`) are usually more accurate but slower.
- If available, Whisper can use GPU through PyTorch.
