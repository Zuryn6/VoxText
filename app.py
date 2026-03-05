import streamlit as st
import whisper
import noisereduce as nr
import librosa
import soundfile as sf
import os
import numpy as np

st.set_page_config(page_title="VoxText", page_icon="🎙️")

st.title("🎙️ VoxText")
st.markdown("Upload an audio file and let AI do the dirty work.")


model_name = st.selectbox(
    "Choose the Whisper model",
    ["tiny", "base", "small", "medium", "large"],
    index=3,
)

language = st.sidebar.selectbox(
    "Audio Language",
    ["it", "en", "es", "fr", "de", "auto"],
    index=0,
    help="Forcing the language can improve accuracy and reduce 'hallucination' errors."
)

apply_denoise = st.sidebar.checkbox("Apply Noise Reduction", value=True)

@st.cache_resource
def load_model(model_name: str):
    return whisper.load_model(model_name)

model = load_model(model_name)

def preprocess_audio(input_path):
    # Upload audio 
    data, sr = librosa.load(input_path, sr=16000)
    
    if apply_denoise:
        # Noise reduction
        data = nr.reduce_noise(y=data, sr=sr, prop_decrease=0.75)
    
    # Normalization
    data = librosa.util.normalize(data)
    
    # Save cleaned audio
    clean_path = "cleaned_audio.wav"
    sf.write(clean_path, data, sr)
    return clean_path

# Upload File
audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"])

if audio_file is not None:
    # Show player audio
    st.audio(audio_file)
    
    if st.button("Transcribe Audio"):
        with st.spinner("Processing... This might take a minute."):
            with open("temp_audio.mp3", "wb") as f:
                f.write(audio_file.getbuffer())
            
            # Preprocess audio
            clean_audio_path = preprocess_audio("temp_audio.mp3")
            
            # Transcribe audio
            lang_param = None if language == "auto" else language
            result = model.transcribe(clean_audio_path, language=lang_param)
            
            # Result
            st.success("Transcription completed!")
            st.subheader("Extracted Text:")
            st.write(result["text"])
            
            # Download Option
            st.download_button(
                label="Download Transcription (.txt)",
                data=result["text"],
                file_name="transcription.txt",
                mime="text/plain"
            )
            
            # Pulizia
            os.remove("temp_audio.mp3")
            os.remove(clean_audio_path)