import streamlit as st
import whisper
import noisereduce as nr
import librosa
import soundfile as sf
import os
import numpy as np
import matplotlib.pyplot as plt

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

st.sidebar.divider()
st.sidebar.subheader("🎚️ Audio Filters")

# Slider for Noise Reduction
prop_decrease = st.sidebar.slider(
    "Noise Reduction Intensity", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.45, 
    step=0.05,
    help="Higher values remove more noise but may distort the voice. 0.4-0.6 is ideal for lectures."
)

# Slider for Beam Size
beam_size = st.sidebar.slider(
    "Transcription Complexity (Beam Size)", 
    min_value=1, 
    max_value=10, 
    value=5,
    help="Higher values increase accuracy on difficult audio but make transcription slower."
)

initial_prompt = st.sidebar.text_area(
    "Topic or Keywords:",
    placeholder="E.g.: Calculus 2 Lecture, partial derivatives, surface integral, Gauss theorem...",
    help="Entering technical terms helps the AI recognize them correctly even if the audio is muffled."
)

@st.cache_resource
def load_model(model_name: str):
    return whisper.load_model(model_name)

model = load_model(model_name)

def preprocess_audio(input_path, noise_val):
    data, sr = librosa.load(input_path, sr=16000)
    
    # Apply Noise Reduction based on slider
    if noise_val > 0:
        data = nr.reduce_noise(y=data, sr=sr, prop_decrease=noise_val)
    
    # High-pass filter
    data = librosa.effects.preemphasis(data)
    
    # Normalization
    data = librosa.util.normalize(data)
    
    clean_path = "temp_cleaned.wav"
    sf.write(clean_path, data, sr)
    return data, sr, clean_path

def plot_waveforms(raw_data, clean_data, sr):
    fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(10, 4))
    
    # Original Waveform
    librosa.display.waveshow(raw_data, sr=sr, ax=ax[0], color='gray')
    ax[0].set(title='Original Waveform (Raw)')
    ax[0].label_outer()

    # Cleaned Waveform
    librosa.display.waveshow(clean_data, sr=sr, ax=ax[1], color='blue')
    ax[1].set(title='Optimized Waveform (AI Input)')
    
    plt.tight_layout()
    return fig

# Upload File
audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"])

if audio_file is not None:
    # 1. Save temporary raw file
    with open("temp_raw.mp3", "wb") as f:
        f.write(audio_file.getbuffer())
    
    # 2. INTERACTIVE PRE-PROCESSING & VISUALIZATION
    with st.spinner("Analyzing and cleaning audio..."):
        raw_data, sr = librosa.load("temp_raw.mp3", sr=16000)
        clean_data, _, clean_path = preprocess_audio("temp_raw.mp3", prop_decrease)
        
        st.divider()
        st.subheader("📊 Audio Signal Analysis")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info("Original Audio")
            st.audio(audio_file)
        with col2:
            st.success("Cleaned Audio (Optimized for AI)")
            st.audio(clean_path)

        # Show Waveform Comparison
        st.pyplot(plot_waveforms(raw_data, clean_data, sr))

    # 3. TRANSCRIPTION BUTTON
    st.divider()
    st.info("Check the optimized waveform and audio above. If it sounds clear, proceed to transcription.")
    
    if st.button(f"🚀 Start {model_name.upper()} Transcription"):
        model = load_model(model_name)
        
        with st.spinner("AI is transcribing... This might take a while depending on the file length."):
            lang_param = None if language == "auto" else language
            
            result = model.transcribe(
                clean_path, 
                language=lang_param, 
                beam_size=beam_size, 
                initial_prompt=initial_prompt,
                condition_on_previous_text=True
            )
            
            st.success("Transcription Completed!")
            st.subheader("Extracted Text:")
            st.text_area("Final Transcript", result["text"], height=400)
            
            # Download Button
            st.download_button(
                label="Download Transcript (.txt)",
                data=result["text"],
                file_name="lecture_transcript.txt",
                mime="text/plain"
            )

    # Cleanup temporary files (optional, but good practice)
    os.remove("temp_raw.mp3")
    os.remove(clean_path)