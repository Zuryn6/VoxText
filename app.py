import streamlit as st
import whisper
import os

st.set_page_config(page_title="VoxText", page_icon="🎙️")

st.title("🎙️ VoxText")
st.markdown("Upload an audio file and let AI do the dirty work.")

@st.cache_resource
def load_model():
    return whisper.load_model("medium")

model = load_model()

# Upload File
audio_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "m4a", "ogg"])

if audio_file is not None:
    # Show player audio
    st.audio(audio_file)
    
    if st.button("Transcribe Audio"):
        with st.spinner("Processing... This might take a minute."):
            with open("temp_audio.mp3", "wb") as f:
                f.write(audio_file.getbuffer())
            
            # Transcribe audio
            result = model.transcribe("temp_audio.mp3")
            
            # Result
            st.success("Transcription completed!")
            st.subheader("Extracted Text:")
            st.write(result["text"])
            
            # Download Option
            st.download_button(
                label="Scarica Trascrizione (.txt)",
                data=result["text"],
                file_name="trascrizione.txt",
                mime="text/plain"
            )
            
            # Pulizia
            os.remove("temp_audio.mp3")