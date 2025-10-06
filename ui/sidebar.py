import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

def sidebar_ui():
    st.sidebar.header('ChatVid — Upload & Settings')
    uploaded = st.sidebar.file_uploader('Upload a video', type=['mp4','mov','mkv','webm'])
    fps_sample = st.sidebar.slider('Frames per second', 1, 2, 1)

    return {
        'uploaded_file': uploaded,
        'fps_sample': fps_sample
    }
