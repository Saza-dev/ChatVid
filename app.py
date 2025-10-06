import os
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
import re

# Load env
ROOT = Path(__file__).parent
load_dotenv(ROOT / '.env')

# App imports
from ui.sidebar import sidebar_ui
from ui.chat_ui import chat_ui
from modules.video_processor import VideoProcessor
from modules.embedder import Embedder
from modules.chat_engine import ChatEngine

st.set_page_config(page_title="VidChat", layout="wide")

# Ensure data folders exist
for p in [ROOT / 'data' / 'uploads', ROOT / 'data' / 'audio', ROOT / 'data' / 'frames', ROOT / 'data' / 'chroma_db']:
    p.mkdir(parents=True, exist_ok=True)

st.title("ChatVID — Chat with your videos")

# --- Instantiation with UI settings ---
conf = sidebar_ui()

# Pass UI settings to the processor
vp = VideoProcessor()

embedder = Embedder(persist_directory=str(ROOT / 'data' / 'chroma_db'))
chat_engine = ChatEngine(embedder=embedder)

if 'processed_videos' not in st.session_state:
    st.session_state.processed_videos = set()

uploaded = conf.get('uploaded_file')

if uploaded is not None:
    uploads_dir = ROOT / 'data' / 'uploads'
    vid_path = uploads_dir / uploaded.name
    
    # Only save if not already saved
    if not vid_path.exists():
        with open(vid_path, "wb") as f:
            f.write(uploaded.getbuffer())
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.video(vid_path)

    # Sanitize filename for collection name
    file_stem = Path(vid_path).stem
    sanitized_stem = re.sub(r'[^a-zA-Z0-9._-]', '_', file_stem)
    collection_name = f"video_{sanitized_stem}"
    
    if collection_name not in st.session_state.processed_videos:
        with st.spinner('Processing video (this may take a while)...'):
            all_texts = []
            all_metadatas = []

            # process audio
            if vp.has_audio(vid_path):
                transcript, segments = vp.transcribe_audio(vid_path)
                if transcript:
                    all_texts.append(transcript)
                    all_metadatas.append({'source': uploaded.name, 'type': 'transcript'})
            
            # frames captioning
            captions = vp.caption_frames(vid_path, interval_sec=conf['fps_sample'])
            if captions:
                caption_text = "\n".join(captions)
                all_texts.append(caption_text)
                all_metadatas.append({'source': uploaded.name, 'type': 'visual_captions'})
            
            # saving texts
            if all_texts:
                docs = vp.build_documents(texts=all_texts, metadatas=all_metadatas)
                embedder.add_documents(collection_name=collection_name, documents=docs)
                st.success('Successfully indexed audio transcript and visual captions!')
            else:
                st.warning('Could not extract any text or captions from the video.')
        
        # Mark as processed
        st.session_state.processed_videos.add(collection_name)
    
    # Show chat UI
    chat_ui(chat_engine=chat_engine, collection_name=collection_name)

else:
    st.info('Upload a video from the sidebar to get started.')