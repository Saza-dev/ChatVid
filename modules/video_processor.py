import os
import ffmpeg
from groq import Groq
from dotenv import load_dotenv
from typing import Tuple, List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from modules.framer_captioner import FrameCaptioner

load_dotenv()

class VideoProcessor:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.captioner = FrameCaptioner()
        # Initialize a text splitter for creating document chunks
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def has_audio(self, video_path: str) -> bool:
        """Checks if the video file contains an audio stream."""
        try:
            probe = ffmpeg.probe(video_path)
            audio_streams = [stream for stream in probe['streams'] if stream['codec_type'] == 'audio']
            return len(audio_streams) > 0
        except ffmpeg.Error as e:
            print(f"Error probing video: {e.stderr}")
            return False

    def extract_audio(self, video_path: str) -> str:
        """Extract audio track from video and save as a .wav file"""
        audio_dir = "data/audio"
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, os.path.basename(video_path).split(".")[0] + ".wav")
        
        try:
            (
                ffmpeg.input(video_path)
                .output(audio_path, format="wav", ac=1, ar="16k")
                .overwrite_output()
                .run(quiet=True)
            )
            return audio_path
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e.stderr}")
            return None

    def transcribe_audio(self, video_path: str) -> Tuple[str, list]:
        """Extracts audio and transcribes it, returning text and segments."""
        audio_path = self.extract_audio(video_path)
        if not audio_path:
            return "", []
            
        with open(audio_path, "rb") as f:
            response = self.client.audio.transcriptions.create(
                model=os.getenv('AUD_MODEL'),
                file=f,
                response_format="verbose_json" 
            )
        
        full_transcript = response.text
        segments = response.segments
        # Clean up the temp audio file
        os.remove(audio_path)
        return full_transcript, segments

    def caption_frames(self, video_path: str, interval_sec: int = 10) -> List[str]:
        """Extracts keyframes and generates a caption for each one."""
        frames_dir = "data/frames"
        video_stem = os.path.basename(video_path).split('.')[0]
        
        # Create a subdirectory for the specific video's frames
        video_frames_dir = os.path.join(frames_dir, video_stem)
        os.makedirs(video_frames_dir, exist_ok=True)

        output_pattern = os.path.join(video_frames_dir, f"{video_stem}_%04d.jpg")

        try:
            (
                ffmpeg.input(video_path)
                .filter("fps", fps=f"1/{interval_sec}")
                .output(output_pattern, qscale=2)
                .overwrite_output()
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            print(f"Error extracting frames: {e.stderr}")
            return []

        frame_files = sorted(
            [os.path.join(video_frames_dir, f) for f in os.listdir(video_frames_dir) if f.endswith(".jpg")]
        )
        
        captions = []
        for frame_path in frame_files:
            caption = self.captioner.caption_image(frame_path)
            captions.append(caption)
            # clean up frame file after captioning
            os.remove(frame_path)
            
        return captions

    def build_documents(self, texts: List[str], metadatas: List[dict]) -> List[Document]:
        """Builds and chunks LangChain documents from text and metadata."""
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if i < len(metadatas) else metadatas[0]
            chunks = self.text_splitter.split_text(text)
            for j, chunk in enumerate(chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_index'] = j
                doc = Document(page_content=chunk, metadata=chunk_metadata)
                documents.append(doc)
        return documents