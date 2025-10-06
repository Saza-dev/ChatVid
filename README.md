#  ChatVid — Chat with Your Videos 💬🎬

VidChatRAG is a **Retrieval-Augmented Generation (RAG)** powered **Streamlit app** that lets you **chat with any video you upload**.
It automatically extracts audio, transcribes it using **whisper-large-v3**, generates **frame captions** using **llama-4-scout**, and stores both text and visual embeddings in **ChromaDB** for intelligent question answering. The RAG Chatbot is powered by **openai/gpt-oss-120b**.

---

## 🚀 Features

✅ **Upload any video** format (MP4, MOV, etc.).  
✅ **Automatic audio extraction** using **FFmpeg**.  
✅ **High-accuracy transcription** via **Whisper**.  
✅ **Detailed frame captioning** with **llama-4-scout**.  
✅ **Efficient chunking & embedding** using **Sentence Transformers**.  
✅ **Fast vector search** powered by **ChromaDB**.  
✅ **Intelligent RAG Chatbot** using the **Groq** LLM API for real-time responses.  
✅ **User-friendly UI** built with **Streamlit**, including video preview and a chat interface.

---

## ⚙️ Setup Instructions

### 1️⃣ **Clone the Repository**
Clone the project to your local machine.
```bash
git clone https://github.com/Saza-dev/ChatVid.git
cd ChatVid
````

### 2️⃣ **Create a Virtual Environment**

It's recommended to use a virtual environment to manage dependencies.

```bash
# Using conda
conda create -n vidchatrag python=3.10 -y
conda activate vidchatrag

# Or using venv
# python -m venv venv
# source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3️⃣ **Install Dependencies**

Install all the required Python packages.

```bash
pip install -r requirements.txt
```

### 4️⃣ **Configure Environment Variables**

Create a `.env` file in the root directory of the project and add your API key and configuration details.

```env
EMBEDDING_MODEL=all-MiniLM-L6-v2 
AUD_MODEL=whisper-large-v3
CAP_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b
```

### 5️⃣ **Run the Application**

Launch the Streamlit app.

```bash
streamlit run app.py
```

-----

## 🧩 Requirements

  * **Python 3.10+**
  * **FFmpeg**: Must be installed and accessible from your system's PATH.
      * You can verify your installation by running: `ffmpeg -version`
  * **Groq API Key**: Get a free API key from the [Groq Console](https://console.groq.com/).
