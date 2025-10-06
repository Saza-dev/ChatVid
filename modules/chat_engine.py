import os
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

load_dotenv()

PROMPT_TEMPLATE = """
You are a helpful AI assistant for answering questions about a video.
Use the following pieces of retrieved context, which are from a video's audio transcript and visual scene descriptions, to answer the user's question.
Frame your answer from the perspective of someone who has watched the video. Use phrases like "In the video...", "The speaker mentions...", or "Visually, the scene shows...".
If you don't know the answer, just say that you don't know, don't try to make up an answer.

CONTEXT:
{context}

CHAT HISTORY:
{chat_history}

QUESTION:
{question}

ANSWER:
"""

class ChatEngine:
    def __init__(self, embedder):
        self.embedder = embedder
        self.llm = ChatGroq(
            groq_api_key=os.getenv("GROQ_API_KEY"),
            model=os.getenv("GROQ_MODEL"),
            temperature=0.3,
        )

    def get_chain(self, collection_name: str):
        retriever = self.embedder.get_retriever(collection_name)

        custom_prompt = PromptTemplate(
            template=PROMPT_TEMPLATE,
            input_variables=["context", "chat_history", "question"]
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": custom_prompt}
        )
        return chain

    def answer(self, collection_name: str, question: str, chat_history: list):
        chain = self.get_chain(collection_name)
        result = chain({"question": question, "chat_history": chat_history})
        return result