from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

client = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

video_id = "FWyiYzgPNi4"

# Extracting transcripts
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id=video_id, languages=["en"])
    transcript = " ".join([t.text for t in transcript_list])
    print(transcript)

except TranscriptsDisabled:
    transcript_list = []


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,    
)


# Splitting transcripts
chunks = splitter.create_documents(transcript)

len(chunks)

