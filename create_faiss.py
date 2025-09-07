import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


load_dotenv()
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=os.getenv("GEMINI_API_KEY")
)


website_url = ["https://www.vaidrix.com/","https://www.vaidrix.com/about", "https://www.vaidrix.com/services", "https://www.vaidrix.com/projects", "https://www.vaidrix.com/blog","https://www.vaidrix.com/contact"]


loader = WebBaseLoader(web_paths=website_url)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(docs)

db = FAISS.from_documents(docs, embedding_model)

db.save_local("faiss_index")

print("FAISS index created with Gemini embeddings!")
