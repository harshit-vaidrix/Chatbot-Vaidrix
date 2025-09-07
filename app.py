import os
from dotenv import load_dotenv
from flask import Flask, request, render_template, session
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings


load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")


embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=os.getenv("GEMINI_API_KEY")
)

db = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 20})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY")
)

CUSTOM_PROMPT = PromptTemplate(
    template="""You are Vaidrix, a helpful virtual assistant for Vaidrix website.
You must only use the provided context to answer user questions.
If the user asks about something outside the website (e.g., math problems, general knowledge, 
or anything not related to the context), politely refuse by saying:
"I can only answer questions related to Vaidrix and its services."

Rules:
- Do not guess or invent information outside the given context.
- Do not answer questions unrelated to the Vaidrix website.
- If the context does not contain the answer, politely refuse.
- Be careful about sensitive or harmful words. Always remain professional.

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"],
)


qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": CUSTOM_PROMPT}
)

@app.route("/")
def index():
    session.clear()
    return render_template("chatbot.html")

@app.route("/ask", methods=["POST"])
def ask_query():
    data = request.get_json()

    query = data.get("query")

    conversation_history = session.get("conversation_history", [])

    result = qa.invoke({"query": query})

    conversation_history.append({"role": "user", "content": query})
    conversation_history.append({"role": "bot", "content": result["result"]})
    session["conversation_history"] = conversation_history

    return {
        "conversation_history": conversation_history,
        "bot_greeting": "Hello! I'm Vaidrix, your virtual assistant. How can I help you today?"
    }, 200


if __name__ == '__main__':
    app.run(debug=True)
