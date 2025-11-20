import os
from flask import Flask, request, render_template_string
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_groq import ChatGroq

load_dotenv()

# Load FAISS index
emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("faiss_index", emb, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    api_key=api_key
)

def answer_query(query):
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
You are an expert IPL analyst with deep knowledge of IPL history, teams, players and statistics.

Below is retrieved context from the IPL knowledge base. 
Use it **ONLY if it is relevant** to the user's question.

If the answer is NOT present in the context, you may use your own general cricket/IPL knowledge to answer accurately.

If the question is about a player who does NOT exist in the context, answer using your own knowledge (do not say "not found" unless the player is fictional).

If the context is irrelevant, IGNORE IT completely.

---
CONTEXT:
{context}
---

QUESTION: {query}

Now give the **best possible answer**, citing stats if available, and keep it factual and clear.
"""


    response = llm.invoke(prompt)
    return response.content

# Flask UI
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>IPL RAG - Groq Powered</title>
    <style>
        body { font-family: Arial; margin: 30px; background: #f3f3f3; }
        .box { background: white; padding: 20px; border-radius: 8px; max-width: 700px; margin: auto; }
        textarea { width: 100%; height: 100px; padding: 10px; }
        button { padding: 10px 20px; margin-top: 10px; background-color: #007bff; color: white; border: none; border-radius: 5px; }
        button:hover { background-color: #0056b3; cursor: pointer; }
        .answer { margin-top: 20px; white-space: pre-wrap; background: #eee; padding: 15px; border-radius: 8px; }
    </style>
</head>
<body>
    <h1 style="text-align:center;">IPL RAG System (Groq Llama3-70B)</h1>
    <div class="box">
        <form method="POST">
            <textarea name="query" placeholder="Ask any IPL question..."></textarea>
            <br>
            <button type="submit">Ask</button>
        </form>
        {% if answer %}
        <div class="answer">
            <h3>Answer:</h3>
            {{ answer }}
        </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    ans = ""
    if request.method == "POST":
        q = request.form["query"]
        ans = answer_query(q)
    return render_template_string(HTML, answer=ans)

if __name__ == "__main__":
    app.run(debug=True)
