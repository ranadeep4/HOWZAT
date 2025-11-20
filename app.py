from flask import Flask, request, render_template, jsonify
from pathlib import Path
import joblib
import traceback
import pandas as pd
import os
from dotenv import load_dotenv

# Defer importing heavy preprocessing modules until needed (avoid import-time sklearn/scipy issues)

# Create Flask app
app = Flask(__name__, template_folder='templates')


def load_model():
    art = Path('artifacts')
    tuned = art / 'RandomForest_tuned.joblib'
    default = art / 'RandomForest.joblib'
    if tuned.exists():
        return joblib.load(tuned)
    if default.exists():
        return joblib.load(default)
    raise FileNotFoundError('No RandomForest model found in artifacts/')


MODEL = None
ARTIFACTS = None

# --- RAG feature lazy state ---
RAG_LOADED = False
RAG_ERROR = None
_emb = None
_vectorstore = None
_retriever = None
_llm = None

load_dotenv()

def ensure_rag_loaded():
    """Attempt to lazily import and initialize FAISS embeddings, vectorstore and the LLM.

    This function is resilient: if required packages are missing it will set
    `RAG_ERROR` with a helpful message instead of raising during app import.
    """
    global RAG_LOADED, RAG_ERROR, _emb, _vectorstore, _retriever, _llm
    if RAG_LOADED or RAG_ERROR:
        return
    try:
        # Import third-party langchain components only when the RAG page is used
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import SentenceTransformerEmbeddings
        from langchain_groq import ChatGroq

        # Load FAISS index and embeddings
        emb = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        # The index directory (relative to the ragfeature folder) — keep same name as before
        index_path = Path('ragfeature') / 'faiss_index'
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index folder not found at '{index_path.resolve()}'")
        vectorstore = FAISS.load_local(str(index_path), emb, allow_dangerous_deserialization=True)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        api_key = os.getenv('GROQ_API_KEY')
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=api_key)

        _emb = emb
        _vectorstore = vectorstore
        _retriever = retriever
        _llm = llm
        RAG_LOADED = True
    except Exception as e:
        RAG_ERROR = str(e)


def answer_query_rag(query: str) -> str:
    """Use the loaded retriever + LLM to answer a query. Returns string or raises."""
    if not RAG_LOADED:
        ensure_rag_loaded()
    if RAG_ERROR:
        raise RuntimeError(f"RAG not available: {RAG_ERROR}")
    docs = _retriever.invoke(query)
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

    response = _llm.invoke(prompt)
    return response.content


def ensure_loaded():
    """Lazily load preprocessing artifacts and model on first request.

    This avoids compatibility issues with Flask versions that may not expose
    the `before_first_request` attribute in the same way.
    """
    global MODEL, ARTIFACTS
    # Import preprocessing artifacts lazily to avoid import-time sklearn issues
    if ARTIFACTS is None:
        try:
            from src.preprocess import load_artifacts as _load_artifacts
            ARTIFACTS = _load_artifacts()
        except Exception:
            ARTIFACTS = {}
    if MODEL is None:
        MODEL = load_model()


def get_dropdown_options():
    """Return dict with lists for teams, venues, batsmen and bowlers extracted from artifacts."""
    ensure_loaded()
    opts = {'bat_team': [], 'bowl_team': [], 'venue': [], 'batsman': [], 'bowler': []}
    ohe = ARTIFACTS.get('onehot') if ARTIFACTS else None
    if ohe is not None:
        # feature order expected: ['bat_team','bowl_team','venue']
        try:
            cats = list(ohe.categories_)
            if len(cats) >= 3:
                opts['bat_team'] = list(cats[0])
                opts['bowl_team'] = list(cats[1])
                opts['venue'] = list(cats[2])
        except Exception:
            pass

    te = ARTIFACTS.get('target_encode') if ARTIFACTS else {}
    # target_encode keys are like 'batsman' and 'bowler' mapping meta
    if te:
        for col in ['batsman', 'bowler']:
            meta = te.get(col)
            if meta and isinstance(meta.get('mapping'), dict):
                opts[col] = sorted(list(meta['mapping'].keys()))

    return opts


@app.route('/', methods=['GET'])
def index():
    # Simple form page
    # load dropdown options from artifacts
    opts = get_dropdown_options()
    return render_template('index.html', options=opts)


def build_input_dataframe(form: dict) -> pd.DataFrame:
    # Expected keys: bat_team, bowl_team, venue, runs, wickets, overs, striker, batsman, bowler
    df = pd.DataFrame([{
        'bat_team': form.get('bat_team', ''),
        'bowl_team': form.get('bowl_team', ''),
        'venue': form.get('venue', ''),
        'runs': float(form.get('runs', 0)),
        'wickets': float(form.get('wickets', 0)),
        'overs': float(form.get('overs', 0.0)),
        'striker': int(form.get('striker', 0)),
        'batsman': form.get('batsman', ''),
        'bowler': form.get('bowler', '')
    }])
    return df


@app.route('/predict', methods=['POST'])
def predict():
    """Handle both form and JSON requests. Validate inputs, preprocess and predict.

    - For form requests: render `index.html` with `prediction` or `errors` context.
    - For JSON requests: return JSON response with `predicted_total` or errors and appropriate HTTP status.
    """
    # Ensure artifacts and model are loaded for dropdowns and prediction
    ensure_loaded()
    is_json = request.is_json
    raw = request.get_json() if is_json else request.form.to_dict()

    # Validation
    errors = []

    def add_err(msg):
        errors.append(msg)

    # Required string fields
    for k in ['bat_team', 'bowl_team', 'venue', 'batsman', 'bowler']:
        v = raw.get(k)
        if v is None or str(v).strip() == '':
            add_err(f"'{k}' is required.")

    # Numeric fields validation
    try:
        runs = float(raw.get('runs', 0))
        if runs < 0:
            add_err('Runs must be >= 0')
    except Exception:
        add_err('Runs must be a number')

    try:
        wickets = int(float(raw.get('wickets', 0)))
        if wickets < 0 or wickets > 10:
            add_err('Wickets must be between 0 and 10')
    except Exception:
        add_err('Wickets must be an integer')

    try:
        overs = float(raw.get('overs', 0.0))
        if overs < 0 or overs > 20:
            add_err('Overs must be between 0 and 20')
        # Business rule: require at least 5 overs of data
        if overs < 5.0:
            add_err('At least 5 overs of data are required to make a prediction')
    except Exception:
        add_err('Overs must be a number (e.g. 12.3)')

    try:
        striker = int(float(raw.get('striker', 0)))
        if striker not in (0, 1):
            add_err('Striker must be 0 or 1')
    except Exception:
        add_err('Striker must be 0 or 1')

    # If errors, return appropriately
    if errors:
        if is_json:
            return jsonify({'errors': errors}), 400
        else:
            # Render form with errors preserved and include dropdown options
            opts = get_dropdown_options()
            return render_template('index.html', errors=errors, form=raw, options=opts)

    # Build dataframe and predict
    try:
        df = build_input_dataframe(raw)
        try:
            from src.preprocess import transform_with_artifacts
            X = transform_with_artifacts(df, ARTIFACTS)
        except Exception:
            # If preprocessing cannot be imported/used, raise
            raise
        pred = MODEL.predict(X)
        val = float(pred[0])
        rounded = round(val, 2)
        if is_json:
            return jsonify({'predicted_total': rounded})
        else:
            opts = get_dropdown_options()
            return render_template('index.html', prediction=rounded, form=raw, options=opts)
    except Exception as e:
        tb = traceback.format_exc()
        if is_json:
            return jsonify({'error': str(e)}), 500
        else:
            opts = get_dropdown_options()
            return render_template('index.html', errors=[str(e)], form=raw, options=opts)


@app.route('/rag', methods=['GET', 'POST'])
def rag():
    """Render RAG QA page and handle query submissions.

    GET: show the form. POST: run retrieval + LLM and render answer.
    """
    error = None
    answer = None
    form = None
    if request.method == 'POST':
        form = request.form.to_dict()
        q = form.get('query', '').strip()
        if q == '':
            error = 'Please enter a question.'
        else:
            try:
                answer = answer_query_rag(q)
            except Exception as e:
                error = str(e)

    return render_template('rag.html', answer=answer, error=error, form=form)


if __name__ == '__main__':
    # Run dev server
    app.run(host='0.0.0.0', port=5000, debug=True)
