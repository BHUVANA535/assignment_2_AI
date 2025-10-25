from flask import Flask, render_template, request
import PyPDF2, json, os, numpy as np, re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import LatentDirichletAllocation
from itertools import chain
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# ---------------- Load pre-extracted data ----------------
base_dir = "extracted_data"
with open(os.path.join(base_dir, "author_texts_pdfminer.json"), 'r', encoding='utf-8') as f:
    authors_texts = json.load(f)
with open(os.path.join(base_dir, "authors_keywords.json"), 'r', encoding='utf-8') as f:
    authors_keywords = json.load(f)
with open(os.path.join(base_dir, "references_dataset.json"), 'r', encoding='utf-8') as f:
    authors_references = json.load(f)

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    uploaded_file = request.files['paper']
    if not uploaded_file:
        return "No file uploaded."

    # 1️⃣ Extract text from PDF
    reader = PyPDF2.PdfReader(uploaded_file)
    input_text = "".join([p.extract_text() or "" for p in reader.pages])

    # 2️⃣ Basic keyword extraction
    words = re.findall(r'\b\w+\b', input_text.lower())
    input_keywords = [w for w, _ in zip(words, range(20))]

    # 3️⃣ Reference extraction
    total_pages = len(reader.pages)
    ref_text = "".join([p.extract_text() or "" for p in reader.pages[int(0.9 * total_pages):]])
    input_references = re.findall(r'\b\d{4}\b|[A-Z][a-z]+ et al\.', ref_text)

    # 4️⃣ Topic modeling
    count_vectorizer = CountVectorizer(stop_words='english')
    input_count = count_vectorizer.fit_transform([input_text])
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    input_topic_dist = lda.fit_transform(input_count)[0]


    # 5️⃣ BERT embedding
    input_emb = bert_model.encode([input_text])[0]

    # 6️⃣ Compute all similarities
    final_scores = {}
    for author, papers in authors_texts.items():
        # Text similarity (TF-IDF)
        corpus = papers + [input_text]
        tfidf_vec = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf_vec.fit_transform(corpus)
        text_sims = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
        text_score = np.mean(text_sims)

        # Keyword similarity
        author_kw = list(chain.from_iterable(authors_keywords.get(author, [])))
        inter = len(set(input_keywords) & set(author_kw))
        uni = len(set(input_keywords) | set(author_kw))
        keyword_score = inter / uni if uni > 0 else 0

        # Reference similarity
        author_refs = list(chain.from_iterable(authors_references.get(author, {}).values()))
        inter = len(set(input_references) & set(author_refs))
        uni = len(set(input_references) | set(author_refs))
        ref_score = inter / uni if uni > 0 else 0

        # Topic similarity
        topic_scores = []
        for p in papers:
            cnt_vec = count_vectorizer.fit_transform([p])
            topic_dist = lda.transform(cnt_vec)[0]
            topic_scores.append(cosine_similarity([input_topic_dist], [topic_dist])[0][0])
            
        topic_score = np.mean(topic_scores) if topic_scores else 0

        # BERT similarity
        author_embs = [bert_model.encode([p])[0] for p in papers]
        bert_sims = [cosine_similarity([input_emb], [emb])[0][0] for emb in author_embs]
        bert_score = np.mean(bert_sims) if bert_sims else 0

        # Final score (equal weights)
        final_scores[author] = np.mean([text_score, keyword_score, ref_score, topic_score, bert_score])

    # 7️⃣ Rank authors
    top_k = 5
    sorted_authors = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return render_template('results.html', results=sorted_authors)
    
if __name__ == '__main__':
    app.run(debug=True)
