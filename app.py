import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42  # deterministic langdetect results

st.set_page_config(page_title="Multilingual News Classifier", page_icon="📰", layout="centered")

LANGDETECT_MAP = {
    "en": "english",
    "hi": "hindi",
    "bn": "bengali",
}

CATEGORY_LABELS = {
    "politics_national": "Politics / National",
    "international": "International",
    "sports": "Sports",
    "entertainment": "Entertainment",
    "technology": "Technology",
}


@st.cache_resource(show_spinner="Loading LaBSE model (first run only)...")
def load_embedder():
    return SentenceTransformer("sentence-transformers/LaBSE")


@st.cache_resource(show_spinner="Loading classifiers...")
def load_classifiers():
    multilingual_clf = joblib.load("multilingual_clf.joblib")
    zeroshot_clf = joblib.load("zeroshot_clf.joblib")
    return multilingual_clf, zeroshot_clf


@st.cache_data(show_spinner="Loading reference headlines...")
def load_reference_data():
    ref_df = pd.read_csv("reference_data.csv")
    ref_embeddings = np.load("reference_embeddings.npy")
    return ref_df, ref_embeddings


def detect_language(text: str) -> str:
    try:
        code = detect(text)
    except Exception:
        return "unknown"
    return LANGDETECT_MAP.get(code, code)


def get_nearest_neighbors(query_embedding, ref_df, ref_embeddings, exclude_lang=None, top_k=3):
    results = {}
    languages = ["english", "hindi", "bengali"]
    if exclude_lang:
        languages = [l for l in languages if l != exclude_lang]

    for lang in languages:
        mask = ref_df["language"] == lang
        lang_idxs = ref_df[mask].index.to_numpy()
        lang_embs = ref_embeddings[lang_idxs]

        sims = cosine_similarity(query_embedding.reshape(1, -1), lang_embs)[0]
        top_idxs = lang_idxs[np.argsort(sims)[::-1][:top_k]]

        results[lang] = [
            {
                "text": ref_df.loc[i, "text_clean"],
                "category": ref_df.loc[i, "category_merged"],
                "similarity": sims[np.where(lang_idxs == i)[0][0]],
            }
            for i in top_idxs
        ]
    return results


def main():
    st.title("📰 Multilingual News Classifier")
    st.caption("English · Hindi · Bengali — one shared model, no translation step")

    embedder = load_embedder()
    multilingual_clf, zeroshot_clf = load_classifiers()
    ref_df, ref_embeddings = load_reference_data()

    st.markdown(
        "Paste a news headline in **English, Hindi, or Bengali**. "
        "The app detects the language, embeds it with LaBSE (a shared multilingual "
        "sentence encoder), and classifies its topic — no translation involved."
    )

    model_choice = st.radio(
        "Choose which trained model to use:",
        options=["multilingual", "zero-shot"],
        format_func=lambda x: (
            "Multilingual model — trained on English + Hindi + Bengali"
            if x == "multilingual"
            else "Zero-shot model — trained on English + Hindi ONLY (Bengali never seen during training)"
        ),
        help=(
            "Use 'zero-shot' with a Bengali headline to see how well the model "
            "generalizes to a language it was never trained on."
        ),
    )

    text = st.text_area("Headline", placeholder="e.g. ভারত বনাম অস্ট্রেলিয়া: সিরিজ জয় নিশ্চিত করল ভারত", height=100)

    show_neighbors = st.checkbox("Show similar headlines in other languages", value=True)

    if st.button("Classify", type="primary") and text.strip():
        detected_lang = detect_language(text)

        with st.spinner("Embedding and classifying..."):
            query_embedding = embedder.encode([text], normalize_embeddings=True)[0]

            clf = multilingual_clf if model_choice == "multilingual" else zeroshot_clf
            pred = clf.predict(query_embedding.reshape(1, -1))[0]
            proba = clf.predict_proba(query_embedding.reshape(1, -1))[0]
            confidence = proba.max()

        st.divider()
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Detected language", detected_lang.capitalize())
        with col2:
            st.metric("Predicted category", CATEGORY_LABELS.get(pred, pred))

        st.progress(float(confidence), text=f"Confidence: {confidence:.1%}")

        if model_choice == "zero-shot" and detected_lang == "bengali":
            st.info(
                "This prediction was made by a model that never saw a single labeled "
                "Bengali example during training — it's relying entirely on LaBSE's "
                "shared multilingual embedding space to generalize from English and Hindi."
            )

        if show_neighbors:
            st.divider()
            st.subheader("Similar headlines in other languages")
            st.caption("Nearest neighbors by cosine similarity in the shared embedding space")

            neighbors = get_nearest_neighbors(
                query_embedding, ref_df, ref_embeddings, exclude_lang=detected_lang
            )
            cols = st.columns(len(neighbors))
            for col, (lang, items) in zip(cols, neighbors.items()):
                with col:
                    st.markdown(f"**{lang.capitalize()}**")
                    for item in items:
                        st.markdown(
                            f"<div style='font-size:0.85em; padding:6px 0; border-bottom:1px solid #333;'>"
                            f"[{CATEGORY_LABELS.get(item['category'], item['category'])}] "
                            f"({item['similarity']:.2f})<br>{item['text']}</div>",
                            unsafe_allow_html=True,
                        )
    elif text.strip() == "":
        st.info("Enter a headline above and click Classify.")


if __name__ == "__main__":
    main()
