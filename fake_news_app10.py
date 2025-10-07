import streamlit as st
import pandas as pd
import spacy
import networkx as nx
import joblib
from datetime import datetime
import json
from transformers import pipeline  # For question answering

class EnhancedFakeNewsDetector:
    def __init__(self):
        # Load spaCy model once
        self.nlp = spacy.load("en_core_web_sm")
        self.triples_storage = []
        self.knowledge_graph = nx.DiGraph()
        self._cached_layout = None
        self._cached_graph_size = 0
        
    # Named Entity Recognition
    def extract_enhanced_entities(self, text):
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            })
        return entities
    
    # Relation Extraction
    def extract_enhanced_relations(self, text):
        doc = self.nlp(text)
        triples = []
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "relcl"):
                subject = None
                for child in token.lefts:
                    if child.dep_ in ("nsubj", "nsubjpass", "nsubj:xsubj"):
                        subject = child.text
                        break
                obj = None
                for child in token.rights:
                    if child.dep_ in ("dobj", "pobj", "obj"):
                        obj = child.text
                        break
                if subject and obj:
                    triples.append({
                        'subject': subject,
                        'relation': token.lemma_,
                        'object': obj,
                        'confidence': 0.8,
                        'source_text': text[:100] + "..." if len(text) > 100 else text
                    })
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ == "nsubj":
                for child in chunk.root.head.children:
                    if child.dep_ in ("attr", "acomp") and child.pos_ == "NOUN":
                        triples.append({
                            'subject': chunk.text,
                            'relation': 'is_a',
                            'object': child.text,
                            'confidence': 0.6,
                            'source_text': text[:100] + "..." if len(text) > 100 else text
                        })
        return triples
    
    # Store triples in memory and graph
    def store_triples(self, triples, metadata=None):
        timestamp = datetime.now().isoformat()
        for triple in triples:
            triple_entry = {
                'id': f"triple_{len(self.triples_storage)}_{timestamp}",
                'subject': triple['subject'],
                'relation': triple['relation'],
                'object': triple['object'],
                'confidence': triple['confidence'],
                'source_text': triple['source_text'],
                'timestamp': timestamp,
                'metadata': metadata or {}
            }
            self.triples_storage.append(triple_entry)
            self.knowledge_graph.add_edge(
                triple['subject'], 
                triple['object'], 
                relation=triple['relation'],
                confidence=triple['confidence'],
                weight=triple['confidence']
            )
        self._cached_layout = None
        self._cached_graph_size = self.knowledge_graph.number_of_nodes() + self.knowledge_graph.number_of_edges()
    
    # Export triples as JSON
    def export_triples_json(self, filename="knowledge_triples.json"):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.triples_storage, f, indent=2, ensure_ascii=False)
        return filename

def main():
    st.set_page_config(page_title="Enhanced Fake News Detection System", page_icon="📰", layout="wide")
    st.title("📰 Enhanced Fake News Detection System")
    st.markdown("**Advanced NER, Relation Extraction, Knowledge Graph & Question Answering**")
    
    # Initialize detector once
    if 'detector' not in st.session_state:
        st.session_state.detector = EnhancedFakeNewsDetector()
    
    # Load trained fake news model & vectorizer
    try:
        if 'model' not in st.session_state:
            st.session_state.model = joblib.load("fake_news_model.pkl")
            st.session_state.vectorizer = joblib.load("tfidf_vectorizer.pkl")
        st.success("✅ Models loaded successfully!")
    except FileNotFoundError:
        st.error("❌ Models not found. Please run the training script first.")
        return
    
    # Load Question Answering model once
    if 'qa_pipeline' not in st.session_state:
        st.session_state.qa_pipeline = pipeline(
            "question-answering", 
            model="distilbert-base-cased-distilled-squad"
        )
    
    # User inputs
    news_text = st.text_area("Enter news text for analysis:", height=200)
    question = st.text_input("Ask a question about this news:")
    
    if st.button("Analyze"):
        if news_text.strip():
            # --- Fake News Detection ---
            X = st.session_state.vectorizer.transform([news_text])
            pred = st.session_state.model.predict(X)[0]
            confidence = st.session_state.model.predict_proba(X)[0].max()
            
            st.subheader("🎯 Analysis Results")
            if pred == 1:
                st.success(f"**Prediction: TRUE NEWS** ✅")
            else:
                st.error(f"**Prediction: FAKE NEWS** ❌")
            st.metric("Confidence", f"{confidence:.2%}")
            
            # --- Named Entity Recognition ---
            entities = st.session_state.detector.extract_enhanced_entities(news_text)
            if entities:
                st.subheader("🏷️ Named Entity Recognition")
                st.dataframe(pd.DataFrame(entities), use_container_width=True)
            
            # --- Relation Extraction ---
            triples = st.session_state.detector.extract_enhanced_relations(news_text)
            if triples:
                st.subheader("🔗 Relation Extraction & Triples")
                st.session_state.detector.store_triples(triples, {
                    'prediction': 'true' if pred == 1 else 'fake',
                    'confidence': confidence,
                    'text_length': len(news_text)
                })
                st.dataframe(pd.DataFrame(triples), use_container_width=True)
            
            # --- Question Answering ---
            if question.strip():
                answer = st.session_state.qa_pipeline(question=question, context=news_text)
                st.subheader("❓ Question Answering")
                st.write(f"**Question:** {question}")
                st.write(f"**Answer:** {answer['answer']} (Confidence: {answer['score']:.2%})")

if __name__ == "__main__":
    main()
