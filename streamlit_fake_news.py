import streamlit as st
import pandas as pd
import numpy as np
import spacy
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pyvis.network import Network
import json
import joblib
from datetime import datetime
import os
from collections import defaultdict
import re


class EnhancedFakeNewsDetector:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        self.triples_storage = []
        self.knowledge_graph = nx.DiGraph()
        self._cached_layout = None
        self._cached_graph_size = 0
        
    def extract_enhanced_entities(self, text):
        """Extract entities with enhanced categorization"""
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
    
    def extract_enhanced_relations(self, text):
        """Extract relations with enhanced triple extraction"""
        doc = self.nlp(text)
        triples = []
        
        
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ("ROOT", "relcl"):
                # Find subject
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
                    triple = {
                        'subject': subject,
                        'relation': token.lemma_,
                        'object': obj,
                        'confidence': 0.8,
                        'source_text': text[:100] + "..." if len(text) > 100 else text
                    }
                    triples.append(triple)
        
        
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ == "nsubj":
                for child in chunk.root.head.children:
                    if child.dep_ in ("attr", "acomp") and child.pos_ == "NOUN":
                        triple = {
                            'subject': chunk.text,
                            'relation': 'is_a',
                            'object': child.text,
                            'confidence': 0.6,
                            'source_text': text[:100] + "..." if len(text) > 100 else text
                        }
                        triples.append(triple)
        
        return triples
    
    def store_triples(self, triples, metadata=None):
        """Store triples with metadata"""
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
        # Invalidate cached layout if graph size changes
        self._cached_layout = None
        self._cached_graph_size = self.knowledge_graph.number_of_nodes() + self.knowledge_graph.number_of_edges()
    
    def get_triples_by_entity(self, entity):
        """Get all triples involving a specific entity"""
        return [t for t in self.triples_storage 
                if entity.lower() in t['subject'].lower() or 
                   entity.lower() in t['object'].lower()]
    
    def export_triples_json(self, filename="knowledge_triples.json"):
        """Export triples to JSON format"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.triples_storage, f, indent=2, ensure_ascii=False)
        return filename
    
    def export_triples_rdf(self, filename="knowledge_triples.ttl"):
        """Export triples to RDF/Turtle format"""
        from rdflib import Graph, Literal, Namespace, URIRef
        from rdflib.namespace import RDF, RDFS
        
        g = Graph()
        ns = Namespace("http://example.org/fakenews/")
        
        for triple in self.triples_storage:
            subject = URIRef(ns + triple['subject'].replace(' ', '_'))
            predicate = URIRef(ns + triple['relation'].replace(' ', '_'))
            obj = URIRef(ns + triple['object'].replace(' ', '_'))
            
            g.add((subject, predicate, obj))
            g.add((subject, RDF.type, ns.Entity))
            g.add((obj, RDF.type, ns.Entity))
        
        g.serialize(destination=filename, format='turtle')
        return filename

def main():
    st.set_page_config(
        page_title="Enhanced Fake News Detection System",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("📰 Enhanced Fake News Detection System")
    st.markdown("**Advanced NER, Relation Extraction, and Knowledge Graph Visualization**")
    
   
    if 'detector' not in st.session_state:
        st.session_state.detector = EnhancedFakeNewsDetector()
    
    
    try:
        model = joblib.load("fake_news_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        st.success("✅ Models loaded successfully!")
    except FileNotFoundError:
        st.error("❌ Models not found. Please run the training script first.")
        return
    
  
    st.sidebar.title("🔧 Controls")
    
   
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📝 Text Analysis")
        news_text = st.text_area(
            "Enter news text for analysis:",
            height=200,
            placeholder="Enter your news article or headline here..."
        )
        
        analyze_button = st.button("🔍 Analyze Text", type="primary")
    
    with col2:
        st.subheader("📊 Quick Stats")
        st.metric("Total Triples", len(st.session_state.detector.triples_storage))
        st.metric("Graph Nodes", st.session_state.detector.knowledge_graph.number_of_nodes())
        st.metric("Graph Edges", st.session_state.detector.knowledge_graph.number_of_edges())
    
    if analyze_button and news_text.strip():
        with st.spinner("Analyzing text..."):
            # Fake news prediction
            X = vectorizer.transform([news_text])
            pred = model.predict(X)[0]
            confidence = model.predict_proba(X)[0].max()
            
            # Display results
            st.subheader("🎯 Analysis Results")
            
            col_pred, col_conf = st.columns(2)
            with col_pred:
                if pred == 1:
                    st.success(f"**Prediction: TRUE NEWS** ✅")
                else:
                    st.error(f"**Prediction: FAKE NEWS** ❌")
            
            with col_conf:
                st.metric("Confidence", f"{confidence:.2%}")
            
            # Enhanced NER
            st.subheader("🏷️ Named Entity Recognition")
            entities = st.session_state.detector.extract_enhanced_entities(news_text)
            
            if entities:
                entity_df = pd.DataFrame(entities)
                st.dataframe(entity_df, use_container_width=True)
                
                # Entity visualization
                entity_counts = entity_df['label'].value_counts()
                fig = px.pie(values=entity_counts.values, names=entity_counts.index, 
                           title="Entity Type Distribution")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No entities found in the text.")
            
            # Enhanced Relation Extraction
            st.subheader("🔗 Relation Extraction & Triples")
            triples = st.session_state.detector.extract_enhanced_relations(news_text)
            
            if triples:
                # Store triples
                st.session_state.detector.store_triples(triples, {
                    'prediction': 'true' if pred == 1 else 'fake',
                    'confidence': confidence,
                    'text_length': len(news_text)
                })
                
                # Display triples
                triple_df = pd.DataFrame(triples)
                st.dataframe(triple_df, use_container_width=True)
                
                # Export options
                col_export1, col_export2 = st.columns(2)
                with col_export1:
                    if st.button("💾 Export as JSON"):
                        filename = st.session_state.detector.export_triples_json()
                        st.success(f"Exported to {filename}")
                
                with col_export2:
                    if st.button("📄 Export as RDF"):
                        filename = st.session_state.detector.export_triples_rdf()
                        st.success(f"Exported to {filename}")
                
                # Knowledge Graph Visualization
                st.subheader("🕸️ Knowledge Graph")
                # Limit graph size for performance
                MAX_GRAPH_NODES = 30
                graph = st.session_state.detector.knowledge_graph
                if graph.number_of_nodes() > MAX_GRAPH_NODES:
                    st.warning(f"Graph too large to display interactively (>{MAX_GRAPH_NODES} nodes). Showing first {MAX_GRAPH_NODES} nodes.")
                    sub_nodes = list(graph.nodes())[:MAX_GRAPH_NODES]
                    subgraph = graph.subgraph(sub_nodes)
                else:
                    subgraph = graph
                # Create interactive network
                net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
                for edge in subgraph.edges(data=True):
                    net.add_node(edge[0], label=edge[0], color="#ff6b6b")
                    net.add_node(edge[1], label=edge[1], color="#4ecdc4")
                    net.add_edge(edge[0], edge[1], label=edge[2].get('relation', ''), color="#95a5a6")
                net.save_graph("knowledge_graph.html")
                with open("knowledge_graph.html", "r", encoding="utf-8") as f:
                    html_content = f.read()
                st.components.v1.html(html_content, height=600)

                # Alternative: Plotly network visualization
                st.subheader("📈 Network Analysis")
                # Cache layout for performance
                detector = st.session_state.detector
                graph_size = subgraph.number_of_nodes() + subgraph.number_of_edges()
                if detector._cached_layout is None or detector._cached_graph_size != graph_size:
                    detector._cached_layout = nx.spring_layout(subgraph)
                    detector._cached_graph_size = graph_size
                pos = detector._cached_layout
                edge_x = []
                edge_y = []
                edge_info = []
                for edge in subgraph.edges():
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
                    edge_info.append(f"{edge[0]} → {edge[1]}")
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color='#888'),
                    hoverinfo='none',
                    mode='lines'
                )
                node_x = []
                node_y = []
                node_text = []
                for node in subgraph.nodes():
                    x, y = pos[node]
                    node_x.append(x)
                    node_y.append(y)
                    node_text.append(node)
                node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers+text',
                    hoverinfo='text',
                    text=node_text,
                    textposition="middle center",
                    marker=dict(
                        size=20,
                        color='lightblue',
                        line=dict(width=2, color='darkblue')
                    )
                )
                fig = go.Figure(data=[edge_trace, node_trace],
                             layout=go.Layout(
                                title='Knowledge Graph Visualization',
                                titlefont_size=16,
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=20,l=5,r=5,t=40),
                                annotations=[ dict(
                                    text="Interactive knowledge graph showing entity relationships",
                                    showarrow=False,
                                    xref="paper", yref="paper",
                                    x=0.005, y=-0.002,
                                    xanchor='left', yanchor='bottom',
                                    font=dict(color="blue", size=12)
                                )],
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                            ))
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("No relations found in the text.")
    
    # Knowledge Base Explorer
    st.subheader("🗄️ Knowledge Base Explorer")
    
    if st.session_state.detector.triples_storage:
        # Search functionality
        search_entity = st.text_input("🔍 Search for entity:", placeholder="Enter entity name...")
        
        if search_entity:
            matching_triples = st.session_state.detector.get_triples_by_entity(search_entity)
            if matching_triples:
                st.write(f"Found {len(matching_triples)} triples for '{search_entity}':")
                search_df = pd.DataFrame(matching_triples)
                st.dataframe(search_df, use_container_width=True)
            else:
                st.info(f"No triples found for '{search_entity}'")
        
        # Show all triples (limit display for performance)
        if st.checkbox("📋 Show All Stored Triples"):
            MAX_TRIPLES_DISPLAY = 100
            triples = st.session_state.detector.triples_storage
            if len(triples) > MAX_TRIPLES_DISPLAY:
                st.warning(f"Showing first {MAX_TRIPLES_DISPLAY} triples out of {len(triples)}.")
                triples = triples[:MAX_TRIPLES_DISPLAY]
            all_triples_df = pd.DataFrame(triples)
            st.dataframe(all_triples_df, use_container_width=True)
    
    # Clear data button
    if st.button("🗑️ Clear All Data", type="secondary"):
        st.session_state.detector.triples_storage = []
        st.session_state.detector.knowledge_graph.clear()
        st.success("All data cleared!")
        st.rerun()

if __name__ == "__main__":
    main()
