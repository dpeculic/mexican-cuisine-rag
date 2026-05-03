import streamlit as st
import numpy as np
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

TOPIC_NAME = "Mexican Cuisine"
TOPIC_ICON = "🌮"

st.set_page_config(page_title=f"{TOPIC_NAME} Knowledge Base", page_icon=TOPIC_ICON, layout="wide")

st.markdown("""
<style>
    .main { background-color: #fdf6ec; }
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #c0392b;
        padding: 10px 16px;
        font-size: 16px;
    }
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 20px 24px;
        margin: 12px 0;
        border-left: 5px solid #4CAF50;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .relevance-high { border-left-color: #27ae60; background: #f0fff4; }
    .relevance-med  { border-left-color: #e67e22; background: #fff8f0; }
    .relevance-low  { border-left-color: #95a5a6; background: #f8f8f8; }
    h1 { color: #c0392b; }
    h2 { color: #2c3e50; }
    .sidebar-banner {
        background: linear-gradient(135deg, #006847, #ffffff, #ce1126);
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        color: white;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 10px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
    }
</style>
""", unsafe_allow_html=True)

DOCUMENTS = [
    "Tacos are one of the most iconic dishes in Mexican cuisine. A taco consists of a folded tortilla filled with beef, chicken, pork, fish, or vegetables. Tacos are topped with salsa, guacamole, cilantro, onion, and lime. They originated as street food in Mexico.",
    "Guacamole is made by mashing ripe avocados with lime juice, salt, cilantro, onion, and tomato. It originated with the Aztecs in the 16th century. It is served as a dip with tortilla chips or as a topping for tacos.",
    "Mole is a complex sauce made from chili peppers, spices, and chocolate with up to 30 ingredients. The most famous is mole negro from Oaxaca. It takes hours to prepare and is served over turkey or chicken.",
    "Tamales are made of masa dough on a corn husk, filled with meat or cheese, then steamed. They have been made in Mexico for thousands of years. Tamales are popular during Christmas and celebrations.",
    "Tequila is made from the blue agave plant in Jalisco. The agave takes 8 to 12 years to mature. Types include blanco, reposado, and anejo, which differ in aging time in oak barrels.",
    "Mexican cuisine varies by region. Northern Mexico uses beef and flour tortillas. Oaxaca is known for seven moles. The Yucatan has Mayan dishes like cochinita pibil.",
    "Enrique Olvera opened Pujol in Mexico City in 2000, ranking among the top 50 restaurants in the world. In 2010 UNESCO added Mexican cuisine to its Intangible Cultural Heritage list.",
    "El taco es el simbolo mas reconocido de la cocina mexicana. Consiste en una tortilla con rellenos como carne asada, pollo o verduras. Los tacos de calle son fundamentales en la cultura mexicana.",
    "El chile es el ingrediente mas importante de la cocina mexicana. Existen mas de 60 variedades, desde el suave poblano hasta el picante habanero. Son la base de salsas y moles.",
    "La cocina mexicana fue declarada Patrimonio Cultural por la UNESCO en 2010. Los ingredientes principales son el maiz, el chile, el frijol y el jitomate.",
]

def chunk_text(text, size=200, overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks

@st.cache_resource(show_spinner="Building search database...")
def build_store(_docs: tuple):
    ef = embedding_functions.DefaultEmbeddingFunction()
    client = chromadb.Client()
    collection = client.create_collection("knowledge_base", embedding_function=ef)
    chunks = []
    ids = []
    for doc in _docs:
        doc_chunks = chunk_text(doc)
        for chunk in doc_chunks:
            chunks.append(chunk)
            ids.append(f"chunk_{len(ids)}")
    collection.add(documents=chunks, ids=ids)
    return collection, chunks

st.sidebar.markdown(f'<div class="sidebar-banner">{TOPIC_ICON} {TOPIC_NAME}</div>', unsafe_allow_html=True)
st.sidebar.markdown("*Your Mexican food search engine*")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["Home", "Search", "Explore Chunks", "About & Stats"], label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption(f"{len(DOCUMENTS)} documents loaded")

if page == "Home":
    st.title(f"{TOPIC_ICON} {TOPIC_NAME} Knowledge Base")

    col_img, col_text = st.columns([1, 2])
    with col_img:
        st.image("https://flagcdn.com/w320/mx.png", caption="Mexico", use_container_width=True)
    with col_text:
        st.markdown("""
        Welcome! This app lets you **search Mexican Cuisine knowledge by meaning**, not just keywords.

        Type any question and the app will find the most relevant passages even if they
        do not contain the exact words you typed.

        This app supports both **English and Spanish** queries!

        ### How it works
        1. Documents are split into small **chunks**
        2. Each chunk is converted into an **embedding**
        3. Chunks are stored in a **ChromaDB** vector database
        4. When you search, your question is compared to every chunk
        5. The closest matches are returned
        """)

    st.markdown("---")
    st.subheader("Explore Mexican Food")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background:#fff3e0;border-radius:16px;padding:30px;text-align:center;font-size:80px;">
        🌮<br><span style="font-size:18px;font-weight:bold;color:#c0392b;">Tacos</span>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background:#e8f5e9;border-radius:16px;padding:30px;text-align:center;font-size:80px;">
        🥑<br><span style="font-size:18px;font-weight:bold;color:#27ae60;">Guacamole</span>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background:#fce4ec;border-radius:16px;padding:30px;text-align:center;font-size:80px;">
        🫔<br><span style="font-size:18px;font-weight:bold;color:#c0392b;">Tamales</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    collection, chunks = build_store(tuple(DOCUMENTS))
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(DOCUMENTS))
    col2.metric("Chunks", len(chunks))
    col3.metric("Embedding model", "ChromaDB Default")

elif page == "Search":
    st.title("🔍 Semantic Search")
    collection, chunks = build_store(tuple(DOCUMENTS))

    query = st.text_input("Ask a question in English or Spanish", placeholder="e.g. What is tequila made from? / Que es el taco?")
    num_results = st.slider("Number of results", 1, 5, 3)

    if query:
        with st.spinner("Searching..."):
            results = collection.query(query_texts=[query], n_results=num_results)

        st.subheader(f"Top {num_results} results")
        docs = results["documents"][0]
        distances = results["distances"][0]
        min_d = min(distances)
        max_d = max(distances)
        for i, (doc, dist) in enumerate(zip(docs, distances), 1):
            if max_d == min_d:
                similarity = 1.0 if i == 1 else 0.5
            else:
                similarity = 1 - ((dist - min_d) / (max_d - min_d + 0.0001))
            similarity = max(0.0, min(1.0, similarity))
            level = "high" if i == 1 else ("med" if i == 2 else "low")
            emoji = "🟢" if level == "high" else ("🟡" if level == "med" else "⚪")
            st.markdown(
                f'<div class="result-card relevance-{level}">'
                f'<small style="color:#888;">{emoji} Result {i} &nbsp;·&nbsp; relevance: {similarity:.0%}</small>'
                f'<p style="margin:8px 0 0;font-size:15px;line-height:1.6;">{doc}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("👆 Type a question above to search the knowledge base.")

    st.caption("Powered by ChromaDB + Sentence Transformers")

elif page == "Explore Chunks":
    st.title("🔬 Explore Chunks")
    collection, chunks = build_store(tuple(DOCUMENTS))

    lengths = [len(c) for c in chunks]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total chunks", len(chunks))
    c2.metric("Avg size", f"{np.mean(lengths):.0f} chars")
    c3.metric("Min size", f"{min(lengths)} chars")
    c4.metric("Max size", f"{max(lengths)} chars")

    st.markdown("---")
    st.subheader("Chunk length distribution")
    chart_df = pd.DataFrame({"Chunk length (chars)": lengths})
    st.bar_chart(chart_df)

    st.markdown("---")
    st.subheader("All chunks")
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"Chunk {i} — {len(chunk)} chars"):
            st.text(chunk)

elif page == "About & Stats":
    st.title("📊 About & Statistics")

    st.subheader("Chunking Strategy")
    st.markdown("""
    This app uses **chunk size = 200 characters** with an overlap of 20 characters.

    **Why chunk size 200?**
    - Small enough to be precise and focused
    - Large enough to contain meaningful context
    - Overlap of 20 chars prevents cutting sentences mid-thought

    **Comparison with larger chunks:**
    - Size 200: more chunks, more precise results
    - Size 500: fewer chunks, more context per result
    """)

    chunks_small = []
    chunks_large = []
    for doc in DOCUMENTS:
        chunks_small.extend(chunk_text(doc, size=200))
        chunks_large.extend(chunk_text(doc, size=500))

    st.markdown("---")
    st.subheader("Chunk Size Comparison Chart")
    chart_data = pd.DataFrame({
        "Chunk Size": ["Size 200 (current)", "Size 500 (larger)"],
        "Number of Chunks": [len(chunks_small), len(chunks_large)]
    })
    st.bar_chart(chart_data.set_index("Chunk Size"))

    st.markdown("---")
    st.subheader("About this app")
    st.markdown(f"""
    - **Topic:** {TOPIC_NAME}
    - **Documents:** {len(DOCUMENTS)} (7 English + 3 Spanish)
    - **Embedding model:** ChromaDB Default (lightweight ONNX)
    - **Languages supported:** English and Spanish
    - **Vector DB:** ChromaDB
    - **Framework:** Streamlit + LangChain
    - **Deployment:** Render.com
    """)