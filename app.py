"""
RAG Knowledge Base - Mexican Cuisine
"""

import streamlit as st
import numpy as np
import pandas as pd

TOPIC_NAME = "Mexican Cuisine"
TOPIC_ICON = "🌮"

st.set_page_config(
    page_title=f"{TOPIC_NAME} Knowledge Base",
    page_icon=TOPIC_ICON,
    layout="wide",
)

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
    "Tacos are one of the most iconic dishes in Mexican cuisine. A taco consists of a folded or rolled tortilla filled with various ingredients such as beef, chicken, pork, fish, or vegetables. Tacos are typically topped with salsa, guacamole, cilantro, onion, and lime. They originated as street food in Mexico and are now eaten all over the world.",
    "Guacamole is a creamy avocado-based dip made by mashing ripe avocados with lime juice, salt, cilantro, onion, and tomato. Guacamole originated with the Aztecs in the 16th century. It is commonly served as a dip with tortilla chips or as a topping for tacos and other Mexican dishes.",
    "Mole is one of the most complex sauces in Mexican cuisine, made from chili peppers, spices, chocolate, and up to 30 different ingredients. The most famous version is mole negro from Oaxaca. It takes hours to prepare and is traditionally served over turkey or chicken.",
    "Tamales are made of masa dough spread on a corn husk, filled with meat or cheese, then folded and steamed. They have been made in Mexico for thousands of years dating back to the Aztecs and Mayans. Tamales are especially popular during Christmas and other celebrations.",
    "Tequila is made from the blue agave plant grown in the state of Jalisco. The agave takes 8 to 12 years to mature. Tequila must be made in Mexico to carry the name. Types include blanco, reposado, and anejo, which differ in aging time in oak barrels.",
    "Mexican cuisine varies greatly by region. Northern Mexico uses more beef and flour tortillas. The Yucatan Peninsula features Mayan-influenced dishes like cochinita pibil. Oaxaca is known as the land of seven moles and is one of the culinary capitals of Mexico.",
    "Mexico has produced world-renowned chefs like Enrique Olvera, born in 1976, whose restaurant Pujol ranks among the top 50 in the world. In 2010, UNESCO added traditional Mexican cuisine to its Intangible Cultural Heritage list.",
    "Enchiladas are corn tortillas rolled around meat or cheese and covered with chili sauce. They are topped with cheese and sour cream and baked in the oven until the cheese melts.",
    "Churros are fried pastry sticks rolled in cinnamon sugar and served with chocolate dipping sauce. They are popular street food at Mexican fairs and festivals.",
    "Horchata is a drink made from rice, water, cinnamon, and sugar. It is served cold and is popular with spicy food because it cools down the heat of chili peppers.",
    "Salsa is a fundamental condiment in Mexican cuisine. Types include salsa roja made with red tomatoes, salsa verde made with tomatillos, and pico de gallo with fresh tomato, onion, and cilantro.",
    "Mexican cooking uses traditional tools like the comal, a flat griddle for tortillas, and the molcajete, a stone mortar for grinding spices. Nixtamalization is an ancient process developed by the Aztecs over 3500 years ago.",
    "El taco es el simbolo mas reconocido de la cocina mexicana. Consiste en una tortilla de maiz o harina con rellenos como carne asada, pollo o verduras. Los tacos de calle son parte fundamental de la cultura culinaria mexicana.",
    "El chile es el ingrediente mas importante de la cocina mexicana. Existen mas de 60 variedades en Mexico, desde el suave poblano hasta el picante habanero. Son la base de salsas, moles y marinadas.",
    "La cocina mexicana fue declarada Patrimonio Cultural Inmaterial por la UNESCO en 2010. Los ingredientes principales son el maiz, el chile, el frijol y el jitomate.",
]

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource(show_spinner="Building search database...")
def build_store(_docs: tuple):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in _docs:
        chunks.extend(splitter.split_text(doc))
    store = Chroma.from_texts(
        texts=chunks,
        embedding=load_model(),
    )
    return store, chunks

st.sidebar.markdown(
    f'<div class="sidebar-banner">{TOPIC_ICON} {TOPIC_NAME}</div>',
    unsafe_allow_html=True
)
st.sidebar.markdown("*Your Mexican food search engine*")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Search", "Explore Chunks", "About & Stats"],
    label_visibility="collapsed"
)
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
    col1, col2, col3 = st.columns(3)
    col1.metric("Documents", len(DOCUMENTS))
    store, chunks = build_store(tuple(DOCUMENTS))
    col2.metric("Chunks", len(chunks))
    col3.metric("Embedding model", "Multilingual MiniLM")

elif page == "Search":
    st.title("🔍 Semantic Search")
    store, chunks = build_store(tuple(DOCUMENTS))

    query = st.text_input(
        "Ask a question in English or Spanish",
        placeholder="e.g. What is tequila made from? / Que es el taco?"
    )
    num_results = st.slider("Number of results", 1, 5, 3)

    if query:
        with st.spinner("Searching..."):
            results = store.similarity_search_with_score(query, k=num_results)
        st.subheader(f"Top {len(results)} results")
        scores = [score for _, score in results]
        min_score = min(scores)
        max_score = max(scores)
        for i, (doc, score) in enumerate(results, 1):
            if max_score == min_score:
                similarity = 1.0 if i == 1 else 0.5
            else:
                similarity = 1 - ((score - min_score) / (max_score - min_score + 0.0001))
            similarity = max(0.0, min(1.0, similarity))
            level = "high" if i == 1 else ("med" if i == 2 else "low")
            emoji = "🟢" if level == "high" else ("🟡" if level == "med" else "⚪")
            st.markdown(
                f'<div class="result-card relevance-{level}">'
                f'<small style="color:#888;">{emoji} Result {i} &nbsp;·&nbsp; relevance: {similarity:.0%}</small>'
                f'<p style="margin:8px 0 0;font-size:15px;line-height:1.6;">{doc.page_content}</p>'
                f'</div>',
                unsafe_allow_html=True
            )
    else:
        st.info("👆 Type a question above to search the knowledge base.")
    st.caption("Powered by paraphrase-multilingual-MiniLM-L12-v2 + ChromaDB")

elif page == "Explore Chunks":
    st.title("🔬 Explore Chunks")
    st.markdown("See how documents are split into chunks by the text splitter.")
    store, chunks = build_store(tuple(DOCUMENTS))

    lengths = [len(c) for c in chunks]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total chunks", len(chunks))
    c2.metric("Avg size", f"{np.mean(lengths):.0f} chars")
    c3.metric("Min size", f"{min(lengths)} chars")
    c4.metric("Max size", f"{max(lengths)} chars")

    st.markdown("---")
    st.subheader("Chunk length distribution")
    st.bar_chart(pd.DataFrame({"Chunk length (chars)": lengths}))

    st.markdown("---")
    st.subheader("All chunks")
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"Chunk {i} — {len(chunk)} chars"):
            st.text(chunk)

elif page == "About & Stats":
    st.title("📊 About & Statistics")

    st.subheader("Chunking Strategy")
    st.markdown("""
    This app uses **chunk_size = 300 characters** with **chunk_overlap = 50 characters**.

    **Why these values?**
    - 300 chars is large enough to contain a complete thought
    - 50 chars overlap prevents cutting sentences mid-thought
    - Smaller chunks (200) are more precise but lose context
    - Larger chunks (500) have more context but are less precise
    """)

    store, chunks = build_store(tuple(DOCUMENTS))
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    s200 = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    s500 = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
    c200 = []
    c500 = []
    for doc in DOCUMENTS:
        c200.extend(s200.split_text(doc))
        c500.extend(s500.split_text(doc))

    st.markdown("---")
    st.subheader("Chunk Size Comparison Chart")
    chart_data = pd.DataFrame({
        "Chunk Size": ["Size 200 (small)", "Size 300 (current)", "Size 500 (large)"],
        "Number of Chunks": [len(c200), len(chunks), len(c500)]
    })
    st.bar_chart(chart_data.set_index("Chunk Size"))

    st.markdown("---")
    st.subheader("About this app")
    st.markdown(f"""
    - **Topic:** {TOPIC_NAME}
    - **Documents:** {len(DOCUMENTS)} (12 English + 3 Spanish)
    - **Embedding model:** paraphrase-multilingual-MiniLM-L12-v2
    - **Languages supported:** English and Spanish
    - **Vector DB:** ChromaDB
    - **Framework:** Streamlit + LangChain
    - **Deployment:** Render.com
    """)