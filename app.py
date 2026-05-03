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
    "Tacos are one of the most iconic dishes in Mexican cuisine. A taco consists of a folded or rolled tortilla filled with various ingredients such as beef, chicken, pork, fish, or vegetables. The tortilla can be either soft or hard. Tacos are typically topped with salsa, guacamole, cilantro, onion, and lime. They originated as street food in Mexico and are now eaten all over the world.",
    "Guacamole is a creamy avocado-based dip that is a staple of Mexican cuisine. It is made by mashing ripe avocados and mixing them with lime juice, salt, cilantro, onion, and tomato. Some recipes also add jalapeno for spice. Guacamole originated with the Aztecs, who first made it in the 16th century. It is commonly served as a dip with tortilla chips or as a topping for tacos.",
    "Enchiladas are a traditional Mexican dish consisting of corn tortillas rolled around a filling and covered with a savory sauce. The filling can include meat, cheese, beans, or vegetables. The sauce is typically made from chili peppers and can be red, green, or brown. Enchiladas are usually topped with cheese, sour cream, and onions. They are baked in the oven until the cheese melts.",
    "Mole is one of the most complex and celebrated sauces in Mexican cuisine. It is made from a blend of chili peppers, spices, chocolate, and many other ingredients, sometimes up to 30 different components. The most famous version is mole negro from Oaxaca. Mole takes hours or even days to prepare properly. It is traditionally served over turkey or chicken with rice on the side.",
    "Tamales are a traditional Mesoamerican dish made of masa, which is a dough made from corn. The masa is spread on a corn husk, filled with meat, cheese, or chilies, then folded and steamed. Tamales have been made in Mexico for thousands of years, dating back to ancient Aztec and Mayan civilizations. They are especially popular during holidays and celebrations like Christmas.",
    "Pozole is a traditional Mexican soup made with hominy corn and meat, usually pork or chicken. It is slow-cooked for hours to develop a rich, deep flavor. Pozole is typically served with toppings such as shredded cabbage, radishes, lime, oregano, and chili flakes. There are three main varieties: red, white, and green, depending on the sauce used. It is a popular dish at celebrations.",
    "Churros are a popular Mexican fried pastry made from a simple dough of flour, water, and salt. The dough is piped through a star-shaped nozzle and fried until golden and crispy. Churros are then rolled in cinnamon sugar and served with a thick chocolate dipping sauce. They are a common street food in Mexico and are especially popular at fairs and festivals.",
    "Tequila is Mexico's most famous alcoholic drink, made from the blue agave plant grown primarily in the state of Jalisco. The agave plant takes 8 to 12 years to mature before it can be harvested. Tequila must be made in Mexico to legally carry the name. There are several types including blanco, reposado, and anejo, which differ in how long they are aged in oak barrels.",
    "Mexican cuisine varies greatly by region. In northern Mexico, the cuisine is heavily influenced by cattle ranching, so beef and cheese are central ingredients. Flour tortillas are more common in the north, while corn tortillas dominate in the south. The Yucatan Peninsula has its own distinct cuisine influenced by Mayan traditions, featuring dishes like cochinita pibil. Oaxaca is known as the land of seven moles.",
    "Mexican cooking uses several traditional techniques and tools that have been passed down for thousands of years. The comal is a flat griddle used to cook tortillas and roast chiles. The molcajete is a stone mortar and pestle used to grind spices and make salsas. Nixtamalization is an ancient process of soaking corn in an alkaline solution to make masa dough for tortillas and tamales.",
    "Mexico has produced several world-renowned chefs who have brought Mexican cuisine to international attention. Enrique Olvera, born in 1976 in Mexico City, opened his restaurant Pujol in 2000, which has consistently ranked among the top 50 restaurants in the world. In 2010, UNESCO added traditional Mexican cuisine to its list of Intangible Cultural Heritage of Humanity.",
    "El taco es el simbolo mas reconocido de la cocina mexicana en el mundo. Consiste en una tortilla de maiz o harina doblada o enrollada con diversos rellenos como carne asada, pollo, pescado o verduras. Los tacos de calle son una parte fundamental de la cultura culinaria mexicana y se pueden encontrar en cada esquina de las ciudades mexicanas.",
    "El chile es el ingrediente mas importante y representativo de la cocina mexicana. Existen mas de 60 variedades de chiles en Mexico, desde los mas suaves como el poblano hasta los mas picantes como el habanero. Los chiles se usan frescos, secos, ahumados o en polvo. Son la base de salsas, moles, adobos y marinadas en toda la cocina mexicana.",
    "La cocina mexicana fue declarada Patrimonio Cultural Inmaterial de la Humanidad por la UNESCO en 2010. Es una de las pocas cocinas del mundo que ha recibido este reconocimiento. La cocina mexicana destaca por su gran diversidad de ingredientes, tecnicas de preparacion y sabores regionales. Los ingredientes principales incluyen el maiz, el chile, el frijol y el jitomate.",
]

CHUNK_SIZE_A = 200
CHUNK_SIZE_B = 500

@st.cache_resource(show_spinner="Loading AI model...")
def load_model():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"}, encode_kwargs={"normalize_embeddings": False})

@st.cache_resource(show_spinner="Building search database...")
def build_store(_docs: tuple, chunk_size: int = 300):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size * 0.15),
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in _docs:
        chunks.extend(splitter.split_text(doc))
    store = Chroma.from_texts(
        texts=chunks,
        embedding=load_model(),
        collection_name=f"kb_{chunk_size}",
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
        placeholder="e.g. What are the most famous Mexican dishes? / Que es el taco?"
    )
    num_results = st.slider("Number of results", 1, 8, 3)

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
        st.info("👆 Type a question above in English or Spanish to search the knowledge base.")

    st.caption("Powered by all-MiniLM-L6-v2 + ChromaDB")

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
    chart_df = pd.DataFrame({"Chunk length (chars)": lengths})
    st.bar_chart(chart_df)

    st.markdown("---")
    st.subheader("All chunks")
    for i, chunk in enumerate(chunks, 1):
        with st.expander(f"Chunk {i} — {len(chunk)} chars"):
            st.text(chunk)

elif page == "About & Stats":
    st.title("📊 About & Statistics")

    st.subheader("Chunking Strategy Comparison")
    st.markdown("This app was tested with two different chunk sizes to compare search quality.")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Chunk size = 200 chars")
        st.markdown("""
        - More chunks created
        - Results are more precise
        - Less surrounding context
        - Better for specific fact queries
        """)
        store_a, chunks_a = build_store(tuple(DOCUMENTS), CHUNK_SIZE_A)
        st.metric("Chunks created", len(chunks_a))

    with col2:
        st.markdown("### Chunk size = 500 chars")
        st.markdown("""
        - Fewer larger chunks
        - More surrounding context
        - May include irrelevant text
        - Better for broad topic queries
        """)
        store_b, chunks_b = build_store(tuple(DOCUMENTS), CHUNK_SIZE_B)
        st.metric("Chunks created", len(chunks_b))

    st.markdown("---")
    st.subheader("Live comparison")
    test_query = st.text_input(
        "Try a query on both chunk sizes at once:",
        placeholder="e.g. How is tequila made?"
    )
    if test_query:
        r_a = store_a.similarity_search_with_score(test_query, k=1)
        r_b = store_b.similarity_search_with_score(test_query, k=1)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Best match (size=200)**")
            if r_a:
                st.info(r_a[0][0].page_content)
        with col2:
            st.markdown("**Best match (size=500)**")
            if r_b:
                st.info(r_b[0][0].page_content)

    st.markdown("---")
    st.subheader("Chunk Size Comparison Chart")
    chart_data = pd.DataFrame({
        "Chunk Size": ["Size 200 (small)", "Size 300 (default)", "Size 500 (large)"],
        "Number of Chunks": [len(chunks_a), len(chunks_a), len(chunks_b)]
    })
    st.bar_chart(chart_data.set_index("Chunk Size"))

    st.markdown("---")
    st.subheader("About this app")
    st.markdown(f"""
    - **Topic:** {TOPIC_NAME}
    - **Documents:** {len(DOCUMENTS)} (11 English + 3 Spanish)
    - **Embedding model:** all-MiniLM-L6-v2
    - **Languages supported:** English and Spanish
    - **Vector DB:** ChromaDB
    - **Framework:** Streamlit + LangChain
    - **Deployment:** Render.com
    """)