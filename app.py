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
    "Tacos are one of the most iconic and beloved dishes in Mexican cuisine, recognized worldwide as a symbol of Mexican food culture. A taco consists of a folded or rolled tortilla filled with various ingredients such as beef, chicken, pork, fish, or vegetables. The tortilla can be either soft corn or hard flour, depending on the region and preference. Tacos are typically topped with fresh salsa, creamy guacamole, cilantro, diced onion, and a squeeze of lime juice. They originated as street food in Mexico City and surrounding areas, where vendors would sell them from small carts and stands. Today, tacos are eaten all over the world and come in hundreds of regional variations, from the fish tacos of Baja California to the barbacoa tacos of central Mexico.",

    "Guacamole is a creamy, rich avocado-based dip that is a staple of Mexican cuisine and one of the most popular condiments in the world. It is made by mashing ripe Hass avocados and mixing them with fresh lime juice, salt, finely chopped cilantro, white onion, and diced tomato. Some traditional recipes also add finely minced jalapeño or serrano pepper for extra heat. Guacamole originated with the Aztecs, who first created it in the 16th century using a molcajete, a traditional stone mortar and pestle. The word guacamole comes from the Nahuatl words ahuacatl (avocado) and molli (sauce). Today it is commonly served as a dip with tortilla chips, as a topping for tacos and burritos, or as a side dish alongside grilled meats.",

    "Mole is one of the most complex, celebrated, and time-consuming sauces in all of Mexican cuisine, often described as the national dish of Mexico. It is made from a rich blend of dried chili peppers, spices, nuts, seeds, chocolate, and many other ingredients, sometimes including up to 30 different components. The preparation of mole can take several days, as each ingredient must be toasted, ground, and combined carefully to achieve the right depth of flavor. The most famous version is mole negro from Oaxaca, which is dark, rich, and slightly bitter from the use of charred chilies and dark chocolate. Other well-known varieties include mole poblano from Puebla, mole verde, and mole amarillo. Mole is traditionally served over turkey or chicken, accompanied by rice and warm tortillas.",

    "Tamales are one of the oldest and most traditional foods in Mexican cuisine, with a history stretching back thousands of years to the ancient Aztec and Mayan civilizations. They are made from masa, a dough prepared from nixtamalized corn, which is spread onto a dried corn husk or banana leaf, filled with savory or sweet ingredients, and then folded and steamed until cooked through. Common fillings include seasoned pork, chicken with salsa verde, cheese with roasted peppers, or sweet fillings like raisins and cinnamon. Tamales have been found in archaeological evidence dating back to 8000 BCE, making them one of the oldest prepared foods in the Americas. They are especially popular during the Christmas season and Dia de Muertos celebrations, when families gather to make large batches together in a tradition called a tamalada.",

    "Tequila is Mexico's most famous and celebrated alcoholic beverage, made exclusively from the blue agave plant grown primarily in the region surrounding the city of Tequila in the state of Jalisco. The blue agave plant takes between 8 and 12 years to fully mature before it can be harvested. Once harvested, the core of the plant, called the pina, is roasted, crushed, and fermented to extract its sugary juice, which is then distilled into tequila. By law, tequila must be produced in Mexico to legally carry the name. There are several main types of tequila classified by aging time: blanco is unaged and bottled immediately after distillation, reposado is aged in oak barrels for 2 to 12 months, and anejo is aged for 1 to 3 years, developing a richer, smoother flavor. Tequila is enjoyed straight, in cocktails like the margarita, or as a shot with salt and lime.",

    "Mexican cuisine varies dramatically from region to region, reflecting the diverse geography, climate, and cultural heritage of the country. In northern Mexico, the cuisine is heavily influenced by cattle ranching traditions, with beef, grilled meats, and flour tortillas playing a central role. The Yucatan Peninsula has a distinct cuisine strongly influenced by ancient Mayan traditions, featuring unique dishes like cochinita pibil, a slow-roasted pork marinated in achiote paste and sour orange juice, cooked underground in a pit. The state of Oaxaca is known as the land of seven moles and is considered one of the culinary capitals of Mexico, famous for its complex sauces, tlayudas, and traditional mezcal production. In central Mexico, dishes like pozole, chiles en nogada, and barbacoa are staples of everyday cooking. This incredible regional diversity is one of the reasons UNESCO recognized Mexican cuisine as an Intangible Cultural Heritage of Humanity in 2010.",

    "Mexico has produced several world-renowned chefs who have brought Mexican cuisine to international attention and elevated it to the level of haute cuisine. Enrique Olvera, born in 1976 in Mexico City, is perhaps the most celebrated. He opened his flagship restaurant Pujol in Mexico City in 2000, which has consistently ranked among the top 50 restaurants in the world according to the prestigious World's 50 Best Restaurants list. Olvera is known for his innovative approach to traditional Mexican ingredients and techniques, creating dishes that honor the past while pushing culinary boundaries. Another influential figure is Diana Kennedy, born in England in 1923, who dedicated her life to documenting authentic Mexican regional cooking. She wrote eight highly influential cookbooks that are considered essential references for anyone serious about Mexican cuisine. In 2010, UNESCO added traditional Mexican cuisine to its list of Intangible Cultural Heritage of Humanity, recognizing its deep cultural significance.",

    "Enchiladas are a classic and deeply traditional Mexican dish that has been eaten for centuries across the country. They consist of corn tortillas that are lightly fried, dipped in a flavorful chili sauce, rolled around a filling, and then baked or served immediately. The filling can include shredded chicken, ground beef, cheese, beans, or vegetables, depending on the region and personal preference. The sauce is typically made from dried red chili peppers like guajillo or ancho, blended with tomatoes, garlic, and spices. Enchiladas are usually topped with crumbled queso fresco, sour cream, sliced onions, and fresh cilantro. There are many regional varieties, including enchiladas verdes topped with tangy green tomatillo sauce, enchiladas rojas with red chili sauce, and enchiladas suizas topped with a rich cream sauce and melted cheese.",

    "Churros are a popular and beloved fried pastry enjoyed across Mexico and much of Latin America, commonly sold by street vendors and at fairs and festivals. They are made from a simple choux-style dough of flour, water, salt, and sometimes eggs, which is piped through a star-shaped nozzle into hot oil and fried until golden brown and crispy on the outside. Once cooked, churros are rolled generously in cinnamon sugar and served hot, often accompanied by a thick, rich chocolate dipping sauce made from melted dark chocolate and warm milk. In Mexico, churros are a popular breakfast food as well as a street snack, and can be found at dedicated churro shops called churrerias. Some variations are filled with cajeta, a rich Mexican caramel made from goat's milk, or with chocolate or strawberry cream. They are especially popular during national holidays and celebrations.",

    "Horchata is a traditional and refreshing Mexican drink with a long history rooted in both Mexican and Spanish culinary traditions. The Mexican version is made by soaking raw long-grain rice in water overnight, then blending it with cinnamon sticks and sugar before straining it through a fine cloth to produce a smooth, milky white liquid. The result is a sweet, lightly spiced drink with a distinctive rice flavor that is served cold over ice. Horchata is especially popular during the hot summer months and is a common accompaniment to spicy Mexican food, as its natural sweetness and creaminess help to cool down the heat of chili peppers. It is widely available at Mexican restaurants, taquerias, and aguas frescas stands, where it is sold alongside other popular drinks like Jamaica hibiscus water and tamarind agua fresca. Some modern variations add vanilla, almonds, or coconut for extra flavor.",

    "Salsa is one of the most fundamental and versatile condiments in Mexican cuisine, used as a dip, a cooking sauce, and a table condiment in virtually every Mexican household and restaurant. The word salsa simply means sauce in Spanish, and there are dozens of distinct varieties throughout Mexico, each with its own flavor profile, heat level, and regional character. Salsa roja is made from blended red tomatoes, chili peppers, garlic, and onion, and can range from mild to very spicy. Salsa verde is made from tomatillos, which are small green fruits with a tart, citrusy flavor, blended with serrano or jalapeño peppers. Pico de gallo, also called salsa fresca, is a chunky, fresh salsa made from diced tomatoes, white onion, cilantro, lime juice, and jalapeño, and is not cooked. Salsas can be made smooth or chunky, raw or cooked, and mild or fiery hot depending on the type and quantity of chili used.",

    "Mexican cooking relies on a set of traditional tools and techniques that have been passed down through generations for thousands of years and remain central to authentic Mexican cuisine today. The comal is a flat, round griddle traditionally made from clay or cast iron, used to cook tortillas, toast chiles and spices, and char tomatoes and onions for salsas. The molcajete is a heavy stone mortar and pestle made from volcanic rock, used to grind spices, make salsas, and prepare guacamole. Using a molcajete produces a more textured, flavorful result than a blender. Nixtamalization is one of the most important ancient techniques in Mexican cooking, developed by the Aztecs over 3500 years ago. It involves soaking dried corn kernels in an alkaline solution of water and calcium hydroxide, which softens the corn, removes the outer hull, and dramatically increases its nutritional value by making niacin and amino acids bioavailable. The resulting treated corn is called nixtamal, which is ground into masa dough used to make tortillas, tamales, and many other traditional foods.",

    "El taco es el simbolo mas reconocido e importante de la cocina mexicana en todo el mundo. Consiste en una tortilla de maiz o de harina de trigo, doblada o enrollada, rellena de diversos ingredientes como carne asada, pollo, pescado, mariscos, o verduras a la plancha. Los tacos se sirven generalmente con salsa, cilantro, cebolla picada y un chorrito de limon fresco. Los tacos de calle son una parte fundamental e inseparable de la cultura culinaria mexicana, y se pueden encontrar en cada esquina de las ciudades y pueblos mexicanos. Cada region del pais tiene sus propias variedades unicas de tacos: los tacos de barbacoa del centro del pais, los tacos de pescado de Baja California, y los tacos al pastor de la Ciudad de Mexico, que fueron influenciados por la inmigracion libanesa en el siglo XX.",

    "El chile es el ingrediente mas importante, representativo y simbolico de la cocina mexicana, y ha sido parte esencial de la alimentacion en Mesoamerica desde hace mas de 6000 anos. Existen mas de 60 variedades de chiles en Mexico, que van desde los mas suaves y dulces como el chile poblano y el chile ancho, hasta los extremadamente picantes como el chile habanero y el chile de arbol. Los chiles se utilizan de multiples formas en la cocina mexicana: frescos, secos, ahumados, en polvo, en pasta o en conserva. Son la base fundamental de salsas, moles, adobos, marinadas y guisos en toda la gastronomia mexicana. La capsaicina, el compuesto quimico que da el picor a los chiles, tambien tiene propiedades medicinales y ha sido utilizada en la medicina tradicional mexicana durante siglos para tratar dolores e inflamaciones.",

    "La cocina mexicana fue declarada Patrimonio Cultural Inmaterial de la Humanidad por la UNESCO en el ano 2010, convirtiendose en una de las pocas cocinas del mundo en recibir este prestigioso reconocimiento internacional. Este honor refleja la profunda riqueza cultural, historica y social de la gastronomia mexicana, que se ha desarrollado durante miles de anos a traves de la fusion de las tradiciones culinarias indigenas con las influencias espanolas, africanas y de otras culturas. La cocina mexicana destaca por su extraordinaria diversidad de ingredientes autoctones, tecnicas de preparacion ancestrales y sabores regionales unicos. Los ingredientes principales que forman la base de la cocina mexicana son el maiz, el chile, el frijol y el jitomate, conocidos colectivamente como la milpa, el sistema agricola tradicional de Mesoamerica. Hoy en dia, la cocina mexicana sigue siendo una expresion viva de la identidad y el orgullo del pueblo mexicano.",
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