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
    """Tacos are one of the most iconic dishes in Mexican cuisine and a symbol of the country's rich culinary heritage. A taco consists of a small hand-sized corn or flour tortilla topped or filled with various ingredients. Common fillings include carne asada which is grilled beef, al pastor which is marinated pork cooked on a vertical spit, carnitas which is slow-cooked pork, chicken, fish, or vegetables. Tacos are typically garnished with fresh cilantro, diced white onion, salsa, lime juice, and sometimes avocado or guacamole. They originated as street food among the working class in Mexico City in the early 20th century and have since become one of the most recognized foods in the world. Each region of Mexico has its own unique taco tradition, from the fish tacos of Baja California to the barbacoa tacos of Hidalgo.""",

    """Guacamole is one of the oldest and most beloved dishes in Mexican cuisine, dating back to the time of the Aztec empire in the 16th century. It is made by mashing ripe Hass avocados in a molcajete and combining them with fresh lime juice, salt, finely chopped white onion, cilantro, and diced tomato. Some variations include minced jalapeño or serrano peppers for added heat. The word guacamole comes from the Nahuatl words ahuacatl meaning avocado and molli meaning sauce. Today guacamole is served as a dip with tortilla chips, as a topping for tacos and burritos, and as a side dish alongside grilled meats. Mexico produces approximately 50 percent of the world's avocados, making it the global leader in avocado production.""",

    """Mole is widely considered the most complex and celebrated sauce in all of Mexican gastronomy and is often referred to as the national dish of Mexico. The most famous variety is mole negro from Oaxaca, which can contain up to 30 different ingredients including multiple types of dried chili peppers, spices such as cumin, cloves and cinnamon, nuts, seeds, plantain, raisins, tomatoes, and dark Mexican chocolate. The preparation of a traditional mole negro can take several days, as each ingredient must be individually toasted or fried, then ground and slowly combined. Mole poblano from the state of Puebla is another famous variety, said to have been invented by nuns at a convent in the 17th century. Mole is traditionally served over turkey or chicken at celebrations and festivals. The seven moles of Oaxaca are a celebrated group of regional sauces that include negro, rojo, coloradito, amarillo, verde, chichilo, and manchamanteles.""",

    """Tamales are one of the oldest prepared foods in the Americas, with archaeological evidence suggesting they have been eaten in Mesoamerica since at least 8000 BCE. They are made from masa, a dough prepared from nixtamalized corn, spread onto a dried corn husk or banana leaf, filled with savory or sweet ingredients, and then folded and steamed for approximately one hour. Common fillings include seasoned pork, chicken in salsa verde, cheese and roasted poblano peppers, or sweet versions with raisins and cinnamon. Tamales are deeply connected to Mexican cultural and religious celebrations. They are especially popular during Christmas and the Dia de la Candelaria in February, when families gather for a tamalada, a communal tamale-making event. In pre-Columbian times, tamales were used as portable food for warriors and travelers due to their durability and high energy content.""",

    """Tequila is Mexico's most internationally recognized alcoholic beverage and one of the country's most important cultural exports. It is produced exclusively from the blue Weber agave plant which grows primarily in the red volcanic soil of the Jalisco highlands and the region surrounding the town of Tequila. By Mexican law, tequila can only be produced in five states: Jalisco, Guanajuato, Michoacan, Nayarit, and Tamaulipas. The blue agave plant requires between 8 and 12 years to reach full maturity. At harvest, skilled workers called jimadores use a sharp tool called a coa to extract the pina, which is the core of the plant. The pinas are then slow-roasted, crushed to extract their juice, and fermented before being distilled. Tequila is classified by aging: blanco is unaged, reposado is aged 2 to 12 months in oak barrels, anejo is aged 1 to 3 years, and extra anejo is aged more than 3 years.""",

    """Mexican cuisine varies dramatically from one region to another, reflecting the country's extraordinary geographic and cultural diversity. In northern states like Sonora and Chihuahua, the cuisine is heavily influenced by cattle ranching traditions, with beef, flour tortillas, and dishes like carne asada and machaca being central. The Yucatan Peninsula has a cuisine deeply rooted in ancient Maya traditions, featuring cochinita pibil, a slow-roasted pork marinated in achiote paste cooked underground in a pit oven. Oaxaca in the south is known as the culinary capital of Mexico, famous for its seven moles, tlayudas, quesillo string cheese, and mezcal. Veracruz on the Gulf Coast has a distinct cuisine influenced by Spanish, African, and Caribbean flavors, featuring huachinango a la veracruzana which is red snapper cooked in a tomato sauce with olives and capers. This incredible regional diversity is one of the reasons UNESCO recognized Mexican cuisine as an Intangible Cultural Heritage of Humanity in 2010.""",

    """Mexico has produced some of the most celebrated and influential chefs in the world, who have elevated traditional Mexican cuisine to international recognition. Enrique Olvera, born in Mexico City in 1976, is widely regarded as the most important figure in modern Mexican gastronomy. He opened his landmark restaurant Pujol in Mexico City in 2000, which has consistently ranked among the top 20 restaurants in the world on the World's 50 Best Restaurants list. Olvera is known for his deep respect for traditional Mexican ingredients which he reinterprets with a creative modern approach. His signature dish is mole madre, a mole sauce that is continuously aged for over 1000 days. Diana Kennedy, a British-born food writer born in 1923, dedicated her life to researching and documenting authentic Mexican regional cooking. She wrote eight highly influential books including The Cuisines of Mexico published in 1972, and was awarded the Order of the Aztec Eagle by the Mexican government for her contributions.""",

    """Mexican street food culture is one of the most vibrant and important aspects of the country's culinary identity. Street food, known as antojitos meaning little cravings, has been a central part of Mexican daily life since pre-Columbian times when markets called tianguis were filled with vendors selling prepared foods. Today, Mexican cities are home to thousands of street food stalls and small restaurants called fondas and cocinas economicas that serve freshly prepared traditional food at affordable prices. Popular street foods beyond tacos include elotes, which are grilled corn on the cob coated with mayonnaise, cheese, chili powder, and lime juice. Tlayudas are large crispy tortillas topped with beans, cheese, and meat, popular in Oaxaca. Esquites are cups of corn kernels cooked with epazote herb and topped with the same condiments as elotes. Tamales, quesadillas, gorditas, and sopes are also widely sold on the streets of Mexico.""",

    """The history of Mexican cuisine is a story of cultural fusion spanning thousands of years, beginning with the ancient civilizations of Mesoamerica and evolving through colonial contact and global exchange. The foundation of Mexican cooking was built by indigenous cultures including the Olmec, Maya, Zapotec, and Aztec civilizations, who developed sophisticated agricultural systems centered on the milpa, a polyculture of corn, beans, and squash. When Spanish conquistadors arrived in 1519, they brought with them ingredients from Europe and Africa including pork, beef, chicken, dairy products, wheat, rice, sugar, and various spices. The blending of indigenous and European ingredients and techniques gave rise to what we now call Mexican cuisine. The colonial period also introduced chocolate, vanilla, and chili peppers from Mexico to the rest of the world, permanently transforming global cooking. Mexico's diverse geography, with its tropical coasts, highland valleys, and arid deserts, allowed for the cultivation of an extraordinary variety of ingredients.""",

    """Chocolate has its origins in Mexico and Mesoamerica, where cacao has been cultivated and consumed for over 3000 years. The ancient Maya and Aztec civilizations considered cacao a sacred gift from the gods and used cacao beans as currency as well as food. They prepared a bitter, spiced drink called xocolatl, made from ground cacao beans mixed with water, chili peppers, cornmeal, and spices, which was consumed cold and frothy. Cacao beans were so valuable that they were used to pay taxes and purchase goods in Aztec markets. When Spanish explorers brought cacao back to Europe in the 16th century, they added sugar and milk to create the sweeter chocolate that became popular worldwide. In Mexican cuisine today, chocolate remains an important ingredient, used in mole sauces, atole which is a warm corn-based drink, champurrado which is a chocolate-thickened atole, and in traditional sweets and pastries. The state of Tabasco and the Soconusco region of Chiapas are considered the heartland of Mexican cacao production.""",

    """Corn, known in Mexico as maiz, is the most sacred and fundamental ingredient in Mexican cuisine and culture, and has been cultivated in Mesoamerica for approximately 9000 years. Mexico is recognized as the birthplace of corn, where ancient farmers selectively bred wild grass called teosinte into the diverse varieties of corn we know today. There are over 60 native varieties of corn in Mexico, ranging in color from white and yellow to blue, red, purple, and black, each with distinct flavors and textures suited to different preparations. Corn is used to make tortillas, tamales, tostadas, gorditas, tlayudas, atole, pozole, and countless other dishes. The process of nixtamalization, developed thousands of years ago, transforms dried corn into nutritionally superior masa by cooking it in an alkaline solution. In many indigenous communities in Mexico, corn cultivation remains deeply tied to spiritual beliefs, rituals, and community identity. The milpa agricultural system, which intercropped corn with beans and squash, sustained Mesoamerican civilizations for millennia.""",

    """Mexican cooking relies on a set of traditional tools and ancient techniques that have been used for thousands of years and remain central to authentic preparation today. The comal is a flat round griddle traditionally made from unglazed clay or cast iron, used to cook tortillas, toast dried chili peppers and spices, and char tomatoes and onions for salsas. The molcajete is a heavy mortar and pestle carved from volcanic basalt rock, used to grind spices, crush chili peppers, and make fresh salsas and guacamole. Grinding in a molcajete produces a coarser, more textured result with a richer flavor than using a blender. Nixtamalization is one of the most important ancient techniques, developed over 3500 years ago, which involves cooking dried corn in an alkaline solution of water and calcium hydroxide. This dramatically increases the corn's nutritional value by releasing niacin and making amino acids bioavailable. The treated corn is called nixtamal and is ground into masa dough to make tortillas, tamales, and other traditional foods.""",

    """El taco es sin duda el platillo mas reconocido e importante de la cocina mexicana en todo el mundo, y representa una parte esencial de la identidad cultural del pais. Consiste en una tortilla de maiz o de harina de trigo, de tamano pequeno, doblada o enrollada alrededor de un relleno que puede incluir carne asada, pollo, pescado, mariscos, frijoles, o una gran variedad de verduras. Los tacos se sirven generalmente acompanados de salsa, cilantro fresco, cebolla blanca finamente picada, y un chorrito generoso de limon recien exprimido. Los tacos de calle son una parte inseparable de la vida cotidiana en Mexico y se pueden encontrar en cada esquina de las ciudades y pueblos del pais, vendidos desde pequenos puestos callejeros llamados taquerias. Cada region de Mexico tiene sus propias variedades unicas: los tacos de barbacoa del Estado de Mexico, los tacos de pescado de Baja California, y los mundialmente famosos tacos al pastor de la Ciudad de Mexico, cuya preparacion fue influenciada por la inmigracion libanesa al pais en el siglo veinte.""",

    """El chile es sin duda el ingrediente mas importante y representativo de toda la cocina mexicana, y ha sido parte fundamental de la alimentacion en Mesoamerica desde hace mas de seis mil anos. En Mexico existen mas de sesenta variedades distintas de chiles, que abarcan desde los completamente suaves como el chile poblano y el chile ancho, hasta los extremadamente picantes como el chile habanero y el chile de arbol. Los chiles se utilizan de multiples formas en la gastronomia mexicana: se consumen frescos, se secan al sol, se ahuman, se muelen en polvo, se preparan en pasta, o se conservan en vinagre. Son la base imprescindible de innumerables salsas, moles, adobos, marinadas y guisos que definen la cocina mexicana. El compuesto quimico responsable del picor en los chiles es la capsaicina, la cual ademas de producir la sensacion de ardor, tiene importantes propiedades medicinales antiinflamatorias que han sido aprovechadas en la medicina tradicional mexicana durante siglos.""",

    """La cocina mexicana fue oficialmente declarada Patrimonio Cultural Inmaterial de la Humanidad por la UNESCO en el ano dos mil diez, convirtiendose en una de las primeras gastronomias del mundo en recibir este prestigioso reconocimiento internacional. Este reconocimiento refleja la extraordinaria profundidad cultural e historica de la gastronomia mexicana, que ha evolucionado durante miles de anos como resultado de la fusion entre las antiguas tradiciones culinarias de los pueblos indigenas de Mesoamerica y las influencias posteriores de la cocina espanola, africana, y de otras culturas del mundo. Los cuatro ingredientes fundamentales que forman la base de la cocina mexicana son el maiz, el chile, el frijol y el jitomate, conocidos colectivamente como la milpa, el sistema agricola tradicional de Mesoamerica desarrollado hace mas de siete mil anos. Hoy en dia, la cocina mexicana sigue siendo una expresion viva de la identidad y el orgullo del pueblo mexicano, celebrada tanto dentro como fuera del pais.""",
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
        chunk_size=500,
        chunk_overlap=75,
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
        placeholder="e.g. What is tequila made from? / Que es el chile?"
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
    This app uses **chunk_size = 500 characters** with **chunk_overlap = 75 characters**.

    **Why these values?**
    - Documents are long paragraphs of 400-700 characters
    - 500 chars captures one complete thought with full context
    - 75 chars overlap prevents cutting sentences mid-thought
    - Smaller chunks (200) lose too much context for longer documents
    - Larger chunks (800) include too much unrelated content per chunk
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
        "Chunk Size": ["Size 200 (small)", "Size 500 (current)"],
        "Number of Chunks": [len(c200), len(c500)]
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