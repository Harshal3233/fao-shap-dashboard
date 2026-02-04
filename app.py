import streamlit as st

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Harshal M Patel | AI Engineer",
    layout="wide",
)

# -------------------------------------------------
# Brand / UI tokens
# -------------------------------------------------
BRAND = {
    "text": "#111827",        # slate-900
    "muted": "#6b7280",       # gray-500
    "border": "rgba(17, 24, 39, 0.12)",
    "soft": "rgba(17, 24, 39, 0.04)",
    "accent": "#2563eb",      # blue-600
    "accent2": "#7c3aed",     # violet-600
    "chip": "rgba(37, 99, 235, 0.10)",
    "note": "rgba(37, 99, 235, 0.08)",
    "codebg": "rgba(17, 24, 39, 0.035)",
}

# -------------------------------------------------
# CSS
# -------------------------------------------------
st.markdown(
    f"""
<style>
.block-container {{
    padding-top: 1.2rem;
    padding-bottom: 2.4rem;
    max-width: 1220px;
}}
section[data-testid="stSidebar"] > div {{
    padding-top: 1.05rem;
}}
h1, h2, h3 {{ letter-spacing: 0.2px; }}
.small-muted {{ color: {BRAND["muted"]}; font-size: 0.95rem; }}
.small {{ font-size: 0.95rem; }}

.hero {{
    border: 1px solid {BRAND["border"]};
    border-radius: 18px;
    padding: 1.15rem 1.2rem;
    background:
        radial-gradient(1200px 220px at 18% -20%, {BRAND["chip"]}, transparent 60%),
        radial-gradient(1200px 220px at 92% 0%, rgba(124, 58, 237, 0.10), transparent 55%),
        #ffffff;
}}
.hero-title {{
    font-size: 2.25rem;
    font-weight: 900;
    color: {BRAND["text"]};
    margin-bottom: 0.12rem;
}}
.hero-tag {{
    font-size: 1.06rem;
    font-weight: 750;
    color: rgba(17, 24, 39, 0.78);
}}
.hero-meta {{
    color: {BRAND["muted"]};
    margin-top: 0.25rem;
}}

.section-title {{
    font-size: 1.35rem;
    font-weight: 900;
    margin: 1.6rem 0 0.55rem 0;
    color: {BRAND["text"]};
}}
.section-line {{
    height: 2px;
    width: 96px;
    border-radius: 999px;
    background: linear-gradient(90deg, {BRAND["accent"]}, {BRAND["accent2"]});
    margin-bottom: 0.95rem;
}}
.section-sub {{
    color: {BRAND["muted"]};
    margin-top: -0.35rem;
    margin-bottom: 0.8rem;
}}

.card {{
    border: 1px solid {BRAND["border"]};
    border-radius: 18px;
    padding: 1.05rem 1.1rem;
    background: #ffffff;
    margin-bottom: 0.95rem;
}}
.card:hover {{ border-color: rgba(17, 24, 39, 0.20); }}
.card-title {{
    font-size: 1.10rem;
    font-weight: 900;
    margin-bottom: 0.10rem;
    color: {BRAND["text"]};
}}
.card-sub {{
    color: {BRAND["muted"]};
    font-size: 0.95rem;
    margin-bottom: 0.55rem;
}}

.pill {{
    display: inline-block;
    padding: 0.23rem 0.62rem;
    margin: 0.16rem 0.28rem 0.10rem 0;
    border-radius: 999px;
    border: 1px solid {BRAND["border"]};
    background: {BRAND["soft"]};
    font-size: 0.84rem;
    color: rgba(17, 24, 39, 0.85);
}}
.pill-accent {{
    background: {BRAND["note"]};
    border-color: rgba(37, 99, 235, 0.20);
}}
.kpi {{
    border: 1px solid {BRAND["border"]};
    border-radius: 14px;
    padding: 0.75rem 0.85rem;
    background: #ffffff;
}}
.kpi-label {{
    color: {BRAND["muted"]};
    font-size: 0.85rem;
    margin-bottom: 0.15rem;
}}
.kpi-value {{
    font-size: 1.15rem;
    font-weight: 900;
    color: {BRAND["text"]};
}}

.side-card {{
    border: 1px solid {BRAND["border"]};
    border-radius: 18px;
    padding: 0.95rem 0.95rem;
    background: #ffffff;
    margin-bottom: 0.85rem;
}}
.side-title {{
    font-weight: 900;
    color: {BRAND["text"]};
    margin-bottom: 0.45rem;
    font-size: 1.02rem;
}}
.navbtn {{
    display: block;
    padding: 0.55rem 0.75rem;
    border: 1px solid {BRAND["border"]};
    border-radius: 12px;
    background: #ffffff;
    color: {BRAND["text"]};
    font-weight: 750;
    text-decoration: none !important;
    margin: 0.35rem 0;
}}
.navbtn:hover {{
    border-color: rgba(17, 24, 39, 0.22);
    background: {BRAND["soft"]};
}}

.anchor {{
    height: 1px;
    margin-top: -70px;
    padding-top: 70px;
}}

/* Blog polish */
.note-meta {{
    display: flex;
    gap: 0.6rem;
    align-items: center;
    flex-wrap: wrap;
    margin: 0.2rem 0 0.6rem 0;
}}
.meta-dot {{
    width: 5px; height: 5px;
    background: rgba(17,24,39,0.35);
    border-radius: 999px;
}}
.callout {{
    border: 1px solid rgba(37, 99, 235, 0.20);
    background: {BRAND["note"]};
    border-radius: 14px;
    padding: 0.75rem 0.85rem;
    margin-top: 0.6rem;
}}
.codebox {{
    border: 1px solid rgba(17, 24, 39, 0.12);
    background: {BRAND["codebg"]};
    border-radius: 14px;
    padding: 0.75rem 0.85rem;
    margin-top: 0.65rem;
}}
.codebox pre {{ margin: 0; font-size: 0.85rem; }}

/* Timeline */
.timeline {{
    position: relative;
    padding-left: 1.1rem;
}}
.timeline::before {{
    content: "";
    position: absolute;
    left: 0.36rem;
    top: 0.2rem;
    bottom: 0.2rem;
    width: 2px;
    background: rgba(17, 24, 39, 0.10);
    border-radius: 999px;
}}
.t-item {{
    position: relative;
    padding-left: 1.25rem;
    margin-bottom: 0.9rem;
}}
.t-dot {{
    position: absolute;
    left: 0.18rem;
    top: 0.42rem;
    width: 12px;
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, {BRAND["accent"]}, {BRAND["accent2"]});
    box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.10);
}}
.t-card {{
    border: 1px solid {BRAND["border"]};
    border-radius: 18px;
    padding: 0.95rem 1.05rem;
    background: #ffffff;
}}
.t-title {{
    font-weight: 900;
    color: {BRAND["text"]};
    font-size: 1.05rem;
}}
.t-meta {{
    color: {BRAND["muted"]};
    margin-top: 0.15rem;
    margin-bottom: 0.55rem;
    font-size: 0.93rem;
}}
.visit {{
    margin-top: 0.55rem;
    padding-top: 0.25rem;
}}
.visit a {{
    color: {BRAND["accent"]};
    font-weight: 800;
    text-decoration: none;
}}
.visit a:hover {{
    text-decoration: underline;
}}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Data
# -------------------------------------------------
PROFILE = {
    "name": "Harshal M Patel",
    "tagline": "AI Engineer | Data Science | Software Engineering | Analytics",
    "location": "Rome, Italy (available to relocate)",
    "email": "harshalmpatel2001@gmail.com",
    "phone": "+91 9974890592",
    "linkedin": "https://www.linkedin.com/in/harshal-patel-246306352/",
}

ABOUT = (
    "AI Engineer focused on turning data into decisions through analysis, modeling, explainability, and deployment. "
    "Work is framed like a data product: define the problem, build a reliable pipeline, and ship an interface people can use."
)

EXPERIENCE = [
    {
        "role": "IT & Marketing Support",
        "company": "Vinayak Infotech",
        "meta": "Ahmedabad, India • May 2024 – 2025",
        "problem": "Operational teams needed reliable systems and smoother technical workflows across support and marketing execution.",
        "approach": [
            "Installed and configured computer systems and hardware for stable day-to-day operations.",
            "Mentored team members in Java and Python (advanced and libraries) and supported testing tasks.",
            "Introduced practical fixes that reduced bottlenecks in marketing workflows.",
        ],
        "outcome": "Improved internal efficiency, reduced technical downtime, and enabled faster execution with clearer technical support.",
        "signals": ["Delivery", "Mentorship", "Operations", "Workflow Efficiency"],
        "tags": ["Python", "Java", "IT Support", "Process Improvement"],
    },
    {
        "role": "Software Engineer",
        "company": "Cignex Datamatics",
        "meta": "Gujarat, India • Aug 2022 – Jul 2023",
        "problem": "Enterprise releases required stronger quality gates to minimize defects and improve stability for multinational clients.",
        "approach": [
            "Developed structured testing workflows using Jira and collaborated closely with QA teams.",
            "Supported Java-based systems and improved testing consistency across environments.",
            "Focused on early defect discovery to reduce downstream rework.",
        ],
        "outcome": "Improved release reliability and reduced post-deployment issues through stronger testing discipline and collaboration.",
        "signals": ["Quality", "Reliability", "QA Collaboration", "Client-Facing Systems"],
        "tags": ["Java", "Jira", "QA", "Collaboration"],
    },
    {
        "role": "Software Engineering Intern",
        "company": "Axis Ray Technology",
        "meta": "Gujarat, India • Jun 2022 – Dec 2022",
        "problem": "Projects needed consistent development support and clean data integration across teams and timelines.",
        "approach": [
            "Strengthened skills in Java, Advanced Java, and SQL-based integration patterns.",
            "Worked on real-world tasks with cross-national teams and project stakeholders.",
            "Supported peer productivity through mentoring and collaborative delivery.",
        ],
        "outcome": "Accelerated delivery readiness by improving implementation consistency and team coordination.",
        "signals": ["Engineering Foundations", "SQL Integration", "Team Support"],
        "tags": ["Advanced Java", "SQL", "Collaboration", "Mentorship"],
    },
    {
        "role": "Sales Executive",
        "company": "AURA ETHNICS",
        "meta": "Jan 2022 – Jun 2022",
        "problem": "Sales performance needed clearer visibility into customer behavior and what drives conversion.",
        "approach": [
            "Used Google Analytics insights to understand traffic, conversion patterns, and customer journeys.",
            "Adjusted listings and promotions using data signals rather than guesswork.",
            "Strengthened customer experience through product storytelling and service quality.",
        ],
        "outcome": "Improved conversion outcomes through data-informed decision-making and stronger customer engagement.",
        "signals": ["Analytics", "Storytelling", "Conversion Focus"],
        "tags": ["Analytics", "Customer Engagement", "Sales"],
    },
]

PROJECTS = [
    {
        "title": "Explainable ML for Agricultural Production (FAOSTAT + SHAP)",
        "one_liner": "Random Forest + SHAP to explain cross-country production drivers (crop vs country effects).",
        "summary": (
            "Built an explainable ML dashboard to compare agricultural production patterns across Italy, France, Germany, and Spain. "
            "Used Random Forest as an interpretive model and SHAP to quantify feature influence transparently."
        ),
        "highlights": [
            "Compared crop-driven vs country-driven production patterns across multiple nations.",
            "Used SHAP to translate model behavior into stakeholder-friendly explanations.",
            "Reported MAE, Relative MAE, and R² to keep evaluation transparent and interpretable.",
        ],
        "stack": ["Python (Advanced)", "Random Forest", "SHAP", "Explainable AI", "Streamlit"],
        "live": "https://fao-shap-dashboard-fyfzpqshuzcpfvz79m6xio.streamlit.app/#d166e061",
        "tags": ["XAI", "SHAP", "ML", "Streamlit"],
    },
    {
        "title": "Rome Airbnb Analysis and Price Prediction",
        "one_liner": "EDA + interactive mapping + SVM classifier to categorize listing prices in Rome.",
        "summary": (
            "Analyzed publicly available Rome Airbnb listings to understand pricing patterns. "
            "Built a dashboard to visualize hotspots and trained an SVM classifier to predict price categories."
        ),
        "highlights": [
            "Room type + neighborhood emerged as primary price drivers.",
            "Premium zones (Centro Storico, Trastevere) highlighted using map-based insights.",
            "Stakeholder-friendly dashboard with clear visuals and variable explanations.",
        ],
        "stack": ["Python (Advanced)", "EDA", "SVM", "Streamlit", "Data Storytelling"],
        "live": "https://rome-airbnb-app-bsr6lkugduccvbrwmjfksk.streamlit.app/",
        "tags": ["EDA", "Maps", "ML", "Streamlit"],
    },
]

SKILLS = {
    "Programming & Data": [
        "Python (Advanced & Libraries)", "Advanced Java", "SQL",
        "EDA", "K-Means", "Regression", "Random Forest", "SVM", "Neural Networks",
        "Data Management", "Model Building & Deployment", "Predictive Analysis",
    ],
    "Tools & Platforms": ["Streamlit", "VS Code", "Jupyter Notebook", "GitHub", "Jira", "MS Excel", "Google Workspace"],
    "Web & Design": ["HTML", "CSS", "JavaScript", "Figma", "Interactive Dashboards"],
}

EDUCATION = [
    {
        "title": "International Master in Data Science (Ongoing)",
        "meta": "Rome Business School | Rome, Italy",
        "bullets": [
            "Applied learning through Oracle labs: data workflows, database foundations, and practical analytics.",
            "IBM labs exposure: enterprise-style problem framing, tooling practices, and real-world DS workflows.",
            "Industry excursions with IBM and Owkin.",
        ],
    },
    {
        "title": "B.Tech in Computer Engineering (First Class Distinction)",
        "meta": "Swarrnim Institute of Technology & Startup | Gujarat, India | 2020 – 2024",
        "bullets": [
            "Debate Champion (2020–2021).",
            "Student Council Member (2020–2024).",
            "Hackathons; supported Blood Donation Camp organization (2023).",
        ],
    },
]

# NEW: separate section for excursions with your IBM details
INDUSTRY_EXCURSIONS = [
    {
        "title": "IBM Industry Excursion | AI + Research Labs",
        "meta": "Rome Business School (Industry Exposure)",
        "bullets": [
            "Interactive session with IBM lab team on the Granite model and enterprise AI workflows.",
            "Discussion focus: integrating an LLM layer to make Granite outputs more specific and context-driven for business use cases.",
            "Hands-on exposure to cybersecurity experiments and lab demonstrations.",
            "Explored applied quantum computing concepts and demonstrations for next-gen computation workflows.",
        ],
        "tags": ["IBM Granite", "LLMs", "Cybersecurity Labs", "Quantum Computing"],
    },
    {
        "title": "Owkin Industry Excursion | AI in Healthcare",
        "meta": "Rome Business School (Industry Exposure)",
        "bullets": [
            "Explored applied AI use cases and problem framing in healthcare-focused environments.",
            "Observed how organizations operationalize ML workflows and translate insights into decisions.",
        ],
        "tags": ["Healthcare AI", "Applied ML", "Problem Framing"],
    },
]

LANGUAGES = "English (Native/Bilingual), Hindi (Native), Italian (Basic – improving)"

EXPLORING = [
    "Embedding OpenAI into data science workflows (retrieval, prompting patterns, evaluation).",
    "Embeddings + semantic search for context-aware analytics and knowledge retrieval.",
    "Model building and deployment patterns (reproducibility, packaging, monitoring mindset).",
    "Strengthening applied skills through Oracle and IBM labs: data workflows, analytics, and DS tooling practices.",
    "Italian language: targeting A2 proficiency for international technical environments.",
    "Exploring LLM models and practical use cases (analysis, summarization, data assistants).",
]

BLOG_NOTES = [
    {
        "title": "How SHAP explains model drivers in real decision-making",
        "subtitle": "Interpretability as a product requirement, not a nice-to-have",
        "date": "Jan 2026",
        "read_time": "3 min read",
        "tags": ["XAI", "SHAP", "Interpretability", "Random Forest"],
        "lead": "When models influence decisions, accuracy alone is insufficient. Explainability creates trust, comparison, and accountability.",
        "bullets": [
            "Use SHAP to quantify feature contribution, not just rank importance.",
            "Compare segments (countries, categories, regions) using consistent explanations.",
            "Translate model behavior into stakeholder language: what drives outcomes and why.",
        ],
        "callout_title": "Hard-earned lesson",
        "callout": "Models become useful when their reasoning can be challenged, explained, and improved.",
        "code_title": "Conceptual SHAP workflow",
        "code": """# Conceptual outline
model = RandomForestRegressor(...)
model.fit(X_train, y_train)

explainer = shap.Explainer(model, X_train)
shap_values = explainer(X_test)

# Global drivers
shap.plots.bar(shap_values)

# Local explanations
shap.plots.waterfall(shap_values[0])""",
    },
    {
        "title": "Why EDA + SVM works well for price categories",
        "subtitle": "Reframing the problem to make results stable and communicable",
        "date": "Jan 2026",
        "read_time": "3 min read",
        "tags": ["EDA", "SVM", "Classification", "Dashboards"],
        "lead": "Pricing problems often improve when the problem is reframed. EDA reveals distribution shifts and outliers, and SVM gives a strong, interpretable baseline.",
        "bullets": [
            "Start with EDA: outliers, skew, neighborhood effects, and feature leakage checks.",
            "Bin prices into categories to stabilize targets and support clearer storytelling.",
            "Use SVM as a strong baseline classifier before adding complexity.",
        ],
        "callout_title": "Hard-earned lesson",
        "callout": "A model that stakeholders understand beats a model nobody trusts.",
        "code_title": "Conceptual SVM pipeline",
        "code": """# Conceptual outline
X = features
y = price_category  # binned labels

pipe = Pipeline([
  ("prep", preprocessor),
  ("clf", SVC(kernel="rbf"))
])

pipe.fit(X_train, y_train)
pred = pipe.predict(X_test)""",
    },
    # NEW NOTE 1
    {
        "title": "Deployment friction: parser & Python version mismatch",
        "subtitle": "Stabilizing builds by pinning versions and simplifying dependencies",
        "date": "Jan 2026",
        "read_time": "3 min read",
        "tags": ["Deployment", "Python", "Dependencies", "Streamlit Cloud"],
        "lead": "Deployments fail more often from environment mismatches than from code. Parser issues and Python version drift are common failure points.",
        "bullets": [
            "Pin Python version and critical libs to avoid dependency resolution surprises.",
            "Reduce heavy or unused packages to keep builds lightweight and stable.",
            "Bypass parser fragility by simplifying inputs, validating formats, and using safer parsing patterns.",
            "Treat the deployment environment as part of the system: reproducibility is a feature.",
        ],
        "callout_title": "Hard-earned lesson",
        "callout": "If a project can’t deploy consistently, it isn’t finished yet.",
        "code_title": "Conceptual fix checklist",
        "code": """# Conceptual checklist
# 1) Pin python version: runtime.txt (example)
# python-3.11

# 2) Pin dependencies: requirements.txt (example)
# streamlit==X.Y.Z
# scikit-learn==A.B.C

# 3) Remove unused heavy libs
# 4) Validate inputs + safer parsing""",
    },
    # NEW NOTE 2
    {
        "title": "Streamlit vs Django: demo speed vs production structure",
        "subtitle": "Why Streamlit shines for prototypes and Django scales for permanent apps",
        "date": "Jan 2026",
        "read_time": "4 min read",
        "tags": ["Django", "Streamlit", "Architecture", "Deployment"],
        "lead": "Different tools win at different stages. Streamlit accelerates demos, while Django supports long-term app structure when features and users grow.",
        "bullets": [
            "Streamlit: fast UI iteration, perfect for showcasing models and dashboards quickly.",
            "Django: stronger separation of concerns (apps/modules), durable routing, auth, and database integration.",
            "Production Django requires deeper planning: segmentation, services, background tasks, and thorough processing pipelines.",
            "A practical approach: Streamlit for model demos; Django for permanent products with users and operational workflows.",
        ],
        "callout_title": "Hard-earned lesson",
        "callout": "A demo sells an idea. A production app supports a business.",
        "code_title": "Conceptual architecture contrast",
        "code": """# Streamlit (demo-first)
# app.py -> UI + light logic

# Django (production-first)
# project/
#   apps/
#   urls.py
#   views.py
#   services/
#   models.py
#   templates/
#   tasks/ (Celery)
#   tests/""",
    },
]

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def pills(items, accent=False):
    cls = "pill pill-accent" if accent else "pill"
    return "".join([f"<span class='{cls}'>{x}</span>" for x in items])

def anchor(anchor_id: str):
    st.markdown(f"<div id='{anchor_id}' class='anchor'></div>", unsafe_allow_html=True)

def section(title: str, subtitle: str | None = None):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-line'></div>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<div class='section-sub'>{subtitle}</div>", unsafe_allow_html=True)

def timeline_open():
    st.markdown("<div class='timeline'>", unsafe_allow_html=True)

def timeline_close():
    st.markdown("</div>", unsafe_allow_html=True)

def timeline_item(e):
    st.markdown("<div class='t-item'>", unsafe_allow_html=True)
    st.markdown("<div class='t-dot'></div>", unsafe_allow_html=True)
    st.markdown("<div class='t-card'>", unsafe_allow_html=True)

    st.markdown(f"<div class='t-title'>{e['role']} • {e['company']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='t-meta'>{e['meta']}</div>", unsafe_allow_html=True)

    st.markdown(pills(e["signals"], accent=True), unsafe_allow_html=True)

    cols = st.columns([0.34, 0.33, 0.33], vertical_alignment="top")
    with cols[0]:
        st.markdown("**Problem**")
        st.write(e["problem"])
    with cols[1]:
        st.markdown("**Approach**")
        st.write("\n".join([f"- {x}" for x in e["approach"]]))
    with cols[2]:
        st.markdown("**Outcome**")
        st.write(e["outcome"])

    st.markdown(pills(e["tags"]), unsafe_allow_html=True)
    st.markdown("</div></div>", unsafe_allow_html=True)

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.markdown("<div class='side-card'>", unsafe_allow_html=True)
    st.markdown("<div class='side-title'>Profile</div>", unsafe_allow_html=True)
    st.markdown(f"**{PROFILE['name']}**", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted'>{PROFILE['tagline']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='small-muted' style='margin-top:0.35rem;'>{PROFILE['location']}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='side-card'>", unsafe_allow_html=True)
    st.markdown("<div class='side-title'>Jump to</div>", unsafe_allow_html=True)
    st.markdown(
        """
<a class="navbtn" href="#overview">Overview</a>
<a class="navbtn" href="#exploring">Exploring</a>
<a class="navbtn" href="#experience">Experience Timeline</a>
<a class="navbtn" href="#projects">Projects</a>
<a class="navbtn" href="#excursions">Industry Excursions</a>
<a class="navbtn" href="#notes">Notes</a>
<a class="navbtn" href="#skills">Skills</a>
<a class="navbtn" href="#education">Education</a>
<a class="navbtn" href="#languages">Languages</a>
<a class="navbtn" href="#contact">Contact</a>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Hero / Overview
# -------------------------------------------------
anchor("overview")
st.markdown("<div class='hero'>", unsafe_allow_html=True)
left, right = st.columns([0.70, 0.30], vertical_alignment="top")

with left:
    st.markdown(f"<div class='hero-title'>{PROFILE['name']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hero-tag'>{PROFILE['tagline']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='hero-meta'>{PROFILE['location']}</div>", unsafe_allow_html=True)
    st.markdown("<div style='height:0.55rem;'></div>", unsafe_allow_html=True)
    st.write(ABOUT)

with right:
    st.markdown("<div class='kpi'><div class='kpi-label'>Deployed Apps</div><div class='kpi-value'>2</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi'><div class='kpi-label'>Focus</div><div class='kpi-value'>AI + Analytics</div></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)
    st.markdown("<div class='kpi'><div class='kpi-label'>Strength</div><div class='kpi-value'>Dashboards + ML</div></div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Exploring
# -------------------------------------------------
anchor("exploring")
section("What’s Being Explored", "Applied labs, deployment patterns, and LLM workflows.")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write("\n".join([f"- {x}" for x in EXPLORING]))
st.markdown(pills(["Oracle Labs", "IBM Labs", "Embeddings", "LLMs", "Deployment Patterns"], accent=True), unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Experience Timeline
# -------------------------------------------------
anchor("experience")
section("Professional Experience Timeline", "Dashboard framing: Problem → Approach → Outcome.")
timeline_open()
for e in EXPERIENCE:
    timeline_item(e)
timeline_close()

# -------------------------------------------------
# Projects
# -------------------------------------------------
anchor("projects")
section("Projects", "Deployed dashboards. Links kept minimal and clean.")

for p in PROJECTS:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown(f"<div class='card-title'>{p['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-sub'>{p['one_liner']}</div>", unsafe_allow_html=True)
    st.markdown(pills(p["stack"]), unsafe_allow_html=True)

    st.markdown("**Summary**")
    st.write(p["summary"])

    st.markdown("**Highlights**")
    st.write("\n".join([f"- {x}" for x in p["highlights"]]))

    st.markdown(
        f"<div class='visit small-muted'> <a href='{p['live']}' target='_blank'>Visit app</a> "
        f"<span style='color:{BRAND['muted']};'>({p['live']})</span></div>",
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Industry Excursions (NEW SECTION)
# -------------------------------------------------
anchor("excursions")
section("Industry Excursions", "Applied exposure to enterprise AI labs and real-world research environments.")

for ex in INDUSTRY_EXCURSIONS:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-title'>{ex['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-sub'>{ex['meta']}</div>", unsafe_allow_html=True)
    st.write("\n".join([f"- {b}" for b in ex["bullets"]]))
    st.markdown(pills(ex["tags"], accent=True), unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Notes
# -------------------------------------------------
anchor("notes")
section("Notes", "Short technical notes: explainability, framing, and dashboard-first ML.")
for note in BLOG_NOTES:
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    st.markdown(f"<div class='card-title'>{note['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-sub'>{note['subtitle']}</div>", unsafe_allow_html=True)

    st.markdown(
        f"""
<div class="note-meta">
  <span class="small-muted">{note["date"]}</span>
  <span class="meta-dot"></span>
  <span class="small-muted">{note["read_time"]}</span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(pills(note["tags"], accent=True), unsafe_allow_html=True)

    st.write(note["lead"])

    with st.expander("Read full note"):
        st.write("\n".join([f"- {x}" for x in note["bullets"]]))

        st.markdown(
            f"""
<div class="callout">
  <strong>{note["callout_title"]}:</strong> {note["callout"]}
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("**" + note["code_title"] + "**")
        st.markdown(f"<div class='codebox'><pre>{note['code']}</pre></div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Skills
# -------------------------------------------------
anchor("skills")
section("Skills", "Core stack and strengths.")
cols = st.columns(3)
for i, (k, items) in enumerate(SKILLS.items()):
    with cols[i]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown(f"<div class='card-title'>{k}</div>", unsafe_allow_html=True)
        st.markdown(pills(items), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Education
# -------------------------------------------------
anchor("education")
section("Education", "Academic foundation with lab-driven skill acquisition.")
for ed in EDUCATION:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-title'>{ed['title']}</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='card-sub'>{ed['meta']}</div>", unsafe_allow_html=True)
    st.write("\n".join([f"- {b}" for b in ed["bullets"]]))
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Languages
# -------------------------------------------------
anchor("languages")
section("Languages")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write(LANGUAGES)
st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# Contact
# -------------------------------------------------
anchor("contact")
section("Contact")
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.write(f"- **Email:** {PROFILE['email']}")
st.write(f"- **Phone:** {PROFILE['phone']}")
st.write(f"- **LinkedIn:** {PROFILE['linkedin']}")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown(
    "<div class='small-muted' style='margin-top:1.2rem;'>Built with Streamlit • Deployed on Streamlit Cloud</div>",
    unsafe_allow_html=True,
)
