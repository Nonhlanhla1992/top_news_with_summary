import os
import re
import requests
import streamlit as st
from collections import Counter

# -------------------------------------------------
# App config
# -------------------------------------------------
st.set_page_config(page_title="Top News + Clean Briefing", layout="centered")
st.title("🗞️ Top News")

API_KEY = os.getenv("NEWSDATA_API_KEY")
if not API_KEY:
    st.error("Missing API key. In Terminal run:\nexport NEWSDATA_API_KEY='YOUR_KEY'")
    st.stop()

BASE_URL = "https://newsdata.io/api/1/latest"

# -------------------------------------------------
# Fetch up to N headlines
# -------------------------------------------------
@st.cache_data(ttl=900)
def fetch_top_n(api_key: str, n: int = 100, language: str = "en", max_calls: int = 8, category: str | None = None):
    items = []
    page_token = None

    for _ in range(max_calls):
        params = {"apikey": api_key, "language": language}
        if category:
            params["category"] = category
        if page_token:
            params["page"] = page_token

        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()

        if data.get("status") != "success":
            raise RuntimeError(str(data))

        items.extend(data.get("results") or [])

        if len(items) >= n:
            break

        page_token = data.get("nextPage")
        if not page_token:
            break

    return items[:n]

# -------------------------------------------------
# Text utilities
# -------------------------------------------------
STOP = set("""
a an the and or but if then than so to of in on for with from by at as is are was were be been being
this that these those it its into over under about across after before during between also not no
can could would should may might will just more most much very per via
""".split())

SPORTS = set("""
nba nfl nhl mlb match game games season playoff all-star dunk overtime quarter finals championship
wrestle wrestling division scoreboard wildcats lakers
""".split())

LIFESTYLE_LOCAL = set("""
wedding weddings horoscope crossword recipe recipes dining travel scenic waterfall park
town county local citizen newsletter obituaries health scores
""".split())

FINANCE_TICKER = set("""
nyse nasdaq etf stock stocks shares bond bonds yield earnings ticker short interest price target
""".split())

def normalize(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text: str):
    return [
        w for w in normalize(text).split()
        if w not in STOP and len(w) > 2 and not w.isdigit()
    ]

def is_low_signal(title: str, desc: str) -> bool:
    ws = set(tokenize((title or "") + " " + (desc or "")))
    if len(ws & SPORTS) >= 2:
        return True
    if len(ws & LIFESTYLE_LOCAL) >= 2:
        return True
    if len(ws & FINANCE_TICKER) >= 3:
        return True
    return False

# -------------------------------------------------
# Storyline clustering
# -------------------------------------------------
def jaccard(a, b):
    A, B = set(a), set(b)
    if not A or not B:
        return 0.0
    return len(A & B) / (len(A | B) or 1)

def cluster_titles(rows, sim_threshold=0.33):
    clusters = []
    for r in rows:
        title = (r.get("title") or "").strip()
        if not title:
            continue
        tks = tokenize(title)
        if not tks:
            continue

        best_i = None
        best_sim = 0.0
        for i, c in enumerate(clusters):
            centroid = [w for w, _ in c["tok_counts"].most_common(25)]
            sim = jaccard(tks, centroid)
            if sim > best_sim:
                best_sim, best_i = sim, i

        if best_i is not None and best_sim >= sim_threshold:
            clusters[best_i]["rows"].append(r)
            clusters[best_i]["tok_counts"].update(tks)
        else:
            clusters.append({"rows":[r], "tok_counts":Counter(tks)})

    clusters.sort(key=lambda c: len(c["rows"]), reverse=True)
    return clusters

# -------------------------------------------------
# Clean narrative paragraph (no %s, no useless intro)
# -------------------------------------------------
TOPIC_LEXICON = {
    "Geopolitics & security": ["war","military","nuclear","sanctions","border","defense","attack","navy"],
    "Elections & governance": ["election","vote","parliament","government","president","minister","court","policy"],
    "Economy & markets": ["inflation","gdp","economy","bank","interest","rate","currency","trade","jobs","markets"],
    "Tech & AI": ["ai","artificial","chip","cyber","data","software","platform"],
    "Climate & disasters": ["climate","flood","storm","drought","wildfire","earthquake"],
    "Public safety & crime": ["police","arrest","trial","fraud","shooting","crime"]
}

def label_cluster(cluster):
    counts = cluster["tok_counts"]
    best_topic = None
    best_score = 0
    for topic, words in TOPIC_LEXICON.items():
        score = sum(counts.get(w, 0) for w in words)
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic or "Other"

def representative_title(cluster):
    counts = cluster["tok_counts"]
    def score(t):
        return sum(counts.get(w, 0) for w in tokenize(t))
    rows = cluster["rows"]
    rows_sorted = sorted(rows, key=lambda r: (-score(r.get("title","")), len(r.get("title","") or "")))
    return rows_sorted[0].get("title","")

def build_clean_paragraph(rows):
    if not rows:
        return "No headlines were available to summarize."

    filtered = [r for r in rows if not is_low_signal(r.get("title",""), r.get("description",""))]
    corpus = filtered if len(filtered) >= 40 else rows

    clusters = cluster_titles(corpus)
    if not clusters:
        return "Insufficient signal to produce a coherent summary."

    # label clusters
    for c in clusters:
        c["topic"] = label_cluster(c)

    # choose top 3 distinct-topic clusters
    chosen = []
    seen = set()
    for c in clusters:
        if c["topic"] not in seen:
            chosen.append(c)
            seen.add(c["topic"])
        if len(chosen) == 3:
            break
    if len(chosen) < 3:
        chosen = clusters[:3]

    # build narrative
    lines = []
    for c in chosen:
        rep = representative_title(c)
        topic = c["topic"]
        lines.append(f"{topic}: {rep}")

    paragraph = (
        f"The current headlines point to several parallel developments. "
        f"{lines[0]}. "
        f"{lines[1]}. "
        f"{lines[2]}. "
        f"Taken together, the feed reflects a dispersed news cycle rather than a single dominant global event."
    )

    return paragraph

# -------------------------------------------------
# UI
# -------------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    n = st.slider("Headlines to pull and summarize", min_value=20, max_value=100, value=100, step=10)
with col2:
    category = st.selectbox("Category", ["(all)", "world", "politics", "business", "technology", "environment"], index=0)

cat_param = None if category == "(all)" else category

try:
    results = fetch_top_n(API_KEY, n=n, category=cat_param)
except Exception as e:
    st.error("Could not fetch news.")
    st.caption(str(e))
    st.stop()

st.subheader("🧠 Summary (paragraph)")
st.write(build_clean_paragraph(results))
st.divider()

st.subheader("🗂️ Top 10 articles")
for a in results[:10]:
    title = a.get("title", "Untitled")
    link = a.get("link", "")
    source = a.get("source_id", "")
    pub = a.get("pubDate", "")

    st.markdown(f"**{title}**")
    st.caption(f"{pub} | {source}")
    if link:
        st.markdown(f"[Open article]({link})")
    st.divider()
