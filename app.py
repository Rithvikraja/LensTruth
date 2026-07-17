import streamlit as st
from streamlit_option_menu import option_menu
import json
import os
import time
from newspaper import Article

try:
    from groq import Groq
except ImportError:
    Groq = None

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="TruthLens AI",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# GROQ CLIENT
# =====================================================
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY) if (API_KEY and Groq) else None


def ask_ai(prompt, retries=3):
    """Send a prompt to Groq AI and return the raw text response."""
    if not client:
        return json.dumps({
            "error": "GROQ_API_KEY not set. Add it to your .env file or "
                     "environment / Streamlit secrets before deploying."
        })

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert AI News Fact Checker. "
                            "Always return ONLY valid JSON when requested."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=4096,
            )
            return response.choices[0].message.content

        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
            else:
                return json.dumps({"error": str(e)})


# =====================================================
# ARTICLE EXTRACTION
# =====================================================
def extract_article(url):
    article = Article(url)
    article.download()
    article.parse()
    return {
        "title": article.title,
        "author": ", ".join(article.authors) if article.authors else "Unknown",
        "text": article.text,
        "date": str(article.publish_date) if article.publish_date else "Unknown",
        "image": article.top_image,
    }


# =====================================================
# CLAIM EXTRACTION
# =====================================================
def extract_claims(article_text):
    prompt = f"""
You are an AI News Fact Checker.

Read the following article carefully.

Extract EVERY important claim.

For each claim classify it as one of:

- Fact
- Opinion
- Prediction
- Speculation

Return ONLY JSON, no markdown fences, no preamble.

Format:

{{
    "claims":[
        {{
            "claim":"...",
            "type":"Fact",
            "confidence":98
        }}
    ]
}}

Article:

{article_text}
"""
    return ask_ai(prompt)


def parse_claims_response(raw_response):
    """Safely strip markdown fences and parse the AI's JSON response."""
    cleaned = raw_response.replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    return data.get("claims", [])


# =====================================================
# TRANSPARENCY SCORE
# =====================================================
def calculate_transparency_score(
    verified_claims,
    total_claims,
    official_sources,
    trusted_sources,
    emotional_words
):
    """Calculate transparency score (0-100)."""
    score = 0

    # Claim Verification (40 Marks)
    if total_claims > 0:
        verification_ratio = verified_claims / total_claims
        score += verification_ratio * 40

    # Official Sources (25 Marks)
    score += min(official_sources * 5, 25)

    # Trusted Sources (20 Marks)
    score += min(trusted_sources * 4, 20)

    # Language Neutrality (15 Marks)
    language_score = max(15 - emotional_words, 0)
    score += language_score

    return round(score)

# =====================================================
# LOGO (top center)
# =====================================================
logo_col1, logo_col2, logo_col3 = st.columns([1, 1, 1])
with logo_col2:
    st.image("logo.png",use_container_width=True)
# =====================================================
# CSS
# =====================================================
st.markdown("""
<style>

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

.stApp {
    background: linear-gradient(-45deg, #0f172a, #111827, #1e3a8a, #0f172a);
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    color: white;
}

@keyframes gradient {
    0% {background-position: 0% 50%;}
    50% {background-position: 100% 50%;}
    100% {background-position: 0% 50%;}
}

.glass {
    background: rgba(255,255,255,0.08);
    backdrop-filter: blur(15px);
    padding: 20px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
}

.title {
    font-size: 55px;
    font-weight: bold;
    text-align: center;
    color: #38BDF8;
}

.subtitle {
    text-align: center;
    font-size: 22px;
    color: #CBD5E1;
    margin-bottom: 25px;
}

.footer {
    text-align: center;
    padding: 20px;
    color: #94A3B8;
}

.stButton>button {
    background: #2563EB;
    color: white;
    border-radius: 12px;
    height: 55px;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background: #1D4ED8;
    transform: scale(1.03);
    transition: 0.3s;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE (for Dashboard/Reports persistence)
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []  # list of {title, score, claims_count, verified}

# =====================================================
# NAVBAR
# =====================================================
selected = option_menu(
    menu_title=None,
    options=["Home", "Analyze", "Dashboard", "Reports"],
    icons=["house", "search", "bar-chart", "file-earmark"],
    orientation="horizontal",
)

# =====================================================
# HOME
# =====================================================
if selected == "Home":

    st.markdown('<p class="title">📰 TruthLens AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">Bringing Transparency to News</p>',
        unsafe_allow_html=True,
    )

    st.info("Know What's Verified • Understand What's Missing • Trust Facts, Not Noise")

    

    st.markdown("## 🚀 Features")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📰 Claim Extraction")
    with col2:
        st.info("✔ Claim Verification")
    with col3:
        st.info("📊 Transparency Score")

# =====================================================
# ANALYZE
# =====================================================
elif selected == "Analyze":

    st.title("🔍 Analyze News")

    left, right = st.columns([2, 1])

    with left:
        url = st.text_input("News URL", placeholder="https://newswebsite.com/article")
        st.write("### OR")
        article = st.text_area("Paste News Article", height=250)
        analyze = st.button("🔍 Analyze Article", use_container_width=True)

    with right:
        st.subheader("Quick Stats")
        stats_claims = st.empty()
        stats_transparency = st.empty()
        stats_confidence = st.empty()
        stats_claims.metric("Claims", "0")
        stats_transparency.metric("Transparency", "--")
        stats_confidence.metric("Confidence", "--")

    if analyze:

        if not url and not article:
            st.warning("Please paste a URL or article.")
        else:
            article_text = article
            article_title = "Pasted Article"

            # --- Extract from URL if provided ---
            if url:
                try:
                    data = extract_article(url)
                    st.success("✅ Article Extracted Successfully")

                    if data["image"]:
                        st.image(data["image"], use_container_width=True)

                    st.subheader(data["title"])
                    st.write("**Author:**", data["author"])
                    st.write("**Published:**", data["date"])

                    with st.expander("### Full Article Text"):
                        st.write(data["text"])

                    article_text = data["text"]
                    article_title = data["title"]

                except Exception as e:
                    st.error(f"Error extracting article: {e}")

            elif article:
                st.write("### Preview")
                st.write(article[:500])

            # --- Claim extraction ---
            if article_text:
                with st.spinner("🤖 AI is extracting factual claims..."):
                    raw_response = extract_claims(article_text)

                st.subheader("📋 Extracted Claims")

                try:
                    claims_list = parse_claims_response(raw_response)

                    if not claims_list:
                        st.warning("No claims were returned.")
                    else:
                        fact_count = 0
                        confidences = []

                        for i, item in enumerate(claims_list, start=1):
                            st.markdown("---")
                            st.markdown(f"## 📌 Claim {i}")
                            st.write(item.get("claim", ""))

                            confidence = item.get("confidence", 0)
                            claim_type = item.get("type", "Unknown")
                            confidences.append(confidence)

                            if claim_type == "Fact":
                                fact_count += 1
                                st.success(f"✅ FACT ({confidence}% confidence)")
                            elif claim_type == "Opinion":
                                st.warning(f"💬 OPINION ({confidence}% confidence)")
                            elif claim_type == "Prediction":
                                st.info(f"🔮 PREDICTION ({confidence}% confidence)")
                            else:
                                st.error(f"🤔 {claim_type.upper()} ({confidence}% confidence)")

                        # --- Transparency score ---
                        total_claims = len(claims_list)
                        avg_confidence = round(sum(confidences) / len(confidences)) if confidences else 0

                        score = calculate_transparency_score(
                            verified_claims=fact_count,
                            total_claims=total_claims,
                            official_sources=1,   # placeholder until source-detection is added
                            trusted_sources=1,    # placeholder until source-detection is added
                            emotional_words=0,    # placeholder until language analysis is added
                        )

                        st.markdown("---")
                        st.subheader("Transparency Score")
                        st.progress(score / 100)
                        st.metric("Transparency", f"{score}%")

                        if score >= 80:
                            st.success("🟢 Highly Transparent")
                        elif score >= 60:
                            st.warning("🟡 Moderately Transparent")
                        else:
                            st.error("🔴 Low Transparency")

                        stats_claims.metric("Claims", str(total_claims))
                        stats_transparency.metric("Transparency", f"{score}%")
                        stats_confidence.metric("Confidence", f"{avg_confidence}%")

                        # Save to session history for Dashboard/Reports
                        st.session_state.history.append({
                            "title": article_title,
                            "score": score,
                            "claims_count": total_claims,
                            "verified": fact_count,
                        })

                except Exception:
                    st.error("Could not parse AI response as JSON.")
                    st.code(raw_response)

# =====================================================
# DASHBOARD
# =====================================================
elif selected == "Dashboard":

    st.title("📊 Dashboard")

    history = st.session_state.history

    if not history:
        a, b, c, d = st.columns(4)
        with a:
            st.metric("Articles", "0")
        with b:
            st.metric("Verified", "0")
        with c:
            st.metric("Avg. Transparency", "--")
        with d:
            st.metric("Avg. Confidence", "--")
        st.info("Analyze an article to populate your dashboard.")
    else:
        total_articles = len(history)
        total_verified = sum(h["verified"] for h in history)
        avg_transparency = round(sum(h["score"] for h in history) / total_articles)

        a, b, c, d = st.columns(4)
        with a:
            st.metric("Articles", str(total_articles))
        with b:
            st.metric("Verified Claims", str(total_verified))
        with c:
            st.metric("Avg. Transparency", f"{avg_transparency}%")
        with d:
            st.metric("Total Claims", str(sum(h["claims_count"] for h in history)))

        st.write("")
        st.subheader("Recent Analyses")
        for h in reversed(history[-10:]):
            st.write(f"**{h['title']}** — {h['score']}% transparent, "
                      f"{h['verified']}/{h['claims_count']} claims verified")

# =====================================================
# REPORTS
# =====================================================
elif selected == "Reports":

    st.title("📄 Reports")

    if not st.session_state.history:
        st.info("No analyses yet. Downloadable reports will appear here after you analyze an article.")
    else:
        report_text = json.dumps(st.session_state.history, indent=2)
        st.download_button(
            "⬇ Download Report (JSON)",
            data=report_text,
            file_name="truthlens_report.json",
            mime="application/json",
        )
        st.code(report_text, language="json")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.markdown(
    """
    <div class="footer">
    <h3 style="color:white;">TruthLens AI</h3>
    <b>IDEA</b><br>
    Innovation • Dynamic • Empowerment • Attraction
    <br><br>
    © 2026 TruthLens AI™. All Rights Reserved.
    <br><br>
    Unauthorized reproduction or distribution of this software,
    its design, or its content is prohibited.
    <br><br>
    Version 1.0
    </div>
    """,
    unsafe_allow_html=True,
)
