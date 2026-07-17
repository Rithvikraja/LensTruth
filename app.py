import streamlit as st
from streamlit_option_menu import option_menu
import json
import os
import time
import base64
from io import BytesIO
from collections import Counter
from datetime import datetime
from newspaper import Article

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors as rl_colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, HRFlowable
)

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
    page_title="TruthLens — The Daily Fact Sheet",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# GROQ CLIENT
# =====================================================
API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=API_KEY) if (API_KEY and Groq) else None

# =====================================================
# COLOR PALETTE (shared between CSS + matplotlib + PDF)
# =====================================================
PAPER = "#F6F1E7"
PAPER_CARD = "#FDFBF6"
INK = "#1B1B1B"
INK_SOFT = "#4A4642"
MASTHEAD_RED = "#7A1B22"
GOLD = "#9C7A2E"
FACT_GREEN = "#285C3B"
PREDICTION_BLUE = "#2E5C8A"

CLAIM_COLOR_MAP = {
    "Fact": FACT_GREEN,
    "Opinion": GOLD,
    "Prediction": PREDICTION_BLUE,
    "Speculation": MASTHEAD_RED,
}


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

Read the following article carefully and do FOUR things in one pass:

1. Write a short, NEUTRAL overview of the article in 2-3 sentences —
   what happened, who is involved, and why it matters. Do not include
   opinion or judgement, just a plain summary.

2. Extract EVERY important claim and classify each as one of:
   - Fact
   - Opinion
   - Prediction
   - Speculation

3. Identify MISSING CONTENT — relevant context, opposing viewpoints,
   sources, data, or caveats that a transparent article should have
   included but did not. For each, note why it matters and how
   important the omission is (High, Medium, Low).

4. Count how many words or phrases in the article are emotionally
   charged / loaded language rather than neutral reporting (e.g.
   "slammed", "shocking", "disaster").

Return ONLY JSON, no markdown fences, no preamble.

Format:

{{
    "overview": "...",
    "claims":[
        {{
            "claim":"...",
            "type":"Fact",
            "confidence":98
        }}
    ],
    "missing_content":[
        {{
            "item":"...",
            "reason":"...",
            "importance":"High"
        }}
    ],
    "emotional_language_count": 3
}}

Article:

{article_text}
"""
    return ask_ai(prompt)


def parse_claims_response(raw_response):
    """Safely strip markdown fences and parse the AI's JSON response.
    Returns the full dict: overview, claims, missing_content, emotional_language_count."""
    cleaned = raw_response.replace("```json", "").replace("```", "").strip()
    data = json.loads(cleaned)
    return {
        "overview": data.get("overview", ""),
        "claims": data.get("claims", []),
        "missing_content": data.get("missing_content", []),
        "emotional_language_count": data.get("emotional_language_count", 0),
    }


def get_base64_of_file(path):
    """Read a local file and return its base64 string, or None if missing."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


# ---- PASTE YOUR BACKGROUND IMAGE HERE ----
# Put the image file next to app.py and set its filename below.
BACKGROUND_IMAGE_PATH = "background.jpeg"
_bg_base64 = get_base64_of_file(BACKGROUND_IMAGE_PATH)
if _bg_base64:
    _bg_ext = BACKGROUND_IMAGE_PATH.split(".")[-1]
    BACKGROUND_CSS = f"""
    .stApp {{
        background-image:
            linear-gradient(rgba(246,241,231,0.88), rgba(246,241,231,0.88)),
            url("data:image/{_bg_ext};base64,{_bg_base64}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    """
else:
    BACKGROUND_CSS = ""


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
    """Calculate transparency score (0-100).
    Measures how well the article SHOWS ITS WORK: sourcing + neutral language."""
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

    return round(min(score, 100))


# =====================================================
# CREDIBILITY SCORE
# =====================================================
def calculate_credibility_score(
    avg_confidence,
    fact_ratio,
    missing_count,
    emotional_words
):
    """Calculate credibility score (0-100).
    Measures how much the article's claims can be TRUSTED: AI confidence in
    claims, share of hard facts vs. opinion/speculation, and penalties for
    content gaps and loaded language."""
    score = 0

    # Average AI confidence across all claims (50 Marks)
    score += (avg_confidence / 100) * 50

    # Share of claims that are hard Facts, not Opinion/Prediction/Speculation (30 Marks)
    score += fact_ratio * 30

    # Penalty for missing context / omitted sources (up to 15 Marks)
    score += max(15 - (missing_count * 3), 0)

    # Penalty for emotionally loaded language (up to 5 Marks)
    score += max(5 - emotional_words, 0)

    return round(min(score, 100))


# Claim type -> newsroom stamp styling
CLAIM_STYLES = {
    "Fact":        {"label": "VERIFIED FACT", "class": "stamp-fact"},
    "Opinion":     {"label": "OPINION",       "class": "stamp-opinion"},
    "Prediction":  {"label": "PREDICTION",    "class": "stamp-prediction"},
    "Speculation": {"label": "SPECULATION",   "class": "stamp-speculation"},
}


def drop_cap_html(text, max_chars=1200):
    """Render a paragraph of text with a classic newspaper drop cap."""
    text = text.strip()
    if not text:
        return ""
    truncated = text[:max_chars]
    first_letter = truncated[0]
    rest = truncated[1:]
    suffix = "…" if len(text) > max_chars else ""
    return (
        f'<span class="dropcap">{first_letter}</span>{rest}{suffix}'
    )


# =====================================================
# MATPLOTLIB CHARTS (reused for on-screen + PDF)
# =====================================================
def create_score_chart(transparency, credibility):
    fig, ax = plt.subplots(figsize=(5, 2.6))
    fig.patch.set_facecolor(PAPER)
    ax.set_facecolor(PAPER)

    labels = ["Transparency", "Credibility"]
    values = [transparency, credibility]
    bar_colors = [INK, MASTHEAD_RED]

    bars = ax.barh(labels, values, color=bar_colors, height=0.5, zorder=3)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Score (%)", fontsize=9, color=INK_SOFT)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(axis="x", color=INK_SOFT, alpha=0.15, zorder=0)

    for bar, val in zip(bars, values):
        ax.text(
            min(val + 3, 96), bar.get_y() + bar.get_height() / 2,
            f"{val}%", va="center", fontsize=10, fontweight="bold", color=INK
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(INK_SOFT)

    fig.tight_layout()
    return fig


def create_claims_pie_chart(type_counts):
    labels = [k for k, v in type_counts.items() if v > 0]
    values = [v for v in type_counts.values() if v > 0]

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    fig.patch.set_facecolor(PAPER)

    if not values:
        labels, values = ["No Claims"], [1]
        pie_colors = [INK_SOFT]
    else:
        pie_colors = [CLAIM_COLOR_MAP.get(l, INK_SOFT) for l in labels]

    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda p: f"{p:.0f}%" if p > 0 else "",
        colors=pie_colors,
        textprops={"fontsize": 9, "color": INK},
        wedgeprops={"edgecolor": PAPER, "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_color(PAPER)
        at.set_fontweight("bold")

    ax.set_title("Claim Breakdown", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def fig_to_png_bytes(fig, dpi=170):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf


# =====================================================
# PDF REPORT GENERATION
# =====================================================
def build_pdf_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="TLTitle", fontName="Helvetica-Bold", fontSize=28,leading=34,
        textColor=rl_colors.HexColor(INK), spaceAfter=14, alignment=TA_CENTER
    ))
    styles.add(ParagraphStyle(
        name="TLKicker", fontName="Helvetica", fontSize=9,
        textColor=rl_colors.HexColor(INK_SOFT), spaceAfter=14,
        alignment=TA_CENTER, leading=12
    ))
    styles.add(ParagraphStyle(
        name="TLSection", fontName="Helvetica-Bold", fontSize=14,
        textColor=rl_colors.HexColor(INK), spaceBefore=14, spaceAfter=8,
        borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        name="TLBody", fontName="Helvetica", fontSize=10.5,
        textColor=rl_colors.HexColor(INK), leading=15
    ))
    styles.add(ParagraphStyle(
        name="TLMeta", fontName="Helvetica-Oblique", fontSize=9,
        textColor=rl_colors.HexColor(INK_SOFT), leading=13
    ))
    styles.add(ParagraphStyle(
        name="TLClaim", fontName="Helvetica", fontSize=10,
        textColor=rl_colors.HexColor(INK), leading=14, spaceAfter=2
    ))
    return styles


def generate_pdf_report(entry):
    """Build a newsroom-styled PDF report for one analyzed article.
    Returns a BytesIO buffer containing the PDF."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=letter,
        topMargin=0.6 * inch, bottomMargin=0.6 * inch,
        leftMargin=0.7 * inch, rightMargin=0.7 * inch,
        title=f"TruthLens Report — {entry.get('title', 'Article')}",
    )
    styles = build_pdf_styles()
    story = []

    # ---- Header ----
    story.append(Paragraph("TruthLens", styles["TLTitle"]))
    story.append(Paragraph(
        f"INDEPENDENT VERIFICATION DESK &nbsp;·&nbsp; FACT DESK REPORT<br/>"
        f"Generated {entry.get('analyzed_at', datetime.now().strftime('%d %B %Y, %H:%M'))}",
        styles["TLKicker"]
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=rl_colors.HexColor(INK)))
    story.append(Spacer(1, 12))

    # ---- Article title / meta ----
    story.append(Paragraph(entry.get("title", "Untitled Article"), styles["TLSection"]))
    story.append(Paragraph(
        f"Source: {entry.get('source', 'Pasted text')} &nbsp;|&nbsp; "
        f"Claims analyzed: {entry.get('claims_count', 0)} &nbsp;|&nbsp; "
        f"Verified facts: {entry.get('verified', 0)}",
        styles["TLMeta"]
    ))
    story.append(Spacer(1, 10))

    # ---- Overview ----
    story.append(Paragraph("Overview", styles["TLSection"]))
    story.append(Paragraph(
        entry.get("overview") or "No overview was generated for this article.",
        styles["TLBody"]
    ))
    story.append(Spacer(1, 10))

    # ---- Scores table ----
    story.append(Paragraph("Scores", styles["TLSection"]))
    score_table_data = [
        ["Metric", "Score"],
        ["Transparency Score", f"{entry.get('transparency_score', 0)}%"],
        ["Credibility Score", f"{entry.get('credibility_score', 0)}%"],
    ]
    score_table = Table(score_table_data, colWidths=[3 * inch, 1.5 * inch])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor(INK)),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.HexColor(PAPER)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("GRID", (0, 0), (-1, -1), 0.6, rl_colors.HexColor(INK_SOFT)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.HexColor(PAPER_CARD)]),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 12))

    # ---- Score bar chart ----
    score_fig = create_score_chart(
        entry.get("transparency_score", 0), entry.get("credibility_score", 0)
    )
    score_png = fig_to_png_bytes(score_fig)
    plt.close(score_fig)
    story.append(RLImage(score_png, width=5.5 * inch, height=2.9 * inch))
    story.append(Spacer(1, 6))

    # ---- Claim breakdown pie chart ----
    pie_fig = create_claims_pie_chart(entry.get("claim_type_counts", {}))
    pie_png = fig_to_png_bytes(pie_fig)
    plt.close(pie_fig)
    story.append(RLImage(pie_png, width=3.6 * inch, height=3.6 * inch))
    story.append(Spacer(1, 10))

    # ---- Missing content ----
    missing_content = entry.get("missing_content", [])
    if missing_content:
        story.append(Paragraph("Missing Context", styles["TLSection"]))
        for i, gap in enumerate(missing_content, start=1):
            story.append(Paragraph(
                f"<b>{i}. {gap.get('item', '')}</b> "
                f"[{gap.get('importance', 'Medium').upper()} PRIORITY]",
                styles["TLClaim"]
            ))
            if gap.get("reason"):
                story.append(Paragraph(f"<i>{gap.get('reason')}</i>", styles["TLMeta"]))
        story.append(Spacer(1, 10))

    # ---- Footer ----
    story.append(HRFlowable(width="100%", thickness=0.8, color=rl_colors.HexColor(INK_SOFT)))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        "© TruthLens AI™ — Automated analysis. Verify important claims independently.",
        styles["TLMeta"]
    ))

    doc.build(story)
    buf.seek(0)
    return buf


# =====================================================
# CSS — NEWSPRINT EDITORIAL THEME
# =====================================================
TODAY = datetime.now().strftime("%A, %d %B %Y").upper()

st.markdown(f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,500;0,700;0,900;1,600&family=PT+Serif:ital,wght@0,400;0,700;1,400&family=Inter:wght@400;500;600;700&display=swap');

#MainMenu {{visibility: hidden;}}
footer {{visibility: hidden;}}
header {{visibility: hidden;}}

:root {{
    --paper: #F6F1E7;
    --paper-card: #FDFBF6;
    --ink: #1B1B1B;
    --ink-soft: #4A4642;
    --rule: #1B1B1B;
    --masthead-red: #7A1B22;
    --gold: #9C7A2E;
}}

html, body, .stApp {{
    background: var(--paper);
    color: var(--ink);
}}

.stApp {{
    background-image:
        radial-gradient(circle at 15% 20%, rgba(0,0,0,0.015) 0, transparent 45%),
        radial-gradient(circle at 85% 80%, rgba(0,0,0,0.015) 0, transparent 45%);
}}

{BACKGROUND_CSS}

* {{
    font-family: 'PT Serif', Georgia, serif;
}}

h1, h2, h3, .masthead, .section-title {{
    font-family: 'Playfair Display', Georgia, serif;
}}

/* ---------- MASTHEAD ---------- */
.masthead-wrap {{
    text-align: center;
    padding-top: 18px;
    margin-bottom: 4px;
}}

.masthead-kicker {{
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    letter-spacing: 4px;
    color: var(--ink-soft);
    text-transform: uppercase;
    margin-bottom: 6px;
}}

.masthead {{
    font-size: 68px;
    font-weight: 900;
    letter-spacing: 1px;
    color: var(--ink);
    line-height: 1;
    margin: 0;
}}

.masthead-sub {{
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 18px;
    color: var(--ink-soft);
    margin-top: 6px;
}}

.dateline {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--ink-soft);
    padding: 10px 4px 6px 4px;
    border-top: 3px solid var(--rule);
    border-bottom: 1px solid var(--rule);
    margin-top: 14px;
}}

.double-rule {{
    border: none;
    border-top: 1px solid var(--rule);
    margin: 2px 0 22px 0;
}}

/* ---------- WIDGET LABELS (force visible ink color) ---------- */
div[data-testid="stWidgetLabel"],
div[data-testid="stWidgetLabel"] * {{
    color: var(--ink) !important;
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 0.5px;
    opacity: 1 !important;
}}

/* ---------- METRIC LABEL/VALUE (force visible) ---------- */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] * {{
    color: var(--ink-soft) !important;
    opacity: 1 !important;
}}
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] * {{
    color: var(--ink) !important;
    opacity: 1 !important;
}}

/* ---------- BORDERED CONTAINERS (st.container(border=True)) act as our cards ---------- */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--paper-card) !important;
    border: 1px solid rgba(27,27,27,0.35) !important;
    border-top: 3px solid var(--ink) !important;
    border-radius: 0 !important;
    box-shadow: 4px 4px 0px rgba(27,27,27,0.06);
    padding: 6px 8px;
}}

/* ---------- ARTICLE / GLASS CARDS ---------- */
.glass, .article-card {{
    background: var(--paper-card);
    padding: 26px 30px;
    border: 1px solid rgba(27,27,27,0.35);
    border-top: 3px solid var(--ink);
    box-shadow: 4px 4px 0px rgba(27,27,27,0.06);
    margin-bottom: 20px;
}}

.section-title {{
    font-size: 26px;
    font-weight: 700;
    border-bottom: 2px solid var(--ink);
    padding-bottom: 8px;
    margin-bottom: 16px;
}}

.eyebrow {{
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: var(--masthead-red);
    font-weight: 700;
    margin-bottom: 4px;
}}

.dropcap {{
    float: left;
    font-family: 'Playfair Display', serif;
    font-size: 64px;
    line-height: 52px;
    font-weight: 900;
    padding: 4px 8px 0 0;
    color: var(--masthead-red);
}}

/* ---------- CLAIM STAMPS ---------- */
.claim-block {{
    border-bottom: 1px dashed rgba(27,27,27,0.35);
    padding: 16px 0;
}}
.claim-number {{
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    letter-spacing: 2px;
    color: var(--ink-soft);
    text-transform: uppercase;
}}
.claim-text {{
    font-size: 18px;
    line-height: 1.5;
    margin: 6px 0 10px 0;
}}
.stamp {{
    display: inline-block;
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    padding: 4px 10px;
    border: 1.5px solid currentColor;
    text-transform: uppercase;
    transform: rotate(-1deg);
}}
.stamp-fact {{ color: #285C3B; }}
.stamp-opinion {{ color: #9C7A2E; }}
.stamp-prediction {{ color: #2E5C8A; }}
.stamp-speculation {{ color: var(--masthead-red); }}
.stamp-missing {{ color: #B8410E; }}

/* ---------- METRICS ---------- */
div[data-testid="stMetric"] {{
    background: var(--paper-card);
    border: 1px solid rgba(27,27,27,0.35);
    padding: 12px 14px;
    text-align: center;
}}
div[data-testid="stMetricLabel"] {{
    font-family: 'Inter', sans-serif !important;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-size: 11px !important;
    color: var(--ink-soft) !important;
}}
div[data-testid="stMetricValue"] {{
    font-family: 'Playfair Display', serif !important;
    color: var(--ink) !important;
}}

/* ---------- ALERTS (recolor to muted newsprint tone) ---------- */
div[data-testid="stAlert"] {{
    background: var(--paper-card) !important;
    border: 1px solid rgba(27,27,27,0.3) !important;
    border-left: 4px solid var(--ink) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
}}
div[data-testid="stAlert"] p {{
    color: var(--ink) !important;
    font-family: 'PT Serif', serif !important;
}}

/* ---------- BUTTONS ---------- */
.stButton>button, .stDownloadButton>button {{
    background: var(--ink);
    color: var(--paper);
    border-radius: 0;
    height: 52px;
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    letter-spacing: 2px;
    text-transform: uppercase;
    border: none;
}}
.stButton>button:hover, .stDownloadButton>button:hover {{
    background: var(--masthead-red);
    color: white;
}}

/* ---------- INPUTS ---------- */
.stTextInput>div>div>input, .stTextArea textarea {{
    background: var(--paper-card) !important;
    border: 1px solid rgba(27,27,27,0.4) !important;
    border-radius: 0 !important;
    color: var(--ink) !important;
    font-family: 'PT Serif', serif !important;
}}

/* ---------- PROGRESS BAR ---------- */
div[data-testid="stProgress"] > div > div {{
    background-color: var(--masthead-red) !important;
}}

/* ---------- FOOTER ---------- */
.footer {{
    text-align: center;
    padding: 30px 0 10px 0;
    color: var(--ink-soft);
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    letter-spacing: 1px;
    border-top: 3px solid var(--rule);
    margin-top: 40px;
}}
.footer h3 {{
    font-family: 'Playfair Display', serif;
    color: var(--ink);
    margin-bottom: 6px;
}}

/* =====================================================
   PAGE-FLIP NAVIGATION ANIMATION (pure CSS, no JS)

   Streamlit re-renders a section's markup as brand-new DOM nodes
   whenever you switch tabs (Home/Analyze/Dashboard/Reports), since
   each tab renders a structurally different tree. Widgets that
   persist across reruns of the SAME tab (typing in a text box,
   clicking a button) get their existing DOM nodes updated in place
   rather than recreated, so the animation below naturally plays
   only on real navigation, not on every rerun.
===================================================== */
.main .block-container {{
    perspective: 2000px;
    -webkit-perspective: 2000px;
}}

@keyframes tlPageFlip {{
    0% {{
        transform: rotateY(-10deg) translateX(-18px) scale(0.98);
        opacity: 0;
    }}
    55% {{
        opacity: 1;
    }}
    100% {{
        transform: rotateY(0deg) translateX(0) scale(1);
        opacity: 1;
    }}
}}

.masthead-wrap,
.dateline,
.glass,
.article-card,
.section-title,
.claim-block,
.eyebrow,
div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stMetric"] {{
    animation: tlPageFlip 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
    transform-origin: left center;
    -webkit-transform-origin: left center;
    backface-visibility: hidden;
    -webkit-backface-visibility: hidden;
}}

/* stagger cards/blocks slightly so a whole section doesn't flip
   as one flat plane — gives it a bit more of a "turning" feel */
.article-card:nth-of-type(2),
div[data-testid="column"]:nth-of-type(2) div[data-testid="stMetric"] {{
    animation-delay: 0.05s;
}}
.article-card:nth-of-type(3),
div[data-testid="column"]:nth-of-type(3) div[data-testid="stMetric"] {{
    animation-delay: 0.1s;
}}

</style>
""", unsafe_allow_html=True)


# =====================================================
# LOGO (centered, top)
# =====================================================
LOGO_PATH = "logo.png"
if os.path.exists(LOGO_PATH):
    logo_l, logo_c, logo_r = st.columns([2, 1, 2])
    with logo_c:
        st.image(LOGO_PATH, use_container_width=True)

# =====================================================
# MASTHEAD
# =====================================================
st.markdown('<div class="masthead-wrap">', unsafe_allow_html=True)
st.markdown('<div class="masthead-kicker">Est. 2026 · Independent Verification Desk</div>', unsafe_allow_html=True)
st.markdown('<p class="masthead">TruthLens</p>', unsafe_allow_html=True)
st.markdown('<p class="masthead-sub">Bringing Transparency to the News</p>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown(f"""
<div class="dateline">
    <span>{TODAY}</span>
    <span>Fact Desk · Vol. I</span>
    <span>Powered by AI Analysis</span>
</div>
<hr class="double-rule">
""", unsafe_allow_html=True)

# =====================================================
# SESSION STATE (for Dashboard/Reports persistence)
# =====================================================
if "history" not in st.session_state:
    st.session_state.history = []  # list of full entry dicts, see Analyze section

# =====================================================
# NAVBAR
# =====================================================
selected = option_menu(
    menu_title=None,
    options=["Home", "Analyze", "Dashboard", "Reports"],
    icons=["house", "search", "bar-chart", "file-earmark"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#F6F1E7"},
        "icon": {"color": "#7A1B22", "font-size": "15px"},
        "nav-link": {
            "font-family": "Inter, sans-serif",
            "font-size": "13px",
            "letter-spacing": "1.5px",
            "text-transform": "uppercase",
            "color": "#1B1B1B",
            "text-align": "center",
            "margin": "0px",
            "padding": "10px 18px",
            "--hover-color": "#EDE6D6",
        },
        "nav-link-selected": {
            "background-color": "#1B1B1B",
            "color": "#F6F1E7",
            "font-weight": "700",
        },
    },
)

# =====================================================
# HOME
# =====================================================
if selected == "Home":

    lede_html = drop_cap_html(
        "Every day, newsrooms and social feeds publish thousands of claims — "
        "facts, opinions, predictions and speculation, often blurred together. "
        "TruthLens reads an article the way a skeptical editor would: it separates "
        "verifiable fact from framing, scores the piece for transparency and "
        "credibility, and gives you a clear paper trail — with a downloadable PDF "
        "report — of what to trust and what to question.",
        max_chars=2000,
    )
    st.markdown(f"""
    <div class="glass">
        <div class="eyebrow">Front Page</div>
        <p class="section-title">Know What's Verified. Understand What's Missing.</p>
        <div>{lede_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title" style="margin-top:10px;">Desk Sections</p>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="article-card"><div class="eyebrow">Section A</div>'
                    '<b>Claim Extraction</b><br>Every factual assertion, pulled and numbered like a wire report.</div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="article-card"><div class="eyebrow">Section B</div>'
                    '<b>Transparency &amp; Credibility</b><br>Two scores: how well it shows its work, and how much to trust it.</div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="article-card"><div class="eyebrow">Section C</div>'
                    '<b>PDF Reports</b><br>A printable, chart-backed report for every article you analyze.</div>',
                    unsafe_allow_html=True)

# =====================================================
# ANALYZE
# =====================================================
elif selected == "Analyze":

    st.markdown('<div class="eyebrow">Newsroom Desk</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Analyze a Story</p>', unsafe_allow_html=True)

    left, right = st.columns([2, 1])

    with left:
        with st.container(border=True):
            st.markdown("""
            <div class="field-label">
                NEWS URL
            </div>
            """, unsafe_allow_html=True)

            url = st.text_input(
                "",
                placeholder="https://newswebsite.com/article",
                label_visibility="collapsed"
            )
            st.markdown("<p style='font-family:Inter,sans-serif;font-size:12px;letter-spacing:1.5px;"
                        "text-transform:uppercase;color:#4A4642;'>— or —</p>", unsafe_allow_html=True)
            st.markdown("""
            <div class="field-label">
                PASTE NEWS ARTICLE
            </div>
            """, unsafe_allow_html=True)

            article = st.text_area(
                "",
                height=220,
                label_visibility="collapsed"
            )
            analyze = st.button("Run the Story", use_container_width=True)


    with right:
        with st.container(border=True):
            st.markdown('<div class="eyebrow">Quick Stats</div>', unsafe_allow_html=True)
            stats_claims = st.empty()
            stats_transparency = st.empty()
            stats_credibility = st.empty()
            stats_claims.metric("Claims", "0")
            stats_transparency.metric("Transparency", "--")
            stats_credibility.metric("Credibility", "--")

    if analyze:

        if not url and not article:
            st.warning("Please paste a URL or article.")
        else:
            article_text = article
            article_title = "Pasted Article"
            article_source = "Pasted text"

            # --- Extract from URL if provided ---
            if url:
                try:
                    data = extract_article(url)
                    st.success("Article extracted successfully.")

                    with st.container(border=True):
                        if data["image"]:
                            st.image(data["image"], use_container_width=True)

                        st.markdown(f'<p class="section-title">{data["title"]}</p>', unsafe_allow_html=True)
                        st.markdown(f'<span class="eyebrow">By {data["author"]} · {data["date"]}</span>',
                                    unsafe_allow_html=True)

                        with st.expander("Full Article Text"):
                            st.markdown(drop_cap_html(data["text"], max_chars=4000), unsafe_allow_html=True)

                    article_text = data["text"]
                    article_title = data["title"]
                    article_source = url

                except Exception as e:
                    st.error(f"Error extracting article: {e}")

            elif article:
                with st.container(border=True):
                    st.markdown('<div class="eyebrow">Preview</div>', unsafe_allow_html=True)
                    st.write(article[:500])

            # --- Claim extraction ---
            if article_text:
                with st.spinner("The desk is fact-checking this story…"):
                    raw_response = extract_claims(article_text)

                try:
                    result = parse_claims_response(raw_response)
                    overview = result["overview"]
                    claims_list = result["claims"]
                    missing_content = result["missing_content"]
                    emotional_count = result["emotional_language_count"]

                    if not claims_list:
                        st.warning("No claims were returned.")
                    else:
                        # --- Overview ---
                        if overview:
                            st.markdown('<p class="section-title">Overview</p>', unsafe_allow_html=True)
                            with st.container(border=True):
                                st.markdown(f'<div class="claim-text">{overview}</div>', unsafe_allow_html=True)

                        st.markdown('<p class="section-title">Extracted Claims</p>', unsafe_allow_html=True)

                        fact_count = 0
                        confidences = []
                        claim_type_counts = Counter()

                        with st.container(border=True):
                            for i, item in enumerate(claims_list, start=1):
                                confidence = item.get("confidence", 0)
                                claim_type = item.get("type", "Unknown")
                                confidences.append(confidence)
                                claim_type_counts[claim_type] += 1
                                style = CLAIM_STYLES.get(
                                    claim_type, {"label": claim_type.upper(), "class": "stamp-speculation"}
                                )

                                if claim_type == "Fact":
                                    fact_count += 1

                                st.markdown(f"""
                                <div class="claim-block">
                                    <div class="claim-number">Claim {i:02d}</div>
                                    <div class="claim-text">{item.get("claim", "")}</div>
                                    <span class="stamp {style['class']}">{style['label']} · {confidence}%</span>
                                </div>
                                """, unsafe_allow_html=True)

                        # --- Missing content ---
                        if missing_content:
                            st.markdown('<p class="section-title">Missing Context</p>', unsafe_allow_html=True)
                            with st.container(border=True):
                                for i, gap in enumerate(missing_content, start=1):
                                    importance = gap.get("importance", "Medium")
                                    st.markdown(f"""
                                    <div class="claim-block">
                                        <div class="claim-number">Gap {i:02d}</div>
                                        <div class="claim-text">{gap.get("item", "")}</div>
                                        <div style="font-style:italic;color:var(--ink-soft);margin-bottom:6px;">
                                            {gap.get("reason", "")}
                                        </div>
                                        <span class="stamp stamp-missing">{importance.upper()} PRIORITY</span>
                                    </div>
                                    """, unsafe_allow_html=True)

                        # --- Scores ---
                        total_claims = len(claims_list)
                        avg_confidence = round(sum(confidences) / len(confidences)) if confidences else 0
                        fact_ratio = (fact_count / total_claims) if total_claims else 0

                        transparency_score = calculate_transparency_score(
                            verified_claims=fact_count,
                            total_claims=total_claims,
                            official_sources=1,   # placeholder until source-detection is added
                            trusted_sources=1,    # placeholder until source-detection is added
                            emotional_words=emotional_count,
                        )
                        credibility_score = calculate_credibility_score(
                            avg_confidence=avg_confidence,
                            fact_ratio=fact_ratio,
                            missing_count=len(missing_content),
                            emotional_words=emotional_count,
                        )

                        st.markdown('<p class="section-title">Transparency &amp; Credibility</p>', unsafe_allow_html=True)
                        with st.container(border=True):
                            score_col, chart_col = st.columns([1, 1])
                            with score_col:
                                st.progress(transparency_score / 100, text=f"Transparency — {transparency_score}%")
                                st.progress(credibility_score / 100, text=f"Credibility — {credibility_score}%")

                                if transparency_score >= 80:
                                    st.success("Highly Transparent — well-sourced and clearly labeled.")
                                elif transparency_score >= 60:
                                    st.warning("Moderately Transparent — some claims need scrutiny.")
                                else:
                                    st.error("Low Transparency — proceed with caution.")

                            with chart_col:
                                score_fig = create_score_chart(transparency_score, credibility_score)
                                st.pyplot(score_fig, use_container_width=True)
                                plt.close(score_fig)

                            pie_fig = create_claims_pie_chart(dict(claim_type_counts))
                            pc1, pc2, pc3 = st.columns([1, 2, 1])
                            with pc2:
                                st.pyplot(pie_fig, use_container_width=True)
                            plt.close(pie_fig)

                        stats_claims.metric("Claims", str(total_claims))
                        stats_transparency.metric("Transparency", f"{transparency_score}%")
                        stats_credibility.metric("Credibility", f"{credibility_score}%")

                        # --- Save to session history for Dashboard/Reports ---
                        entry = {
                            "title": article_title,
                            "source": article_source,
                            "overview": overview,
                            "transparency_score": transparency_score,
                            "credibility_score": credibility_score,
                            "claims_count": total_claims,
                            "verified": fact_count,
                            "claim_type_counts": dict(claim_type_counts),
                            "missing_content": missing_content,
                            "emotional_count": emotional_count,
                            "analyzed_at": datetime.now().strftime("%d %B %Y, %H:%M"),
                        }
                        st.session_state.history.append(entry)

                        # --- Inline PDF download for this analysis ---
                        pdf_buf = generate_pdf_report(entry)
                        st.download_button(
                            "Download PDF Report",
                            data=pdf_buf,
                            file_name=f"truthlens_{article_title[:40].strip().replace(' ', '_') or 'report'}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

                except Exception:
                    st.error("Could not parse AI response as JSON.")
                    st.code(raw_response)

# =====================================================
# DASHBOARD
# =====================================================
elif selected == "Dashboard":

    st.markdown('<div class="eyebrow">Editorial Metrics</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Dashboard</p>', unsafe_allow_html=True)

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
            st.metric("Avg. Credibility", "--")
        st.info("Analyze an article to populate your dashboard.")
    else:
        total_articles = len(history)
        total_verified = sum(h["verified"] for h in history)
        avg_transparency = round(sum(h["transparency_score"] for h in history) / total_articles)
        avg_credibility = round(sum(h["credibility_score"] for h in history) / total_articles)

        a, b, c, d = st.columns(4)
        with a:
            st.metric("Articles", str(total_articles))
        with b:
            st.metric("Verified Claims", str(total_verified))
        with c:
            st.metric("Avg. Transparency", f"{avg_transparency}%")
        with d:
            st.metric("Avg. Credibility", f"{avg_credibility}%")

        st.markdown('<p class="section-title" style="margin-top:20px;">Recent Analyses</p>', unsafe_allow_html=True)
        with st.container(border=True):
            for h in reversed(history[-10:]):
                st.markdown(f"""
                <div class="claim-block">
                    <div class="claim-number">On the Record</div>
                    <div class="claim-text"><b>{h['title']}</b> — {h['transparency_score']}% transparent,
                    {h['credibility_score']}% credible, {h['verified']}/{h['claims_count']} claims verified</div>
                </div>
                """, unsafe_allow_html=True)

# =====================================================
# REPORTS
# =====================================================
elif selected == "Reports":

    st.markdown('<div class="eyebrow">Archive</div>', unsafe_allow_html=True)
    st.markdown('<p class="section-title">Reports</p>', unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No analyses yet. Full PDF reports will appear here after you analyze an article.")
    else:
        for idx, entry in enumerate(reversed(st.session_state.history)):
            real_idx = len(st.session_state.history) - idx
            with st.expander(f"📰  {entry['title']}  ·  {entry['analyzed_at']}", expanded=(idx == 0)):

                st.markdown('<div class="eyebrow">Overview</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="claim-text">{entry.get("overview") or "No overview available."}</div>',
                    unsafe_allow_html=True
                )

                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.metric("Transparency", f"{entry['transparency_score']}%")
                with m2:
                    st.metric("Credibility", f"{entry['credibility_score']}%")
                with m3:
                    st.metric("Claims", str(entry["claims_count"]))
                with m4:
                    st.metric("Verified", str(entry["verified"]))

                chart_col1, chart_col2 = st.columns([1, 1])
                with chart_col1:
                    score_fig = create_score_chart(entry["transparency_score"], entry["credibility_score"])
                    st.pyplot(score_fig, use_container_width=True)
                    plt.close(score_fig)
                with chart_col2:
                    pie_fig = create_claims_pie_chart(entry.get("claim_type_counts", {}))
                    st.pyplot(pie_fig, use_container_width=True)
                    plt.close(pie_fig)

                if entry.get("missing_content"):
                    st.markdown('<div class="eyebrow">Missing Context</div>', unsafe_allow_html=True)
                    for gap in entry["missing_content"]:
                        st.markdown(
                            f"- **{gap.get('item', '')}** "
                            f"_(​{gap.get('importance', 'Medium')} priority)_ — {gap.get('reason', '')}"
                        )

                pdf_buf = generate_pdf_report(entry)
                st.download_button(
                    "Download PDF Report",
                    data=pdf_buf,
                    file_name=f"truthlens_{entry['title'][:40].strip().replace(' ', '_') or 'report'}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key=f"pdf_dl_{real_idx}",
                )

# =====================================================
# FOOTER
# =====================================================
st.markdown(
    """
    <div class="footer">
    <h3>TruthLens</h3>
    IDEA — Innovation · Dynamic · Empowerment · Attraction
    <br><br>
    © 2026 TruthLens AI™. All Rights Reserved.
    <br>
    Unauthorized reproduction or distribution of this software,
    its design, or its content is prohibited.
    <br><br>
    Version 2.1 — Editorial Edition (PDF Reports · Page-Flip Navigation)
    </div>
    """,
    unsafe_allow_html=True,
)
