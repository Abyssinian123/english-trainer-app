#!/usr/bin/env python3
"""
Single-Context English Trainer  v2.0
=====================================
A one-stop English learning application wrapping Listening, Reading, Writing,
Speaking, and Spaced Repetition into a single daily session — backed by SLA theory.

Refactored for: bulletproof state management, academic UI, manual vocab selection,
summary evaluation, model-answer comparison, voice diary, static article fallback,
and SM-2 verified spaced repetition.

Usage:
    pip install -r requirements.txt
    streamlit run english_trainer.py

DeepSeek API key: set in .streamlit/secrets.toml (DEEPSEEK_API_KEY) or env var.
"""

from __future__ import annotations

import streamlit as st
import sqlite3
import os
import json
import time
import re
import urllib.parse
from datetime import date, datetime, timedelta
from typing import Optional, Any
from pathlib import Path

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    OpenAI = None  # type: ignore

# ============================================================================
# PATHS & CONSTANTS
# ============================================================================
APP_DIR = Path(__file__).resolve().parent
DB_PATH = str(APP_DIR / "english_trainer.db")
STATIC_ARTICLES_DIR = APP_DIR / "static_articles"
STATIC_ARTICLES_DIR.mkdir(exist_ok=True)

DEEPSEEK_BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

ESSAY_TOPICS = [
    "Public Health — preventive medicine and life expectancy",
    "Economics — the sharing economy and labour markets",
    "Social Sciences — social media and interpersonal communication",
    "Technology — artificial intelligence in everyday decision-making",
    "Environment — carbon capture economics and renewable energy",
    "Education — the flipped classroom and student autonomy",
    "Psychology — habit formation and lifelong learning",
    "Urban Studies — the 15-minute city and community well-being",
    "Globalisation — cultural homogenisation versus diversity",
    "Ethics — data privacy in the age of surveillance capitalism",
]

DIFFICULTY_LABELS = {
    "Intermediate (B2)": "B2 – upper-intermediate, clear structure, moderate vocabulary",
    "Advanced (C1)": "C1 – advanced, complex sentences, rich vocabulary",
    "Proficiency (C2)": "C2 – near-native, sophisticated argumentation, nuanced lexis",
}

WORD_COUNT_OPTIONS = {
    "Short (~150 words)": 150,
    "Medium (~250 words)": 250,
    "Long (~400 words)": 400,
}

# ============================================================================
# DATABASE  (SQLite)
# ============================================================================
def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables and migrate legacy schemas."""
    with get_db() as conn:
        # -- sessions --
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                date            TEXT    NOT NULL UNIQUE,
                topic           TEXT,
                content_text    TEXT    NOT NULL,
                user_summary    TEXT,
                summary_score   TEXT,
                summary_feedback TEXT,
                ai_question     TEXT,
                user_output     TEXT,
                ai_corrections  TEXT,
                ai_polish       TEXT,
                ai_model_answer TEXT,
                ai_comparison   TEXT,
                difficulty      TEXT    DEFAULT 'Intermediate (B2)',
                word_count      INTEGER DEFAULT 250,
                created_at      TEXT    DEFAULT (datetime('now'))
            )
        """)

        # -- vocab_bank --
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vocab_bank (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                expression        TEXT    NOT NULL,
                definition        TEXT    NOT NULL,
                derivatives       TEXT    DEFAULT '',
                collocations      TEXT    DEFAULT '',
                example           TEXT    DEFAULT '',
                source_text       TEXT    DEFAULT '',
                category          TEXT    DEFAULT 'Uncategorized',
                date_added        TEXT    DEFAULT (date('now')),
                interval          INTEGER DEFAULT 1,
                ease_factor       REAL    DEFAULT 2.5,
                next_review_date  TEXT    DEFAULT (date('now', '+1 day')),
                review_count      INTEGER DEFAULT 0,
                last_reviewed     TEXT
            )
        """)

        # -- essay_bank --
        conn.execute("""
            CREATE TABLE IF NOT EXISTS essay_bank (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                title         TEXT,
                content_text  TEXT    NOT NULL,
                topic         TEXT,
                difficulty    TEXT    DEFAULT 'Intermediate (B2)',
                word_count    INTEGER DEFAULT 250,
                category      TEXT    DEFAULT 'Uncategorized',
                source_type   TEXT    DEFAULT 'ai',
                audio_path    TEXT,
                date_added    TEXT    DEFAULT (date('now'))
            )
        """)

        # -- static_articles --
        conn.execute("""
            CREATE TABLE IF NOT EXISTS static_articles (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                title         TEXT    NOT NULL,
                content_text  TEXT    NOT NULL,
                topic         TEXT,
                difficulty    TEXT    DEFAULT 'Intermediate (B2)',
                word_count    INTEGER,
                audio_path    TEXT
            )
        """)

        # -- Migrations: add columns if they don't exist (SQLite-compatible) --
        _migrate_add_column(conn, "vocab_bank", "derivatives", "TEXT DEFAULT ''")
        _migrate_add_column(conn, "vocab_bank", "collocations", "TEXT DEFAULT ''")
        _migrate_add_column(conn, "vocab_bank", "category", "TEXT DEFAULT 'Uncategorized'")
        _migrate_add_column(conn, "sessions", "summary_score", "TEXT")
        _migrate_add_column(conn, "sessions", "summary_feedback", "TEXT")
        _migrate_add_column(conn, "sessions", "ai_model_answer", "TEXT")
        _migrate_add_column(conn, "sessions", "ai_comparison", "TEXT")
        _migrate_add_column(conn, "sessions", "difficulty", "TEXT DEFAULT 'Intermediate (B2)'")
        _migrate_add_column(conn, "sessions", "word_count", "INTEGER DEFAULT 250")


def _migrate_add_column(conn: sqlite3.Connection, table: str, column: str, col_type: str):
    """Add a column if it does not already exist (safe for SQLite)."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        pass  # column already exists


# ---------------------------------------------------------------------------
# Vocab Bank CRUD
# ---------------------------------------------------------------------------
def save_vocab(expression: str, definition: str, derivatives: str = "",
               collocations: str = "", example: str = "",
               source_text: str = "", category: str = "Uncategorized") -> bool:
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM vocab_bank WHERE expression = ?", (expression,)
        ).fetchone()
        if existing:
            return False
        conn.execute(
            """INSERT INTO vocab_bank
               (expression, definition, derivatives, collocations, example, source_text, category)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (expression, definition, derivatives, collocations, example, source_text, category),
        )
    return True


def get_vocab_due_for_review(limit: int = 5) -> list[dict]:
    today = date.today().isoformat()
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM vocab_bank
               WHERE next_review_date <= ?
               ORDER BY next_review_date ASC, review_count ASC
               LIMIT ?""",
            (today, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def update_vocab_review(vocab_id: int, quality: int):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM vocab_bank WHERE id = ?", (vocab_id,)).fetchone()
        if not row:
            return
        item = dict(row)
        old_interval = item["interval"]
        ease = item["ease_factor"]
        count = item["review_count"] + 1

        if quality < 3:
            new_interval = 1
            ease = max(1.3, ease - 0.2)
        elif quality == 3:
            new_interval = max(1, int(old_interval * 1.3))
        elif quality == 4:
            new_interval = max(2, int(old_interval * ease))
            ease += 0.1
        else:
            new_interval = max(3, int(old_interval * ease * 1.3))
            ease += 0.15

        next_review = (date.today() + timedelta(days=new_interval)).isoformat()
        conn.execute(
            """UPDATE vocab_bank
               SET interval=?, ease_factor=?, next_review_date=?,
                   review_count=?, last_reviewed=?
               WHERE id=?""",
            (new_interval, round(ease, 2), next_review, count,
             date.today().isoformat(), vocab_id),
        )


def get_all_vocab(search: str = "", category: str = "") -> list[dict]:
    with get_db() as conn:
        query = "SELECT * FROM vocab_bank WHERE 1=1"
        params: list[Any] = []
        if search:
            query += " AND (expression LIKE ? OR definition LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        if category and category != "All":
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY date_added DESC"
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_vocab_categories() -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM vocab_bank WHERE category != '' ORDER BY category"
        ).fetchall()
    return [r["category"] for r in rows]


def update_vocab_category(vocab_id: int, category: str):
    with get_db() as conn:
        conn.execute("UPDATE vocab_bank SET category=? WHERE id=?", (category, vocab_id))


def delete_vocab(vocab_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM vocab_bank WHERE id=?", (vocab_id,))


def get_vocab_stats() -> dict:
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) as c FROM vocab_bank").fetchone()["c"]
        due = conn.execute(
            "SELECT COUNT(*) as c FROM vocab_bank WHERE next_review_date <= ?",
            (date.today().isoformat(),),
        ).fetchone()["c"]
        mastered = conn.execute(
            "SELECT COUNT(*) as c FROM vocab_bank WHERE interval >= 30"
        ).fetchone()["c"]
    return {"total": total, "due": due, "mastered": mastered}


# ---------------------------------------------------------------------------
# Essay Bank CRUD
# ---------------------------------------------------------------------------
def save_essay_to_bank(title: str, content_text: str, topic: str = "",
                       difficulty: str = "Intermediate (B2)", word_count: int = 250,
                       category: str = "Uncategorized", source_type: str = "ai",
                       audio_path: str = "") -> bool:
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM essay_bank WHERE content_text = ?", (content_text,)
        ).fetchone()
        if existing:
            return False
        conn.execute(
            """INSERT INTO essay_bank (title, content_text, topic, difficulty, word_count, category, source_type, audio_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (title, content_text, topic, difficulty, word_count, category, source_type, audio_path),
        )
    return True


def get_all_essays(search: str = "", category: str = "") -> list[dict]:
    with get_db() as conn:
        query = "SELECT * FROM essay_bank WHERE 1=1"
        params: list[Any] = []
        if search:
            query += " AND (title LIKE ? OR topic LIKE ?)"
            params.extend([f"%{search}%", f"%{search}%"])
        if category and category != "All":
            query += " AND category = ?"
            params.append(category)
        query += " ORDER BY date_added DESC"
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_essay_categories() -> list[str]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT DISTINCT category FROM essay_bank WHERE category != '' ORDER BY category"
        ).fetchall()
    return [r["category"] for r in rows]


def update_essay_category(essay_id: int, category: str):
    with get_db() as conn:
        conn.execute("UPDATE essay_bank SET category=? WHERE id=?", (category, essay_id))


def delete_essay(essay_id: int):
    with get_db() as conn:
        conn.execute("DELETE FROM essay_bank WHERE id=?", (essay_id,))


# ---------------------------------------------------------------------------
# Static Articles CRUD
# ---------------------------------------------------------------------------
def get_all_static_articles() -> list[dict]:
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM static_articles ORDER BY title").fetchall()
    return [dict(r) for r in rows]


def get_static_article_by_id(article_id: int) -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM static_articles WHERE id=?", (article_id,)).fetchone()
    return dict(row) if row else None


# ---------------------------------------------------------------------------
# Session CRUD
# ---------------------------------------------------------------------------
def save_session_to_db():
    """Persist current session state to the sessions table (upsert by date)."""
    essay = st.session_state.get("essay_text", "")
    if not essay:
        return
    today = date.today().isoformat()
    with get_db() as conn:
        existing = conn.execute("SELECT id FROM sessions WHERE date=?", (today,)).fetchone()
        fields = {
            "content_text": essay,
            "topic": st.session_state.get("essay_topic", ""),
            "user_summary": st.session_state.get("user_summary", ""),
            "summary_score": st.session_state.get("summary_score", ""),
            "summary_feedback": st.session_state.get("summary_feedback", ""),
            "ai_question": st.session_state.get("ai_question", ""),
            "user_output": st.session_state.get("user_output", ""),
            "ai_corrections": st.session_state.get("ai_corrections", ""),
            "ai_polish": st.session_state.get("ai_polish", ""),
            "ai_model_answer": st.session_state.get("ai_model_answer", ""),
            "ai_comparison": st.session_state.get("ai_comparison", ""),
            "difficulty": st.session_state.get("essay_difficulty", "Intermediate (B2)"),
            "word_count": st.session_state.get("essay_word_count", 250),
        }
        if existing:
            set_clause = ", ".join(f"{k}=?" for k in fields)
            conn.execute(
                f"UPDATE sessions SET {set_clause} WHERE date=?",
                list(fields.values()) + [today],
            )
        else:
            cols = ", ".join(fields.keys())
            placeholders = ", ".join("?" for _ in fields)
            conn.execute(
                f"INSERT INTO sessions (date, {cols}) VALUES (?, {placeholders})",
                [today] + list(fields.values()),
            )


def get_today_session() -> Optional[dict]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE date=?", (date.today().isoformat(),)
        ).fetchone()
    return dict(row) if row else None


# ============================================================================
# AI SERVICE  (DeepSeek via OpenAI SDK)
# ============================================================================
def get_client() -> OpenAI:
    api_key = None
    try:
        api_key = st.secrets["DEEPSEEK_API_KEY"]
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        st.error("DeepSeek API key not found. Set DEEPSEEK_API_KEY in .streamlit/secrets.toml or as an environment variable.")
        st.stop()
    if not HAS_OPENAI:
        st.error("The `openai` package is required. Install it with: pip install openai")
        st.stop()
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def call_deepseek(system_prompt: str, user_prompt: str, temperature: float = 0.7,
                  max_tokens: int = 1024) -> str:
    client = get_client()
    try:
        resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"API call failed: {e}")
        return ""


def _strip_json(raw: str) -> str:
    """Remove markdown code fences from a string."""
    s = raw.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)
    return s


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
def generate_essay(topic: str, difficulty: str, word_count: int) -> str:
    system = (
        "You are an IELTS examiner and expert English materials writer. "
        f"Write an essay at {DIFFICULTY_LABELS.get(difficulty, difficulty)} level. "
        f"Target approximately {word_count} words. "
        "Structure: clear introduction, body paragraphs, and conclusion. "
        "Use the specified vocabulary range and sentence complexity. "
        "Return ONLY the essay — no title, no meta-commentary, no self-introduction."
    )
    prompt = f"Topic: {topic}\n\nWrite the essay:"
    return call_deepseek(system, prompt, temperature=0.8, max_tokens=word_count * 3)


def evaluate_summary(essay: str, user_summary: str) -> dict:
    system = (
        "You are an IELTS examiner evaluating a learner's concise summary of an essay. "
        "The learner was asked to write a short paragraph capturing the main ideas. "
        "Evaluate on TWO criteria:\n"
        "1. Key Information Coverage — did they capture the core thesis and the most important supporting points?\n"
        "2. Structural Accuracy — did they faithfully represent the essay's argument structure without distortion?\n\n"
        "Return ONLY valid JSON with these keys:\n"
        '- "score": a string like "8/10"\n'
        '- "key_coverage": assessment of whether main ideas were captured (1-2 sentences)\n'
        '- "structural_accuracy": assessment of argument structure fidelity (1-2 sentences)\n'
        '- "gaps": what important points were missed or misunderstood\n'
        '- "improvement": specific, actionable advice for better summarisation\n'
        "No code fences, no extra text."
    )
    prompt = (
        f"ESSAY:\n{essay}\n\n"
        f"LEARNER'S SUMMARY:\n{user_summary}\n\n"
        "Evaluate on Key Information Coverage and Structural Accuracy. Return JSON only."
    )
    raw = call_deepseek(system, prompt, temperature=0.3, max_tokens=500)
    try:
        return json.loads(_strip_json(raw))
    except json.JSONDecodeError:
        return {
            "score": "N/A",
            "key_coverage": "Could not parse evaluation.",
            "structural_accuracy": "Could not parse evaluation.",
            "gaps": "N/A",
            "improvement": "Try to capture the main argument and key supporting points in a concise paragraph.",
        }


def lookup_vocab_batch(words: list[str], context: str) -> list[dict]:
    """Given a list of words/phrases, return definition, derivatives, and collocations."""
    word_list = "\n".join(f"- {w}" for w in words)
    system = (
        "You are an expert English lexicographer and EFL materials writer. "
        "For each word or phrase listed, provide:\n"
        "1. A clear, concise English definition (1 sentence)\n"
        "2. Common derivatives / word-family members (派生词) — list 2-4 related forms\n"
        "3. Frequent collocations (常见搭配) — list 3-5 natural word partnerships\n\n"
        "Return ONLY a valid JSON array. Each element must have keys: "
        '"expression", "definition", "derivatives", "collocations". '
        "Derivatives and collocations should be comma-separated strings. "
        "No code fences, no extra commentary."
    )
    prompt = (
        f"Context (the essay these words appear in):\n{context[:500]}\n\n"
        f"Words/phrases to look up:\n{word_list}"
    )
    raw = call_deepseek(system, prompt, temperature=0.3, max_tokens=800)
    try:
        result = json.loads(_strip_json(raw))
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    # Fallback
    return [
        {
            "expression": w,
            "definition": f"(Definition unavailable for '{w}')",
            "derivatives": "",
            "collocations": "",
        }
        for w in words
    ]


def generate_question(essay: str) -> str:
    system = (
        "You are an IELTS speaking/writing examiner. Based on the essay, write ONE "
        "thought-provoking open-ended question that demands critical thinking. "
        "The question should require the learner to take a position, provide reasons, "
        "and use advanced vocabulary in a ~150-word response. Return ONLY the question."
    )
    prompt = f"Essay:\n{essay}\n\nGenerate the discussion question:"
    return call_deepseek(system, prompt, temperature=0.9, max_tokens=150)


def generate_feedback_full(user_text: str, essay_topic: str, question: str,
                           essay_text: str) -> dict:
    """Return corrections, polish, model answer, and comparison."""
    system = (
        "You are an IELTS writing examiner (Band 8-9 standard). "
        "A learner has responded to a discussion question based on an essay. "
        "Perform ALL of the following tasks. Output each section under its EXACT header:\n\n"
        "###CORRECTIONS###\n"
        "Fix grammar, word-choice, and collocation errors in the learner's text. "
        "Keep the learner's voice and structure. Be precise.\n\n"
        "###POLISH###\n"
        "Rewrite the learner's ideas with sophisticated vocabulary and sentence variety — "
        "Band-8 level. Keep the same core arguments.\n\n"
        "###MODEL_ANSWER###\n"
        "Write your OWN model answer to the question (~150 words). This should demonstrate "
        "an ideal Band-9 response with nuanced reasoning and academic vocabulary.\n\n"
        "###COMPARISON###\n"
        "Compare the learner's critical thinking direction with the model answer. "
        "Is their logic sound? Are there weaknesses in reasoning? What did they do well? "
        "Be constructive and specific. 3-5 sentences."
    )
    prompt = (
        f"ESSAY CONTEXT:\n{essay_text[:600]}\n\n"
        f"TOPIC: {essay_topic}\n\n"
        f"QUESTION: {question}\n\n"
        f"LEARNER'S RESPONSE:\n{user_text}\n\n"
        "Provide corrections, polish, model answer, and comparison as specified."
    )
    raw = call_deepseek(system, prompt, temperature=0.5, max_tokens=1200)

    def _extract(tag: str) -> str:
        pattern = rf"###{tag}###\s*(.*?)(?=###|\Z)"
        m = re.search(pattern, raw, re.DOTALL)
        return m.group(1).strip() if m else ""

    return {
        "corrections": _extract("CORRECTIONS") or "(could not parse)",
        "polish": _extract("POLISH") or "(could not parse)",
        "model_answer": _extract("MODEL_ANSWER") or "(could not parse)",
        "comparison": _extract("COMPARISON") or "(could not parse)",
    }


def evaluate_voice_diary(transcript: str, topic_hint: str = "") -> dict:
    """Evaluate a spoken entry for oral delivery, grammar, and native expression."""
    system = (
        "You are an expert speaking coach and IELTS examiner. "
        "Evaluate the following spoken transcript for:\n"
        "1. Oral delivery style (fluency, naturalness, pacing hints)\n"
        "2. Grammatical accuracy (flag specific errors)\n"
        "3. Native expression level (collocations, idiomaticity, register)\n\n"
        "Return ONLY valid JSON:\n"
        '{"fluency": "...", "grammar_feedback": "...", "expression_level": "...", "overall_score": "X/10", "improvement_tips": "..."}\n'
        "No code fences."
    )
    prompt = (
        f"{'Topic context: ' + topic_hint if topic_hint else 'Free-speaking entry.'}\n\n"
        f"Transcript:\n{transcript}\n\n"
        "Evaluate as specified. Return JSON only."
    )
    raw = call_deepseek(system, prompt, temperature=0.4, max_tokens=600)
    try:
        return json.loads(_strip_json(raw))
    except json.JSONDecodeError:
        return {
            "fluency": "Could not evaluate.",
            "grammar_feedback": "Could not evaluate.",
            "expression_level": "Could not evaluate.",
            "overall_score": "N/A",
            "improvement_tips": "Try again with a clearer transcript.",
        }


# ============================================================================
# STATIC ARTICLE SEED DATA
# ============================================================================
def seed_static_articles():
    """Insert sample static articles if the table is empty."""
    with get_db() as conn:
        count = conn.execute("SELECT COUNT(*) as c FROM static_articles").fetchone()["c"]
        if count > 0:
            return

    articles = [
        {
            "title": "The Hidden Cost of Fast Fashion",
            "content_text": (
                "The global fashion industry produces over 100 billion garments annually, "
                "making it one of the largest polluters on the planet. Fast fashion, characterised "
                "by rapid production cycles and low-cost materials, has democratised style but at "
                "a severe environmental and social cost. The average consumer now purchases 60% more "
                "clothing than they did two decades ago, yet keeps each item for half as long.\n\n"
                "The environmental toll begins with raw material extraction. Cotton, the most widely "
                "used natural fibre, requires vast quantities of water — a single t-shirt can consume "
                "up to 2,700 litres. Synthetic fibres such as polyester, derived from petroleum, shed "
                "microplastics with every wash, contributing to the estimated 1.4 million trillion "
                "plastic particles currently floating in our oceans.\n\n"
                "Socially, the pressure to reduce costs has driven manufacturing to countries with "
                "weak labour protections. Garment workers, predominantly women, often earn less than "
                "a living wage and face unsafe working conditions. The 2013 Rana Plaza collapse in "
                "Bangladesh, which killed over 1,100 workers, brought global attention to these issues, "
                "yet meaningful reform has been frustratingly slow.\n\n"
                "Solutions are emerging from multiple directions. Circular fashion models emphasise "
                "durability, repairability, and recycling. Brands are experimenting with rental "
                "platforms and take-back schemes. Consumers, particularly younger generations, are "
                "increasingly voting with their wallets, favouring sustainable labels and second-hand "
                "marketplaces. Governments are also stepping in — the European Union's Strategy for "
                "Sustainable Textiles mandates that by 2030 all textile products sold in the EU must "
                "be durable, repairable, and recyclable. Whether these measures can reverse decades "
                "of overconsumption remains an open question."
            ),
            "topic": "Environment — fashion industry sustainability",
            "difficulty": "Advanced (C1)",
            "word_count": 280,
            "audio_path": "",
        },
        {
            "title": "The Promise and Peril of AI in Healthcare",
            "content_text": (
                "Artificial intelligence is reshaping healthcare with breathtaking speed. Machine "
                "learning algorithms now detect cancers in medical images with accuracy rivalling "
                "or exceeding that of experienced radiologists. Natural language processing systems "
                "mine electronic health records to predict patient deterioration hours before a "
                "human clinician would notice. These advances promise earlier diagnosis, personalised "
                "treatment plans, and reduced medical errors.\n\n"
                "Yet the integration of AI into clinical practice raises profound questions. "
                "Algorithmic bias is a pressing concern: models trained predominantly on data from "
                "wealthy, white populations may perform poorly for minority groups, potentially "
                "widening existing health disparities. A widely cited 2019 study found that a "
                "commercial algorithm used in US hospitals systematically underestimated the health "
                "needs of Black patients compared to equally sick White patients.\n\n"
                "Data privacy presents another challenge. Medical data is among the most sensitive "
                "personal information, and the appetite of AI companies for vast training datasets "
                "creates tension with patient confidentiality. Regulatory frameworks such as GDPR "
                "in Europe and HIPAA in the United States were designed before the current AI "
                "revolution, leaving significant gaps in protection.\n\n"
                "The path forward requires a delicate balance. AI should augment rather than replace "
                "clinical judgment; the technology works best when it empowers doctors, not when it "
                "operates as a black box. Transparency in model development, diverse training data, "
                "and robust regulatory oversight are essential. If these conditions are met, AI "
                "could help deliver on the long-standing promise of precision medicine — the right "
                "treatment for the right patient at the right time."
            ),
            "topic": "Technology — AI in medicine",
            "difficulty": "Advanced (C1)",
            "word_count": 255,
            "audio_path": "",
        },
        {
            "title": "Rethinking the University Degree",
            "content_text": (
                "For decades, a university degree has been sold as the gold standard of career "
                "preparation. Parents save for years, students take on debt, and governments "
                "subsidise the system on the assumption that higher education reliably produces "
                "productive, employable citizens. But cracks are appearing in this consensus.\n\n"
                "The economic calculus is shifting. In many developed economies, the wage premium "
                "associated with a bachelor's degree has stagnated or declined, while tuition costs "
                "have continued to climb. In the United States, outstanding student loan debt has "
                "surpassed $1.7 trillion, a figure that weighs heavily on the life choices of an "
                "entire generation. Meanwhile, major employers including Google, Apple, and IBM have "
                "dropped degree requirements for many roles, signalling a shift toward skills-based "
                "hiring.\n\n"
                "Alternative pathways are proliferating. Coding bootcamps, industry certifications, "
                "apprenticeships, and micro-credentials offer faster, cheaper routes to employable "
                "skills. Online platforms like Coursera and edX bring Ivy League courses to anyone "
                "with an internet connection. These alternatives do not carry the same social cachet "
                "as a traditional degree, but they are increasingly recognised by employers desperate "
                "for specific competencies.\n\n"
                "The future is unlikely to be binary — degrees or no degrees. A more plausible "
                "scenario is a hybrid model in which universities unbundle their offerings, combining "
                "short technical certifications with broader liberal arts education. The institutions "
                "that thrive will be those that embrace flexibility, lifelong learning, and genuine "
                "alignment with the needs of a rapidly changing labour market."
            ),
            "topic": "Education — value of university degrees",
            "difficulty": "Intermediate (B2)",
            "word_count": 240,
            "audio_path": "",
        },
        {
            "title": "The Psychology of Digital Minimalism",
            "content_text": (
                "The average smartphone user checks their device 58 times a day, with half of those "
                "checks occurring during working hours. Notifications, designed by teams of behavioural "
                "psychologists, exploit the same neural reward pathways that make slot machines "
                "addictive. The result is a global attention crisis, with measurable consequences for "
                "mental health, productivity, and the quality of our relationships.\n\n"
                "Digital minimalism, a philosophy popularised by computer scientist Cal Newport, "
                "proposes a radical rethink: rather than using technology mindlessly by default, "
                "we should intentionally curate our digital lives to support our values and goals. "
                "This is not a Luddite rejection of technology but a deliberate, selective approach. "
                "A digital minimalist might keep a smartphone but delete all social media apps; they "
                "might schedule specific times for email rather than responding reactively throughout "
                "the day.\n\n"
                "The evidence supporting this approach is mounting. Studies have found that reducing "
                "social media use to 30 minutes per day significantly decreases loneliness and "
                "depression. Workers who batch their email into two or three daily sessions report "
                "lower stress and higher productivity than those who maintain constant connectivity. "
                "The mechanism is straightforward: deep, focused work and meaningful social interaction "
                "both require sustained attention — precisely the resource that fragmentary technology "
                "erodes.\n\n"
                "Implementing digital minimalism is challenging in a world designed for maximal "
                "engagement. It requires what psychologists call 'friction' — deliberately making "
                "undesired behaviours harder. Deleting apps, using website blockers, and keeping "
                "phones out of the bedroom are all friction strategies. The goal is not to eliminate "
                "technology but to reclaim agency over one's own attention, treating it as the finite "
                "and precious resource it truly is."
            ),
            "topic": "Psychology — digital habits and attention",
            "difficulty": "Intermediate (B2)",
            "word_count": 265,
            "audio_path": "",
        },
    ]

    with get_db() as conn:
        for a in articles:
            conn.execute(
                """INSERT INTO static_articles (title, content_text, topic, difficulty, word_count, audio_path)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (a["title"], a["content_text"], a["topic"], a["difficulty"], a["word_count"], a["audio_path"]),
            )


# ============================================================================
# CUSTOM CSS  (Academic Minimalist)
# ============================================================================
def inject_css():
    st.markdown("""
    <style>
    /* ---- Font ---- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,400;14..32,500;14..32,600;14..32,700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,500;8..60,600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* ---- Hide default header toolbar & top chrome ---- */
    header[data-testid="stHeader"] {
        display: none !important;
    }
    div[data-testid="stToolbar"] {
        display: none !important;
    }
    #MainMenu {
        display: none !important;
    }
    footer {
        display: none !important;
    }
    /* Kill the massive top padding Streamlit injects */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0.5rem !important;
    }
    div[data-testid="stAppViewContainer"] {
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }

    /* ---- Main background ---- */
    .stApp {
        background: #f7f8fa;
    }

    /* ---- Sidebar ---- */
    section[data-testid="stSidebar"] {
        min-width: 220px !important;
        max-width: 240px !important;
        background: #f0f2f5;
        border-right: 1px solid #dde1e6;
    }
    section[data-testid="stSidebar"] * {
        color: #2c3e50 !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-weight: 500;
        font-size: 0.92rem;
        padding: 6px 10px;
        border-radius: 6px;
        transition: background 0.15s;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: #e2e6eb;
    }
    /* Hide the "Navigation" label */
    section[data-testid="stSidebar"] .stRadio > label:first-child {
        display: none;
    }

    /* ---- Cards ---- */
    .card {
        background: #ffffff;
        border: 1px solid #dde1e6;
        border-radius: 10px;
        padding: 24px 28px;
        margin: 14px 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
    }
    .card h3, .card h4 {
        margin-top: 0;
        color: #1e293b;
        font-weight: 700;
    }

    /* ---- Step pills ---- */
    .step-row {
        display: flex;
        gap: 10px;
        margin: 8px 0 24px 0;
        align-items: stretch;
    }
    .step-pill {
        flex: 1;
        padding: 18px 10px;
        border-radius: 10px;
        text-align: center;
        font-size: 0.95rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        transition: all 0.2s ease;
        border: 1.5px solid transparent;
    }
    .step-pill.active {
        background: #e8edf4;
        border-color: #4a6fa5;
        color: #1e3a5f;
        box-shadow: 0 2px 10px rgba(74,111,165,0.12);
    }
    .step-pill.done {
        background: #eef2e8;
        border-color: #8aa87d;
        color: #3d5a2e;
    }
    .step-pill.pending {
        background: #f5f6f8;
        border-color: #e2e6eb;
        color: #94a3b8;
    }
    .step-pill .step-num {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        opacity: 0.7;
    }

    /* ---- Vocab card ---- */
    .vocab-entry {
        background: #ffffff;
        border: 1px solid #dde1e6;
        border-left: 4px solid #4a6fa5;
        border-radius: 8px;
        padding: 18px 20px;
        margin: 10px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    }
    .vocab-entry .expr {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 8px;
    }
    .vocab-entry .meta {
        font-size: 0.85rem;
        color: #64748b;
        line-height: 1.6;
    }
    .vocab-entry .meta strong {
        color: #475569;
    }

    /* ---- Essay text block ---- */
    .essay-text {
        font-family: 'Source Serif 4', 'Georgia', serif;
        font-size: 1.05rem;
        line-height: 1.85;
        color: #334155;
        background: #fcfcfd;
        border: 1px solid #e8ecf0;
        border-radius: 8px;
        padding: 24px 28px;
        white-space: pre-wrap;
    }

    /* ---- Flashcard ---- */
    .flashcard-front, .flashcard-back {
        background: #ffffff;
        border: 1.5px solid #dde1e6;
        border-radius: 14px;
        padding: 36px 28px;
        text-align: center;
        min-height: 180px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transition: all 0.25s ease;
    }
    .flashcard-front .word {
        font-size: 2rem;
        font-weight: 800;
        color: #1e3a5f;
        letter-spacing: -0.3px;
    }
    .flashcard-back .word {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 10px;
    }
    .flashcard-back .def {
        font-size: 1rem;
        color: #475569;
        margin-bottom: 6px;
    }
    .flashcard-back .deriv {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 4px;
    }
    .flashcard-back .colloc {
        font-size: 0.88rem;
        color: #64748b;
        margin-bottom: 8px;
    }
    .flashcard-back .ex {
        font-size: 0.92rem;
        color: #64748b;
        font-style: italic;
        background: #f8f9fb;
        padding: 10px 14px;
        border-radius: 8px;
    }

    /* ---- Buttons ---- */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.15s ease !important;
        border: 1px solid transparent !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    .stButton > button[kind="primary"] {
        background: #4a6fa5 !important;
        border-color: #4a6fa5 !important;
        color: #fff !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: #3d5d8a !important;
    }

    /* ---- Inputs ---- */
    textarea, input[type="text"], select {
        border-radius: 8px !important;
        border: 1.5px solid #dde1e6 !important;
        transition: border-color 0.15s ease !important;
    }
    textarea:focus, input[type="text"]:focus {
        border-color: #4a6fa5 !important;
        box-shadow: 0 0 0 3px rgba(74,111,165,0.08) !important;
    }

    /* ---- Progress bar ---- */
    div[data-testid="stProgress"] > div > div {
        background: linear-gradient(90deg, #4a6fa5, #7a9bb5) !important;
    }

    /* ---- Success / Info boxes ---- */
    div[data-testid="stSuccess"] {
        background: #eef3ea !important;
        border-left: 4px solid #6a9b5e !important;
        border-radius: 8px !important;
    }

    /* ---- Dividers ---- */
    hr {
        border: none;
        border-top: 1px solid #e2e6eb;
        margin: 22px 0;
    }

    /* ---- Expander ---- */
    details {
        border: 1px solid #dde1e6 !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ============================================================================
# UI COMPONENTS
# ============================================================================
def render_step_indicator(current_step: int):
    steps = [
        (1, "Listening"),
        (2, "Reading"),
        (3, "Output"),
        (4, "Review"),
    ]
    st.markdown('<div class="step-row">', unsafe_allow_html=True)
    for num, label in steps:
        if num < current_step:
            cls = "done"
        elif num == current_step:
            cls = "active"
        else:
            cls = "pending"
        st.markdown(
            f"""<div class="step-pill {cls}">
                <div class="step-num">Step {num}</div>
                <div>{label}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)
    progress = (current_step - 1) / 3.0
    st.progress(min(progress, 1.0))


def render_vocab_result(entry: dict, index: int):
    """Render a single vocabulary lookup result with save button."""
    expr = entry.get("expression", "")
    definition = entry.get("definition", "")
    derivatives = entry.get("derivatives", "")
    collocations = entry.get("collocations", "")

    st.markdown(f"""
    <div class="vocab-entry">
        <div class="expr">{expr}</div>
        <div class="meta">
            <strong>Definition:</strong> {definition}<br>
            <strong>Derivatives (派生词):</strong> {derivatives or '—'}<br>
            <strong>Collocations (常见搭配):</strong> {collocations or '—'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    cat_options = ["Uncategorized"] + _get_category_options()
    col1, col2, col3 = st.columns([2, 1.5, 1])
    with col1:
        cat = st.selectbox("Category", cat_options, key=f"vcat_{index}_{expr[:15]}", label_visibility="collapsed")
    with col2:
        if st.button("Save to Bank", key=f"sv_{index}_{expr[:15]}", use_container_width=True):
            ok = save_vocab(
                expression=expr, definition=definition,
                derivatives=derivatives, collocations=collocations,
                source_text=st.session_state.get("essay_text", ""),
                category=cat,
            )
            if ok:
                st.toast(f"'{expr}' saved to Vocabulary Bank.")
                st.session_state.setdefault("saved_expressions", set()).add(expr)
                time.sleep(0.2)
                st.rerun()
            else:
                st.toast(f"'{expr}' is already in your bank.")


def _get_category_options() -> list[str]:
    """Merge built-in categories with user-created ones from DB."""
    builtin = ["General", "Academic", "Business", "Technology", "Environment", "Health", "Social", "Idioms & Phrases"]
    try:
        db_cats = get_vocab_categories()
    except Exception:
        db_cats = []
    seen = set(builtin)
    merged = list(builtin)
    for c in db_cats:
        if c not in seen and c != "Uncategorized":
            merged.append(c)
            seen.add(c)
    return merged


def render_flashcard(word_data: dict, card_key: str):
    """Render a flip-card. Uses session_state keyed by card_key to avoid cross-card bleed."""
    f_key = f"fc_flipped_{card_key}"
    r_key = f"fc_rated_{card_key}"
    if f_key not in st.session_state:
        st.session_state[f_key] = False
    if r_key not in st.session_state:
        st.session_state[r_key] = False

    expr = word_data["expression"]
    definition = word_data.get("definition", "")
    derivatives = word_data.get("derivatives", "")
    collocations = word_data.get("collocations", "")
    example = word_data.get("example", "")

    if not st.session_state[f_key]:
        st.markdown(f"""
        <div class="flashcard-front">
            <div style="color:#94a3b8; font-size:0.82rem; margin-bottom:8px;">Tap to reveal</div>
            <div class="word">{expr}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Flip Card", key=f"flipbtn_{card_key}", use_container_width=True):
            st.session_state[f_key] = True
            st.rerun()
    else:
        st.markdown(f"""
        <div class="flashcard-back">
            <div class="word">{expr}</div>
            <div class="def"><strong>Definition:</strong> {definition}</div>
            <div class="deriv"><strong>Derivatives:</strong> {derivatives or '—'}</div>
            <div class="colloc"><strong>Collocations:</strong> {collocations or '—'}</div>
            {f'<div class="ex">{example}</div>' if example else ''}
        </div>
        """, unsafe_allow_html=True)

        if not st.session_state[r_key]:
            st.markdown("**How well did you remember?**")
            cols = st.columns(5)
            labels = ["Forgot", "Vague", "Almost", "Good", "Perfect"]
            for i, (col, lbl) in enumerate(zip(cols, labels)):
                with col:
                    if st.button(lbl, key=f"ratebtn_{card_key}_{i}", use_container_width=True):
                        update_vocab_review(word_data["id"], i + 1)
                        st.session_state[r_key] = True
                        st.toast("Rating recorded.", icon="✅")
                        st.rerun()
        else:
            st.success("Rating recorded.")


# ============================================================================
# SESSION STATE
# ============================================================================
def init_session_state():
    defaults: dict[str, Any] = {
        "page": "Daily Session",
        # Step tracking
        "step": 1,
        # Essay config
        "essay_source": "ai",  # "ai" or "static"
        "essay_topic": "",
        "essay_difficulty": "Intermediate (B2)",
        "essay_word_count": 250,
        "static_article_id": None,
        # Generated content (LOCKED once set)
        "essay_text": "",
        # Step 1
        "user_summary": "",
        "summary_score": "",
        "summary_feedback": "",
        "step1_complete": False,
        # Step 2 — manual vocab lookup
        "lookup_words_input": "",
        "lookup_results": [],
        "saved_expressions": set(),
        # Step 3
        "ai_question": "",
        "user_output": "",
        "ai_corrections": "",
        "ai_polish": "",
        "ai_model_answer": "",
        "ai_comparison": "",
        "feedback_generated": False,
        # Step 4
        "review_words": [],
        "review_index": 0,
        "session_saved": False,
        # Voice Diary
        "voice_transcript": "",
        "voice_evaluation": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset_daily_session():
    """Clear only daily-session keys for a fresh start."""
    keys = [
        "step", "essay_source", "essay_topic", "essay_difficulty", "essay_word_count",
        "static_article_id", "essay_text",
        "user_summary", "summary_score", "summary_feedback", "step1_complete",
        "lookup_words_input", "lookup_results", "saved_expressions",
        "ai_question", "user_output", "ai_corrections", "ai_polish",
        "ai_model_answer", "ai_comparison", "feedback_generated",
        "review_words", "review_index", "session_saved",
    ]
    for k in keys:
        st.session_state.pop(k, None)


# ============================================================================
# PAGE: DAILY SESSION
# ============================================================================
def page_daily_session():
    st.title("Single-Context English Trainer")
    st.caption("One topic. Four steps. Deep learning every day.")

    render_step_indicator(st.session_state.get("step", 1))

    step = st.session_state["step"]
    if step == 1:
        step1_listening()
    elif step == 2:
        step2_reading()
    elif step == 3:
        step3_output()
    elif step == 4:
        step4_review()


# ---------------------------------------------------------------------------
# STEP 1 — Listening (with summary evaluation)
# ---------------------------------------------------------------------------
def step1_listening():
    st.markdown("## Step 1: Listening")

    # ------ PHASE A: Essay not yet generated ------
    if not st.session_state.get("essay_text"):
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Configure Today's Session")

            source = st.radio(
                "Content source",
                options=["AI-generated essay", "Static article library"],
                horizontal=True,
                key="source_radio",
            )
            st.session_state["essay_source"] = "ai" if "AI" in source else "static"

            if st.session_state["essay_source"] == "ai":
                col1, col2, col3 = st.columns(3)
                with col1:
                    topic = st.selectbox("Topic", options=ESSAY_TOPICS)
                with col2:
                    difficulty = st.selectbox("Difficulty", options=list(DIFFICULTY_LABELS.keys()))
                with col3:
                    wc_label = st.selectbox("Length", options=list(WORD_COUNT_OPTIONS.keys()))
                word_count = WORD_COUNT_OPTIONS[wc_label]
                st.session_state["essay_topic"] = topic
                st.session_state["essay_difficulty"] = difficulty
                st.session_state["essay_word_count"] = word_count

                if st.button("Generate Essay & Audio", use_container_width=True, type="primary"):
                    _generate_essay_and_audio(topic, difficulty, word_count)
            else:
                articles = get_all_static_articles()
                if not articles:
                    st.warning("No static articles available. Please add some or switch to AI generation.")
                else:
                    article_options = {f"{a['title']} ({a['difficulty']}, ~{a['word_count']}w)": a["id"] for a in articles}
                    selected = st.selectbox("Choose an article", options=list(article_options.keys()))
                    if st.button("Load Article", use_container_width=True, type="primary"):
                        aid = article_options[selected]
                        article = get_static_article_by_id(aid)
                        if article:
                            st.session_state["essay_text"] = article["content_text"]
                            st.session_state["essay_topic"] = article.get("topic", "")
                            st.session_state["essay_difficulty"] = article.get("difficulty", "Intermediate (B2)")
                            st.session_state["essay_word_count"] = article.get("word_count", 250)
                            st.session_state["essay_source"] = "static"
                            st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
        return

    # ------ PHASE B: Essay is ready, show audio player ------
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Listen Carefully")
        st.caption("The full text is hidden. Focus on listening comprehension. You may pause, replay, and scrub the audio.")

        if st.session_state.get("essay_text"):
            safe_text = st.session_state.essay_text.replace("'", "\\'").replace('"', '\\"').replace('\n', ' ')

            custom_player_html = f"""
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #e9ecef; margin: 10px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;">
                <div style="display: flex; align-items: center; gap: 15px;">
                    <button id="play-pause-btn" onclick="togglePlay()" style="background-color: #4f709c; color: white; border: none; padding: 8px 18px; border-radius: 20px; cursor: pointer; font-weight: bold; width: 80px; transition: background 0.2s;">Play</button>

                    <span id="current-time" style="font-family: monospace; color: #495057; font-size: 13px;">00:00</span>

                    <input type="range" id="progress-bar" min="0" max="100" value="0" oninput="onSliderDrag(this.value)" onchange="onSliderRelease(this.value)" style="flex-grow: 1; accent-color: #4f709c; cursor: pointer; height: 5px; border-radius: 5px; appearance: none; background: #dee2e6;">

                    <span id="total-time" style="font-family: monospace; color: #6c757d; font-size: 13px;">00:00</span>
                </div>

                <script>
                    var synth = window.speechSynthesis;
                    var fullText = "{safe_text}";
                    var utterance = null;
                    var isPlaying = false;
                    var isPaused = false;

                    var wordCount = fullText.split(/\s+/).length;
                    var totalSeconds = Math.ceil((wordCount / 140) * 60);
                    var elapsedSeconds = 0;
                    var timerInterval = null;
                    var scrubTimeout = null;

                    document.getElementById('total-time').innerText = formatTime(totalSeconds);

                    function formatTime(secs) {{
                        var m = Math.floor(secs / 60).toString().padStart(2, '0');
                        var s = (secs % 60).toString().padStart(2, '0');
                        return m + ':' + s;
                    }}

                    function onSliderDrag(value) {{
                        clearInterval(timerInterval);
                        elapsedSeconds = Math.floor((value / 100) * totalSeconds);
                        document.getElementById('current-time').innerText = formatTime(elapsedSeconds);
                    }}

                    function onSliderRelease(value) {{
                        clearTimeout(scrubTimeout);
                        scrubTimeout = setTimeout(function() {{
                            executeScrub(value);
                        }}, 200);
                    }}

                    function executeScrub(value) {{
                        elapsedSeconds = Math.floor((value / 100) * totalSeconds);
                        if (isPlaying || isPaused) {{
                            synth.cancel();
                            var words = fullText.split(/\s+/);
                            var startIndex = Math.floor((value / 100) * words.length);
                            var remainingText = words.slice(startIndex).join(" ");
                            if (remainingText.trim().length === 0) {{
                                resetPlayer();
                                return;
                            }}
                            utterance = new SpeechSynthesisUtterance(remainingText);
                            utterance.lang = 'en-US';
                            utterance.rate = 0.95;
                            utterance.onend = function() {{ resetPlayer(); }};
                            if (synth.paused) {{
                                synth.resume();
                            }}
                            if (isPlaying) {{
                                setTimeout(function() {{
                                    synth.speak(utterance);
                                    startTimer();
                                }}, 50);
                            }} else {{
                                setTimeout(function() {{
                                    synth.speak(utterance);
                                    synth.pause();
                                }}, 50);
                            }}
                        }}
                    }}

                    function togglePlay() {{
                        var btn = document.getElementById('play-pause-btn');
                        if (!isPlaying) {{
                            if (isPaused) {{
                                synth.resume();
                                isPaused = false;
                                isPlaying = true;
                                btn.innerText = 'Pause';
                                startTimer();
                            }} else {{
                                synth.cancel();
                                utterance = new SpeechSynthesisUtterance(fullText);
                                utterance.lang = 'en-US';
                                utterance.rate = 0.95;
                                utterance.onend = function() {{ resetPlayer(); }};
                                synth.speak(utterance);
                                isPlaying = true;
                                btn.innerText = 'Pause';
                                startTimer();
                            }}
                        }} else {{
                            synth.pause();
                            isPaused = true;
                            isPlaying = false;
                            btn.innerText = 'Play';
                            clearInterval(timerInterval);
                        }}
                    }}

                    function startTimer() {{
                        clearInterval(timerInterval);
                        timerInterval = setInterval(function() {{
                            if (synth.speaking && !synth.paused) {{
                                if (elapsedSeconds < totalSeconds) {{
                                    elapsedSeconds++;
                                    document.getElementById('current-time').innerText = formatTime(elapsedSeconds);
                                    document.getElementById('progress-bar').value = (elapsedSeconds / totalSeconds) * 100;
                                }}
                            }}
                        }}, 1000);
                    }}

                    function resetPlayer() {{
                        isPlaying = false;
                        isPaused = false;
                        elapsedSeconds = 0;
                        clearInterval(timerInterval);
                        clearTimeout(scrubTimeout);
                        document.getElementById('play-pause-btn').innerText = 'Play';
                        document.getElementById('current-time').innerText = '00:00';
                        document.getElementById('progress-bar').value = 0;
                    }}
                </script>
            </div>
            """
            st.components.v1.html(custom_player_html, height=85)
        st.markdown('</div>', unsafe_allow_html=True)

    # ------ PHASE C: Summary input ------
    if not st.session_state.get("step1_complete"):
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Main Ideas Summary")
            st.caption(
                "Write a concise paragraph summarising the core tenets of what you heard. "
                "Your summary will be evaluated on Key Information Coverage and Structural Accuracy."
            )

            user_summary = st.text_area(
                "Your summary",
                value=st.session_state.get("user_summary", ""),
                height=120,
                placeholder=(
                    "Write a short paragraph covering the main argument and key supporting points. "
                    "e.g. The speaker argues that preventive healthcare significantly reduces "
                    "long-term medical expenditure by catching diseases early, though implementation "
                    "faces challenges in funding and public awareness..."
                ),
                label_visibility="collapsed",
            )
            st.session_state["user_summary"] = user_summary

            if st.button("Submit Summary & Unlock Text", use_container_width=True, type="primary"):
                if not user_summary.strip():
                    st.warning("Please write a summary before proceeding.")
                else:
                    with st.spinner("Evaluating your summary against the essay..."):
                        evaluation = evaluate_summary(st.session_state["essay_text"], user_summary)
                    st.session_state["summary_score"] = evaluation.get("score", "N/A")
                    st.session_state["summary_feedback"] = (
                        f"**Key Information Coverage:** {evaluation.get('key_coverage', '—')}\n\n"
                        f"**Structural Accuracy:** {evaluation.get('structural_accuracy', '—')}\n\n"
                        f"**Gaps:** {evaluation.get('gaps', '—')}\n\n"
                        f"**How to improve:** {evaluation.get('improvement', '—')}"
                    )
                    st.session_state["step1_complete"] = True
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Show evaluation results
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### Summary Evaluation — Score: {st.session_state.get('summary_score', 'N/A')}")
            st.markdown(st.session_state.get("summary_feedback", ""))
            st.markdown('</div>', unsafe_allow_html=True)

        if st.button("Continue to Reading", use_container_width=True, type="primary"):
            st.session_state["step"] = 2
            st.rerun()

    # Regenerate
    with st.expander("Start over with a different topic"):
        if st.button("Discard & Regenerate"):
            _reset_daily_session()
            st.rerun()


def _generate_essay_and_audio(topic: str, difficulty: str, word_count: int):
    """Generate essay + audio and lock both into session state atomically."""
    with st.spinner("DeepSeek is writing your essay..."):
        essay = generate_essay(topic, difficulty, word_count)
    if not essay:
        st.error("Failed to generate essay. Check your API key and try again.")
        return
    st.session_state["essay_text"] = essay
    st.session_state["essay_topic"] = topic
    st.session_state["essay_difficulty"] = difficulty
    st.session_state["essay_word_count"] = word_count
    st.rerun()


# ---------------------------------------------------------------------------
# STEP 2 — Reading & Manual Vocab Lookup
# ---------------------------------------------------------------------------
def step2_reading():
    st.markdown("## Step 2: Reading & Vocabulary")

    # -- Full essay --
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Full Essay")
        st.markdown(f'<div class="essay-text">{st.session_state["essay_text"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -- Manual vocab lookup --
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Look Up Words & Expressions")
        st.caption(
            "Copy and paste words or phrases from the essay above that you want to learn. "
            "Separate them with commas or put each on a new line. DeepSeek will provide "
            "definitions, derivatives (派生词), and collocations (常见搭配)."
        )

        with st.form(key="vocab_lookup_form", clear_on_submit=False):
            words_input = st.text_area(
                "Words to look up",
                value=st.session_state.get("lookup_words_input", ""),
                height=80,
                placeholder="e.g. democratised, environmental toll, circular fashion models",
                label_visibility="collapsed",
                key="vocab_lookup_textarea",
            )
            submitted = st.form_submit_button(
                "Look Up Selected Words",
                use_container_width=True,
            )
            if submitted:
                parsed = _parse_word_list(words_input)
                if not parsed:
                    st.warning("Please enter at least one word or phrase.")
                else:
                    st.session_state["lookup_words_input"] = words_input
                    with st.spinner(f"Looking up {len(parsed)} word(s)..."):
                        results = lookup_vocab_batch(parsed, st.session_state["essay_text"])
                    st.session_state["lookup_results"] = results
                    st.rerun()

        # Show results
        results = st.session_state.get("lookup_results", [])
        if results:
            st.markdown("---")
            st.markdown("#### Lookup Results")
            for i, entry in enumerate(results):
                render_vocab_result(entry, i)

        st.markdown('</div>', unsafe_allow_html=True)

    # -- Save essay to bank --
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Save Essay to Bank")
        cat_options = ["Uncategorized"] + _get_category_options()
        col_a, col_b, col_c = st.columns([2, 1.5, 1])
        with col_a:
            essay_title = st.text_input("Essay title (optional)", placeholder="e.g. Fast Fashion Analysis", key="essay_title_input")
        with col_b:
            essay_cat = st.selectbox("Category", cat_options, key="essay_cat_select")
        with col_c:
            if st.button("Save Essay", use_container_width=True):
                title = essay_title.strip() or f"Essay – {st.session_state.get('essay_topic', 'Untitled')[:60]}"
                ok = save_essay_to_bank(
                    title=title,
                    content_text=st.session_state["essay_text"],
                    topic=st.session_state.get("essay_topic", ""),
                    difficulty=st.session_state.get("essay_difficulty", "Intermediate (B2)"),
                    word_count=st.session_state.get("essay_word_count", 250),
                    category=essay_cat,
                    source_type=st.session_state.get("essay_source", "ai"),
                )
                if ok:
                    st.toast("Essay saved to Essay Bank.")
                else:
                    st.toast("This essay is already in your bank.")
        st.markdown('</div>', unsafe_allow_html=True)

    # -- Navigation --
    col_back, col_next = st.columns([1, 2])
    with col_back:
        if st.button("Back to Listening", use_container_width=True):
            st.session_state["step"] = 1
            st.rerun()
    with col_next:
        if st.button("Continue to Output", use_container_width=True, type="primary"):
            st.session_state["step"] = 3
            st.rerun()


def _parse_word_list(raw: str) -> list[str]:
    """Parse comma-separated or newline-separated words into a clean list."""
    parts = re.split(r"[,\n]+", raw)
    return [w.strip() for w in parts if w.strip()]


# ---------------------------------------------------------------------------
# STEP 3 — Output (with persistent essay view & model answer comparison)
# ---------------------------------------------------------------------------
def step3_output():
    st.markdown("## Step 3: Output")

    # -- Persistent essay reference --
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Reference Essay")
        st.markdown(f'<div class="essay-text">{st.session_state["essay_text"]}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -- Generate question --
    if not st.session_state.get("ai_question"):
        with st.spinner("Crafting a discussion question..."):
            q = generate_question(st.session_state["essay_text"])
            st.session_state["ai_question"] = q
        st.rerun()

    # -- Question --
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Discussion Question")
        st.info(st.session_state["ai_question"])
        st.markdown('</div>', unsafe_allow_html=True)

    # -- User response --
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Your Response")
        st.caption("Write approximately 150 words in response to the question above. You may reference the essay.")

        user_output = st.text_area(
            "Your written response",
            value=st.session_state.get("user_output", ""),
            height=160,
            placeholder="In my opinion...",
            label_visibility="collapsed",
        )
        st.session_state["user_output"] = user_output
        st.markdown('</div>', unsafe_allow_html=True)

    # -- Generate feedback --
    if st.button("Submit for AI Feedback", use_container_width=True, type="primary"):
        if not user_output.strip():
            st.warning("Please write a response before submitting.")
        else:
            with st.spinner("DeepSeek is evaluating your response..."):
                fb = generate_feedback_full(
                    user_text=user_output,
                    essay_topic=st.session_state.get("essay_topic", ""),
                    question=st.session_state.get("ai_question", ""),
                    essay_text=st.session_state.get("essay_text", ""),
                )
            st.session_state["ai_corrections"] = fb["corrections"]
            st.session_state["ai_polish"] = fb["polish"]
            st.session_state["ai_model_answer"] = fb["model_answer"]
            st.session_state["ai_comparison"] = fb["comparison"]
            st.session_state["feedback_generated"] = True
            st.rerun()

    # -- Show feedback --
    if st.session_state.get("feedback_generated"):
        st.markdown("---")
        st.markdown("### AI Feedback & Analysis")

        # Row 1: Corrections | Polish
        col_left, col_right = st.columns(2)
        with col_left:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Your Version (with corrections)")
                st.markdown(st.session_state["ai_corrections"] or "*No corrections provided.*")
                st.markdown('</div>', unsafe_allow_html=True)
        with col_right:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Native Speaker Polish")
                st.success(st.session_state["ai_polish"] or "*Polish unavailable.*")
                st.markdown('</div>', unsafe_allow_html=True)

        # Row 2: Model Answer | Comparison
        col_left2, col_right2 = st.columns(2)
        with col_left2:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Model Answer (Band 9)")
                st.info(st.session_state["ai_model_answer"] or "*Model answer unavailable.*")
                st.markdown('</div>', unsafe_allow_html=True)
        with col_right2:
            with st.container():
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Critical Thinking Comparison")
                st.markdown(st.session_state["ai_comparison"] or "*Comparison unavailable.*")
                st.markdown('</div>', unsafe_allow_html=True)

    # -- Navigation --
    st.markdown("---")
    col_back2, col_next2 = st.columns([1, 2])
    with col_back2:
        if st.button("Back to Reading", use_container_width=True):
            st.session_state["step"] = 2
            st.rerun()
    with col_next2:
        if st.button("Continue to Review", use_container_width=True, type="primary"):
            st.session_state["step"] = 4
            st.rerun()


# ---------------------------------------------------------------------------
# STEP 4 — Spaced Repetition Review
# ---------------------------------------------------------------------------
def step4_review():
    st.markdown("## Step 4: Spaced Repetition Review")

    # Fetch review words (once per session)
    if not st.session_state.get("review_words"):
        due_words = get_vocab_due_for_review(limit=5)
        st.session_state["review_words"] = due_words
        st.session_state["review_index"] = 0

    review_words: list[dict] = st.session_state["review_words"]
    current_idx: int = st.session_state.get("review_index", 0)

    if not review_words:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### No Words Due for Review")
            st.markdown(
                "Your vocabulary bank is either empty or all words are on schedule. "
                "Save expressions from your daily reading to build your review queue."
            )
            st.markdown('</div>', unsafe_allow_html=True)
        _finalize_session()
        return

    # Progress within review
    if current_idx < len(review_words):
        st.progress(current_idx / len(review_words))
        st.caption(f"Card {current_idx + 1} of {len(review_words)}")

        word_data = review_words[current_idx]
        card_key = f"step4_{word_data['id']}"
        render_flashcard(word_data, card_key)

        # After rating, show next button
        r_key = f"fc_rated_{card_key}"
        if st.session_state.get(r_key):
            if st.button("Next Card", key=f"nextcard_{card_key}", use_container_width=True, type="primary"):
                st.session_state["review_index"] = current_idx + 1
                if st.session_state["review_index"] >= len(review_words):
                    _finalize_session()
                st.rerun()
    else:
        _finalize_session()


def _finalize_session():
    if not st.session_state.get("session_saved"):
        save_session_to_db()
        st.session_state["session_saved"] = True

    st.markdown("---")
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("## Session Complete")
        st.markdown(f"**Topic:** {st.session_state.get('essay_topic', '—')}")
        st.markdown(f"**Summary score:** {st.session_state.get('summary_score', '—')}")
        st.markdown(f"**Difficulty:** {st.session_state.get('essay_difficulty', '—')}")
        st.markdown('</div>', unsafe_allow_html=True)

    stats = get_vocab_stats()
    st.markdown(
        f"Vocabulary Bank: {stats['total']} words | "
        f"{stats['mastered']} mastered | {stats['due']} due for review"
    )

    if st.button("Start a New Session", use_container_width=True, type="primary"):
        _reset_daily_session()
        st.rerun()


# ============================================================================
# PAGE: VOICE DIARY
# ============================================================================
def page_voice_diary():
    st.title("Voice Diary")
    st.caption("Decoupled speaking practice. Record or upload your spoken English and receive AI evaluation on fluency, grammar, and native expression.")

    # -- Input method --
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Record or Upload")

        tab1, tab2 = st.tabs(["Upload Audio File", "Type Transcript Directly"])

        with tab1:
            uploaded = st.file_uploader(
                "Upload a spoken entry (MP3, WAV, M4A, WEBM)",
                type=["mp3", "wav", "m4a", "webm", "ogg"],
                label_visibility="collapsed",
            )
            if uploaded:
                st.audio(uploaded)
                st.info(
                    "Automatic speech-to-text transcription requires a separate STT service "
                    "(e.g., OpenAI Whisper). For now, please manually transcribe your speech below, "
                    "or switch to the 'Type Transcript Directly' tab."
                )

        with tab2:
            st.caption("Type what you said (or would say) — the AI will evaluate it as spoken English.")

        transcript = st.text_area(
            "Your spoken transcript",
            value=st.session_state.get("voice_transcript", ""),
            height=140,
            placeholder="Type your spoken entry here...",
            label_visibility="collapsed",
        )
        st.session_state["voice_transcript"] = transcript

        topic_hint = st.text_input(
            "Topic (optional)",
            placeholder="e.g. My thoughts on remote work",
        )

        if st.button("Evaluate My Speaking", use_container_width=True, type="primary"):
            if not transcript.strip():
                st.warning("Please provide a transcript to evaluate.")
            else:
                with st.spinner("Evaluating your spoken English..."):
                    evaluation = evaluate_voice_diary(transcript, topic_hint)
                st.session_state["voice_evaluation"] = evaluation
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    # -- Results --
    evaluation = st.session_state.get("voice_evaluation")
    if evaluation:
        st.markdown("---")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### Speaking Evaluation — Score: {evaluation.get('overall_score', 'N/A')}")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### Fluency & Delivery")
                st.markdown(evaluation.get("fluency", "—"))
                st.markdown("#### Grammatical Accuracy")
                st.markdown(evaluation.get("grammar_feedback", "—"))
            with col_b:
                st.markdown("#### Native Expression Level")
                st.markdown(evaluation.get("expression_level", "—"))
                st.markdown("#### Improvement Tips")
                st.markdown(evaluation.get("improvement_tips", "—"))
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# PAGE: VOCABULARY BANK
# ============================================================================
def page_vocab_bank():
    st.title("Vocabulary Bank")

    stats = get_vocab_stats()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Words", stats["total"])
    with col2:
        st.metric("Due for Review", stats["due"])
    with col3:
        st.metric("Mastered (30d+)", stats["mastered"])
    with col4:
        rate = f"{int(stats['mastered'] / stats['total'] * 100)}%" if stats["total"] > 0 else "—"
        st.metric("Retention Rate", rate)

    st.markdown("---")

    # Filters
    col_s, col_c = st.columns([2, 1])
    with col_s:
        search = st.text_input("Search expressions or definitions", placeholder="e.g. 'democratised'")
    with col_c:
        cat_options = ["All"] + _get_category_options()
        cat_filter = st.selectbox("Category", cat_options)

    vocab_list = get_all_vocab(search=search, category=cat_filter if cat_filter != "All" else "")

    if not vocab_list:
        st.info("Your vocabulary bank is empty. Look up words during your daily reading sessions to build it.")
        return

    st.caption(f"Showing {len(vocab_list)} word(s)")

    for item in vocab_list:
        with st.container():
            st.markdown('<div class="vocab-entry">', unsafe_allow_html=True)
            st.markdown(f'<div class="expr">{item["expression"]}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="meta">
                <strong>Definition:</strong> {item['definition']}<br>
                <strong>Derivatives:</strong> {item.get('derivatives') or '—'}<br>
                <strong>Collocations:</strong> {item.get('collocations') or '—'}<br>
                <strong>Category:</strong> {item.get('category', 'Uncategorized')} &nbsp;|&nbsp;
                <strong>Interval:</strong> {item['interval']}d &nbsp;|&nbsp;
                <strong>Reviews:</strong> {item['review_count']} &nbsp;|&nbsp;
                <strong>Next review:</strong> {item.get('next_review_date', '—')}
            </div>
            """, unsafe_allow_html=True)

            col_act1, col_act2, col_act3 = st.columns([1.5, 1, 1])
            with col_act1:
                new_cat = st.selectbox(
                    "Change category",
                    ["Uncategorized"] + _get_category_options(),
                    key=f"vocab_cat_{item['id']}",
                    label_visibility="collapsed",
                )
                if new_cat != item.get("category", "Uncategorized"):
                    update_vocab_category(item["id"], new_cat)
                    st.rerun()
            with col_act2:
                if st.button("Review Now", key=f"rvnow_{item['id']}", use_container_width=True):
                    st.session_state["quick_review_word"] = item
                    st.rerun()
            with col_act3:
                if st.button("Delete", key=f"delv_{item['id']}", use_container_width=True):
                    delete_vocab(item["id"])
                    st.toast(f"Deleted '{item['expression']}'.")
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Quick review modal
    quick_word = st.session_state.get("quick_review_word")
    if quick_word:
        st.markdown("---")
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Quick Review")
            render_flashcard(quick_word, f"quick_{quick_word['id']}")
            if st.button("Close Quick Review", use_container_width=True):
                st.session_state.pop("quick_review_word", None)
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

    # Export
    if vocab_list:
        st.markdown("---")
        csv_data = "Expression,Definition,Derivatives,Collocations,Category,Date Added,Interval,Next Review\n"
        for v in vocab_list:
            csv_data += (
                f'"{v["expression"]}","{v["definition"]}","{v.get("derivatives", "")}",'
                f'"{v.get("collocations", "")}","{v.get("category", "")}",'
                f'{v["date_added"]},{v["interval"]},{v.get("next_review_date", "")}\n'
            )
        st.download_button(
            "Export as CSV",
            data=csv_data,
            file_name=f"vocab_bank_{date.today().isoformat()}.csv",
            mime="text/csv",
        )


# ============================================================================
# PAGE: ESSAY BANK
# ============================================================================
def page_essay_bank():
    st.title("Essay Bank")
    st.caption("Saved essays from your daily sessions. Revisit, re-read, or re-use for practice.")

    col_s, col_c = st.columns([2, 1])
    with col_s:
        search = st.text_input("Search essays by title or topic", placeholder="e.g. 'fashion'")
    with col_c:
        cat_opts = ["All"] + (get_essay_categories() or [])
        cat_filter = st.selectbox("Category", cat_opts, key="essay_cat_filter")

    essays = get_all_essays(search=search, category=cat_filter if cat_filter != "All" else "")

    if not essays:
        st.info("Your essay bank is empty. Save essays from your daily reading sessions.")
        return

    st.caption(f"Showing {len(essays)} essay(s)")

    for essay in essays:
        with st.container():
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"### {essay.get('title', 'Untitled')}")
            st.caption(
                f"Topic: {essay.get('topic', '—')} | "
                f"Difficulty: {essay.get('difficulty', '—')} | "
                f"Words: {essay.get('word_count', '—')} | "
                f"Source: {essay.get('source_type', '—')} | "
                f"Category: {essay.get('category', 'Uncategorized')} | "
                f"Saved: {essay.get('date_added', '—')}"
            )
            with st.expander("Read full essay"):
                st.markdown(f'<div class="essay-text">{essay["content_text"]}</div>', unsafe_allow_html=True)

            col_cat, col_use, col_del = st.columns([1.5, 1, 1])
            with col_cat:
                new_cat = st.selectbox(
                    "Category",
                    ["Uncategorized"] + _get_category_options(),
                    key=f"essay_cat_{essay['id']}",
                    label_visibility="collapsed",
                )
                if new_cat != essay.get("category", "Uncategorized"):
                    update_essay_category(essay["id"], new_cat)
                    st.rerun()
            with col_use:
                if st.button("Use in Session", key=f"use_essay_{essay['id']}", use_container_width=True):
                    _reset_daily_session()
                    st.session_state["essay_text"] = essay["content_text"]
                    st.session_state["essay_topic"] = essay.get("topic", "")
                    st.session_state["essay_difficulty"] = essay.get("difficulty", "Intermediate (B2)")
                    st.session_state["essay_word_count"] = essay.get("word_count", 250)
                    st.session_state["essay_source"] = "bank"
                    st.session_state["page"] = "Daily Session"
                    st.toast("Essay loaded into Daily Session.")
                    st.rerun()
            with col_del:
                if st.button("Delete", key=f"del_essay_{essay['id']}", use_container_width=True):
                    delete_essay(essay["id"])
                    st.toast("Essay deleted.")
                    st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# MAIN APP
# ============================================================================
def main():
    st.set_page_config(
        page_title="English Trainer",
        page_icon="✍",  # writing hand (monochrome-ish)
        layout="wide",
        initial_sidebar_state="expanded",
    )

    init_db()
    seed_static_articles()
    init_session_state()
    inject_css()

    # ── Sidebar ────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding:12px 0 6px 0;">
            <div style="font-size:1.1rem; font-weight:700; color:#1e3a5f;">
                English Trainer
            </div>
            <div style="font-size:0.75rem; color:#7a8a9a; margin-top:2px;">
                SLA-informed &bull; Single Context
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        nav_options = [
            "Daily Session",
            "Voice Diary",
            "Vocabulary Bank",
            "Essay Bank",
        ]
        current_page = st.session_state.get("page", "Daily Session")
        default_idx = nav_options.index(current_page) if current_page in nav_options else 0

        page = st.radio(
            "Navigation",
            options=nav_options,
            index=default_idx,
            label_visibility="collapsed",
        )

        if page != st.session_state.get("page"):
            st.session_state["page"] = page
            st.rerun()

        st.markdown("---")

        # Quick stats
        stats = get_vocab_stats()
        st.markdown(f"""
        <div style="font-size:0.82rem; color:#5a6a7a; line-height:1.8;">
            Vocabulary: <strong style="color:#1e3a5f;">{stats['total']}</strong><br>
            Due today: <strong style="color:#c47f30;">{stats['due']}</strong><br>
            Mastered: <strong style="color:#5a8a4a;">{stats['mastered']}</strong>
        </div>
        """, unsafe_allow_html=True)

    # ── Main content ───────────────────────────────────────────────────
    current_page = st.session_state.get("page", "Daily Session")

    if current_page == "Daily Session":
        page_daily_session()
    elif current_page == "Voice Diary":
        page_voice_diary()
    elif current_page == "Vocabulary Bank":
        page_vocab_bank()
    elif current_page == "Essay Bank":
        page_essay_bank()


if __name__ == "__main__":
    main()
