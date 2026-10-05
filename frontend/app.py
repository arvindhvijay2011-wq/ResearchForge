"""Streamlit frontend for ResearchForge.

The frontend executes the LangGraph workflow exactly once per research request,
streams node updates into the interface, and renders the final report returned
by that same execution.
"""

from __future__ import annotations

import html
import os
import sys
from pathlib import Path
from typing import Any


# Streamlit executes this file as `frontend/app.py`, which places the
# `frontend` directory ahead of the project root on sys.path. Because the
# backend package is also named `app`, Python can otherwise resolve this
# frontend module as `app` and make `app.graph` appear to be missing.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from app.graph.workflow import build_graph


load_dotenv()


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-mini")

AGENT_ORDER: list[tuple[str, str, str]] = [
    ("planner", "Planner", "Research decomposition"),
    ("scout", "Research Scouts", "Independent web investigation"),
    ("analyst", "Analyst", "Cross-source analysis"),
    ("critic", "Critic", "Evidence audit"),
    ("synthesizer", "Synthesizer", "Final synthesis"),
]


def create_llm() -> ChatOpenAI:
    """Create the configured OpenAI chat model."""

    return ChatOpenAI(model=MODEL_NAME)


def inject_styles(font_scale: float) -> None:
    """Inject the ResearchForge visual system into Streamlit."""

    css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html { font-size: calc(16px * __FONT_SCALE__); }

:root {
    --rf-bg: #07090f;
    --rf-panel: rgba(17, 21, 31, 0.76);
    --rf-panel-strong: rgba(21, 26, 38, 0.92);
    --rf-border: rgba(255,255,255,0.085);
    --rf-border-bright: rgba(255,255,255,0.15);
    --rf-text: #f4f7fb;
    --rf-muted: #8b96a8;
    --rf-dim: #5f6b7d;
    --rf-blue: #64a8ff;
    --rf-cyan: #5be7ff;
    --rf-violet: #9b7cff;
    --rf-green: #48d597;
    --rf-amber: #ffc85c;
    --rf-red: #ff7272;
    --rf-font-scale: __FONT_SCALE__;
}

html, body, [class*="css"] {
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    font-size: 1rem !important;
}



html,
body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] *,
input,
textarea,
button,
select,
option {
    font-family:
        "Plus Jakarta Sans",
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif !important;
}


/* Restore Streamlit's Material Symbols font for its internal icons.
   The global Plus Jakarta Sans rule must not replace the icon font,
   otherwise ligature names such as "keyboard_arrow_down" appear as text. */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
.material-symbols-outlined,
span[class*="material-symbols"],
i[class*="material-symbols"] {
    font-family:
        "Material Symbols Rounded",
        "Material Symbols Outlined" !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-feature-settings:
        "liga" 1,
        "calt" 1 !important;
    -webkit-font-feature-settings:
        "liga" 1,
        "calt" 1 !important;
}

html, body, [class*="css"], input, textarea, button, select, option {
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
}

body, [data-testid="stAppViewContainer"], [data-testid="stAppViewBlockContainer"] {
    font-size: calc(1rem * __FONT_SCALE__) !important;
}


/* Page-wide font-size scaling. */
[data-testid="stAppViewContainer"] {
    zoom: var(--rf-font-scale);
}

[data-testid="stAppViewContainer"] > section {
    min-width: calc(100% / var(--rf-font-scale));
}

.stApp {
    background:
        radial-gradient(circle at 12% 0%, rgba(85, 108, 255, 0.12), transparent 28%),
        radial-gradient(circle at 90% 14%, rgba(46, 220, 255, 0.075), transparent 25%),
        radial-gradient(circle at 50% 100%, rgba(117, 78, 255, 0.06), transparent 34%),
        var(--rf-bg);
    color: var(--rf-text);
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    visibility: hidden;
    height: 0;
}

[data-testid="stDecoration"] {
    display: none;
}

.block-container {
    max-width: 1180px;
    padding-top: 1.6rem;
    padding-bottom: 4rem;
}

div[data-testid="stVerticalBlock"] {
    gap: 0.65rem;
}

.rf-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 2.2rem;
}

.rf-brand {
    display: flex;
    align-items: center;
    gap: 0.85rem;
}

.rf-logo {
    width: 46px;
    height: 46px;
    display: grid;
    place-items: center;
    border: 1px solid rgba(112, 174, 255, 0.22);
    border-radius: 14px;
    background:
        radial-gradient(circle at 35% 30%, rgba(91,231,255,.14), transparent 48%),
        rgba(19, 24, 36, .82);
    box-shadow:
        0 10px 35px rgba(0,0,0,.28),
        inset 0 1px rgba(255,255,255,.08),
        0 0 28px rgba(91,231,255,.045);
}

.rf-brand-name {
    color: #f7f9fc;
    font-size: calc(0.88rem * var(--rf-font-scale));
    font-weight: 800;
    letter-spacing: 0.19em;
}

.rf-brand-sub {
    color: #59677b;
    font-size: calc(0.67rem * var(--rf-font-scale));
    margin-top: 0.18rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.rf-mode {
    color: #607087;
    font-size: calc(0.68rem * var(--rf-font-scale));
    letter-spacing: 0.16em;
    text-transform: uppercase;
}

.rf-size-control {
    text-align: left;
}

.rf-size-label {
    color: #5f6e82;
    font-size: calc(.59rem * var(--rf-font-scale));
    font-weight: 800;
    letter-spacing: .13em;
    text-transform: uppercase;
    margin: 0 0 .25rem .35rem;
}

[data-testid="stSelectbox"] > div > div {
    min-height: 38px;
    border-radius: 11px !important;
    border: 1px solid var(--rf-border-bright) !important;
    background: rgba(255,255,255,.035) !important;
}

[data-testid="stSelectbox"] div,
[data-testid="stSelectbox"] input {
    font-size: calc(.74rem * var(--rf-font-scale)) !important;
}


/* Search field cleanup: absolutely no decorative right-side circle. */
[data-testid="stForm"] [data-testid="stTextArea"]::before,
[data-testid="stForm"] [data-testid="stTextArea"]::after,
[data-testid="stForm"] [data-testid="stTextArea"] textarea::before,
[data-testid="stForm"] [data-testid="stTextArea"] textarea::after {
    content: none !important;
    display: none !important;
}

/* Preserve only the intended left sparkle as a background, not a pseudo-element. */
[data-testid="stForm"] [data-testid="stTextArea"] {
    background:
        radial-gradient(
            circle at 23px 50%,
            rgba(228, 239, 255, .95) 0 2px,
            transparent 2.7px
        ),
        linear-gradient(
            180deg,
            rgba(120, 144, 180, .15),
            rgba(8, 13, 22, .66)
        ),
        rgba(8, 12, 19, .72) !important;
}


[data-testid="stForm"],
[data-testid="stForm"] > div,
[data-testid="stForm"] > div > div,
[data-testid="stForm"] fieldset {
    border: 0 !important;
    outline: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Search composer: one clean glass capsule, no moving outline. */
[data-testid="stForm"] {
    position: relative !important;
    isolation: isolate;
    padding: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

.rf-composer-shell {
    position: relative;
    padding: 0 !important;
    margin: 0 !important;
    border: 0 !important;
    background: transparent !important;
    box-shadow: none !important;
}

[data-testid="stForm"] [data-testid="stTextArea"] {
    position: relative !important;
    margin: 0 !important;
    border-radius: 28px !important;
    padding: 7px !important;
    background:
        linear-gradient(180deg, rgba(120, 144, 180, .15), rgba(8, 13, 22, .66)),
        rgba(8, 12, 19, .72) !important;
    border: 1px solid rgba(207, 223, 250, .24) !important;
    box-shadow:
        0 22px 65px rgba(0,0,0,.28),
        inset 0 1px 0 rgba(255,255,255,.13),
        0 0 30px rgba(93, 138, 255, .045) !important;
}

[data-testid="stForm"] [data-testid="stTextArea"]::before {
    content: "✦";
    position: absolute;
    left: 19px;
    top: 50%;
    z-index: 9;
    transform: translateY(-50%);
    color: #e4efff;
    font-size: 1.05rem;
    text-shadow: 0 0 12px rgba(114, 188, 255, .72);
    pointer-events: none;
}


[data-testid="stForm"] [data-testid="stTextArea"] textarea {
    min-height: 92px !important;
    margin: 0 !important;
    padding: 18px 22px 18px 50px !important;
    border-radius: 22px !important;
    border: 1px solid rgba(229, 239, 255, .11) !important;
    background:
        linear-gradient(180deg, rgba(91, 112, 145, .25), rgba(15, 22, 33, .89)) !important;
    color: #f5f8ff !important;
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    font-size: 1rem !important;
    line-height: 1.55 !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.105),
        inset 0 -14px 30px rgba(0,0,0,.08) !important;
}

[data-testid="stForm"] [data-testid="stTextArea"],
[data-testid="stForm"] [data-testid="stTextArea"] *,
[data-testid="stForm"] [data-testid="stTextArea"] textarea:focus,
[data-testid="stForm"] [data-testid="stTextArea"] textarea:focus-visible {
    outline: 0 !important;
    outline-color: transparent !important;
}

[data-testid="stForm"] [data-testid="stTextArea"] textarea:focus,
[data-testid="stForm"] [data-testid="stTextArea"] textarea:focus-visible {
    border-color: rgba(229, 239, 255, .11) !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.105),
        0 0 0 1px rgba(128, 177, 255, .035),
        0 0 24px rgba(77, 127, 255, .045) !important;
}


/* Streamlit/BaseWeb focus cleanup. */
[data-testid="stForm"] [data-testid="stTextArea"],
[data-testid="stForm"] [data-testid="stTextArea"] > div,
[data-testid="stForm"] [data-testid="stTextArea"] > div > div,
[data-testid="stForm"] [data-testid="stTextArea"] [data-baseweb="base-input"],
[data-testid="stForm"] [data-testid="stTextArea"] [data-baseweb="textarea"],
[data-testid="stForm"] [data-testid="stTextArea"] textarea,
[data-testid="stForm"] [data-testid="stTextArea"] [aria-invalid="true"] {
    outline: none !important;
    outline-offset: 0 !important;
}

[data-testid="stForm"] [data-testid="stTextArea"]:focus-within,
[data-testid="stForm"] [data-testid="stTextArea"] > div:focus-within,
[data-testid="stForm"] [data-testid="stTextArea"] [data-baseweb="base-input"]:focus-within,
[data-testid="stForm"] [data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within,
[data-testid="stForm"] [data-testid="stTextArea"] [aria-invalid="true"] {
    border-color: rgba(207, 223, 250, .24) !important;
    outline: none !important;
    box-shadow:
        0 22px 65px rgba(0,0,0,.28),
        inset 0 1px 0 rgba(255,255,255,.13),
        0 0 30px rgba(93, 138, 255, .045) !important;
}

[data-testid="stForm"] [data-testid="stTextArea"] textarea:focus,
[data-testid="stForm"] [data-testid="stTextArea"] textarea:focus-visible {
    outline: none !important;
    border-color: rgba(229, 239, 255, .11) !important;
    box-shadow:
        inset 0 1px 0 rgba(255,255,255,.105),
        inset 0 -14px 30px rgba(0,0,0,.08) !important;
}

[data-testid="stForm"] [data-testid="stTextArea"] textarea::placeholder {
    color: rgba(214, 226, 245, .68) !important;
}

[data-testid="stForm"] [data-testid="stFormSubmitButton"] button,
[data-testid="stForm"] button[kind="primary"] {
    width: 100% !important;
    margin-top: .72rem !important;
    min-height: 46px !important;
    border-radius: 14px !important;
    border: 1px solid rgba(140, 176, 255, .22) !important;
    background: linear-gradient(135deg, #6f93ff, #7968ff) !important;
    color: #fff !important;
    font-family: "Plus Jakarta Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    font-size: calc(.82rem * var(--rf-font-scale)) !important;
    font-weight: 700 !important;
    box-shadow: 0 10px 30px rgba(89, 104, 255, .18) !important;
}

[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover,
[data-testid="stForm"] button[kind="primary"]:hover {
    filter: brightness(1.06);
    transform: translateY(-1px);
}

.rf-hero {
    position: relative;
    overflow: hidden;
    padding: 3.7rem 3.5rem 3.2rem;
    border: 1px solid var(--rf-border);
    border-radius: 30px;
    background:
        linear-gradient(145deg, rgba(22, 27, 40, .83), rgba(9, 12, 20, .78));
    box-shadow:
        0 35px 100px rgba(0,0,0,.34),
        inset 0 1px rgba(255,255,255,.065);
}

.rf-hero::before {
    content: "";
    position: absolute;
    width: 430px;
    height: 430px;
    right: -170px;
    top: -240px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(91,231,255,.13), transparent 67%);
    pointer-events: none;
}

.rf-hero::after {
    content: "";
    position: absolute;
    width: 320px;
    height: 320px;
    left: -210px;
    bottom: -250px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(155,124,255,.09), transparent 68%);
    pointer-events: none;
}

.rf-eyebrow {
    position: relative;
    color: #75baff;
    font-size: 0.69rem;
    font-weight: 800;
    letter-spacing: .18em;
    text-transform: uppercase;
    margin-bottom: 1rem;
}

.rf-title {
    position: relative;
    max-width: 780px;
    margin: 0;
    color: #f6f8fb;
    font-size: clamp(2.55rem, calc(6vw * var(--rf-font-scale)), calc(5.1rem * var(--rf-font-scale)));
    line-height: .94;
    letter-spacing: -.065em;
    font-weight: 800;
}

.rf-title-accent {
    color: #8eaaff;
}

.rf-description {
    position: relative;
    max-width: 690px;
    margin: 1.35rem 0 0;
    color: #a5b0c0;
    font-size: 0.98rem;
    line-height: 1.75;
}

.rf-composer-shell {
    margin-top: 1.35rem;
    padding: .5rem;
    border: 1px solid var(--rf-border-bright);
    border-radius: 19px;
    background: rgba(8, 12, 19, .68);
    box-shadow:
        0 18px 55px rgba(0,0,0,.24),
        inset 0 1px rgba(255,255,255,.045);
}

.rf-section-label {
    color: #627086;
    font-size: calc(0.67rem * var(--rf-font-scale));
    font-weight: 800;
    letter-spacing: .17em;
    text-transform: uppercase;
    margin: 1.9rem 0 .65rem;
}

.rf-pipeline {
    padding: 1.15rem;
    border: 1px solid var(--rf-border);
    border-radius: 21px;
    background: rgba(12, 16, 25, .68);
}

.rf-agent {
    display: grid;
    grid-template-columns: 42px minmax(0, 1fr) auto;
    align-items: center;
    gap: .85rem;
    min-height: 64px;
    padding: .7rem .8rem;
    border: 1px solid transparent;
    border-radius: 15px;
    transition: .2s ease;
}

.rf-agent + .rf-agent {
    margin-top: .32rem;
}

.rf-agent-active {
    border-color: rgba(91,231,255,.16);
    background: linear-gradient(90deg, rgba(91,231,255,.055), rgba(155,124,255,.025));
}

.rf-agent-done {
    background: rgba(72,213,151,.025);
}

.rf-agent-icon {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    border: 1px solid var(--rf-border);
    border-radius: 12px;
    background: rgba(255,255,255,.035);
    color: #7e8ca1;
    font-size: .78rem;
    font-weight: 800;
}

.rf-agent-active .rf-agent-icon {
    color: var(--rf-cyan);
    border-color: rgba(91,231,255,.22);
    box-shadow: 0 0 22px rgba(91,231,255,.07);
}

.rf-agent-done .rf-agent-icon {
    color: var(--rf-green);
    border-color: rgba(72,213,151,.2);
}

.rf-agent-name {
    color: #dfe6ef;
    font-size: .84rem;
    font-weight: 700;
}

.rf-agent-detail {
    color: #687589;
    font-size: .68rem;
    margin-top: .2rem;
}

.rf-agent-state {
    color: #5f6e82;
    font-size: .64rem;
    letter-spacing: .1em;
    text-transform: uppercase;
}

.rf-agent-active .rf-agent-state {
    color: #6bdff5;
}

.rf-agent-done .rf-agent-state {
    color: #58d99e;
}

.rf-report {
    margin-top: .7rem;
    padding: 2rem 2.1rem;
    border: 1px solid var(--rf-border);
    border-radius: 24px;
    background: linear-gradient(145deg, rgba(18,22,33,.78), rgba(9,12,19,.82));
    box-shadow: 0 25px 75px rgba(0,0,0,.25);
}

.rf-report-title {
    margin: 0;
    color: #f7f9fc;
    font-size: clamp(1.7rem, calc(3vw * var(--rf-font-scale)), calc(2.65rem * var(--rf-font-scale)));
    line-height: 1.08;
    letter-spacing: -.04em;
}

.rf-summary {
    margin-top: 1.1rem;
    padding: 1rem 1.15rem;
    border-left: 2px solid #8d7cff;
    border-radius: 0 13px 13px 0;
    background: rgba(141,124,255,.055);
    color: #bdc6d4;
    line-height: 1.78;
}

.rf-section-title {
    margin: 1.8rem 0 .45rem;
    color: #e9eef5;
    font-size: 1.12rem;
    letter-spacing: -.02em;
}

.rf-section-content {
    color: #abb6c6;
    line-height: 1.8;
    font-size: .91rem;
}

.rf-caveat {
    color: #9ba7b7;
    line-height: 1.65;
    font-size: .83rem;
    margin: .35rem 0;
}

.rf-source {
    padding: .8rem .9rem;
    margin: .4rem 0;
    border: 1px solid var(--rf-border);
    border-radius: 13px;
    background: rgba(255,255,255,.025);
}

.rf-source-title {
    color: #dbe3ed;
    font-size: .79rem;
    font-weight: 700;
}

.rf-source-meta {
    color: #66758a;
    font-size: .65rem;
    margin-top: .25rem;
}

.rf-source-url {
    color: #639ee7;
    font-size: .66rem;
    word-break: break-all;
    margin-top: .25rem;
}

.rf-empty {
    padding: 1.1rem;
    border: 1px dashed var(--rf-border-bright);
    border-radius: 15px;
    color: #69778a;
    text-align: center;
    font-size: .78rem;
}

button[kind="primary"] {
    border: 1px solid rgba(91,231,255,.18) !important;
    background: linear-gradient(135deg, #4d8cff, #695cff) !important;
    box-shadow: 0 8px 28px rgba(77,140,255,.18) !important;
    transition: transform .16s ease, filter .16s ease !important;
}

button[kind="primary"]:hover {
    transform: translateY(-1px);
    filter: brightness(1.08);
}

button[kind="secondary"] {
    border-color: var(--rf-border) !important;
    background: rgba(255,255,255,.025) !important;
}

textarea {
    background: transparent !important;
}

@media (max-width: 760px) {
    .block-container {
        padding: 1rem .75rem 3rem;
    }

    .rf-header {
        margin-bottom: 1.25rem;
    }

    .rf-mode {
        display: none;
    }

    .rf-hero {
        padding: 2.25rem 1.2rem 1.45rem;
        border-radius: 23px;
    }

    .rf-title {
        font-size: clamp(2.35rem, calc(13vw * var(--rf-font-scale)), calc(4rem * var(--rf-font-scale)));
    }

    .rf-description {
        font-size: .87rem;
    }

    .rf-agent {
        grid-template-columns: 38px minmax(0, 1fr);
    }

    .rf-agent-state {
        grid-column: 2;
        margin-top: -.25rem;
    }

    .rf-report {
        padding: 1.25rem;
        border-radius: 19px;
    }
}
</style>
"""
    css = css.replace("__FONT_SCALE__", str(font_scale))
    st.markdown(css, unsafe_allow_html=True)


def render_header() -> float:
    """Render the application header and page-wide font-size control."""

    sizes: dict[str, float] = {
        "Small": 0.90,
        "Default": 1.00,
        "Large": 1.12,
        "XL": 1.24,
    }

    if "rf_font_size" not in st.session_state:
        st.session_state["rf_font_size"] = "Default"

    left, right = st.columns(
        [5.8, 1.55],
        vertical_alignment="center",
    )

    with left:
        st.markdown(
            r'''
<div class="rf-header">
    <div class="rf-brand">
        <div class="rf-logo" aria-label="ResearchForge logo">
            <svg width="31" height="31" viewBox="0 0 72 72" fill="none"
                 xmlns="http://www.w3.org/2000/svg" role="img">
                <path d="M9 17.5C9 14.7 11.2 12.5 14 12.5H47V27.5H14C11.2 27.5 9 25.3 9 22.5V17.5Z"
                      fill="url(#rfHammerMetal)" stroke="#F7FAFF" stroke-width="2.2"/>
                <path d="M15 13V27" stroke="#697991" stroke-width="2" opacity=".72"/>
                <path d="M39 27L56 44" stroke="#8E9DB4" stroke-width="9.5" stroke-linecap="round"/>
                <path d="M39 27L56 44" stroke="#EAF0F8" stroke-width="2.3" stroke-linecap="round" opacity=".72"/>
                <path d="M52 40L61 49" stroke="#65748A" stroke-width="9.5" stroke-linecap="round"/>
                <path d="M50 7L38 28H49L43 40L65 15H53L58 7H50Z"
                      fill="#FFC85C" stroke="#FFE7A8" stroke-width="1.55"
                      stroke-linejoin="round"/>
                <defs>
                    <linearGradient id="rfHammerMetal" x1="11" y1="13" x2="46" y2="28" gradientUnits="userSpaceOnUse">
                        <stop stop-color="#F6F9FD"/>
                        <stop offset=".42" stop-color="#C5D0DF"/>
                        <stop offset="1" stop-color="#8E9DB3"/>
                    </linearGradient>
                </defs>
            </svg>
        </div>
        <div>
            <div class="rf-brand-name">RESEARCHFORGE</div>
            <div class="rf-brand-sub">Multi-agent research intelligence</div>
        </div>
    </div>
</div>
''' ,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="rf-size-control"><div class="rf-size-label">Text size</div>',
            unsafe_allow_html=True,
        )
        selected_size = st.selectbox(
            "Text size",
            options=list(sizes),
            key="rf_font_size",
            label_visibility="collapsed",
        )
        st.markdown('</div>', unsafe_allow_html=True)

    return sizes[selected_size]


def render_hero() -> None:
    """Render the ResearchForge landing hero."""

    st.markdown(
        """
<div class="rf-hero">
    <div class="rf-eyebrow">Multi-Agent Intelligence</div>
    <h1 class="rf-title">
        Research without<br>
        <span class="rf-title-accent">the rabbit hole.</span>
    </h1>
    <p class="rf-description">
        One question becomes a coordinated investigation. ResearchForge
        decomposes the problem, sends independent Scouts into the web,
        cross-checks the evidence, audits the research, and forges the result
        into one clear research brief.
    </p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_pipeline(statuses: dict[str, str]) -> None:
    """Render the live multi-agent pipeline."""

    icons = {
        "planner": "01",
        "scout": "02",
        "analyst": "03",
        "critic": "04",
        "synthesizer": "05",
    }

    rows: list[str] = []

    for key, name, detail in AGENT_ORDER:
        state = statuses.get(key, "waiting")

        if state == "done":
            state_label = "Completed"
            class_name = "rf-agent rf-agent-done"
        elif state == "active":
            state_label = "Working"
            class_name = "rf-agent rf-agent-active"
        else:
            state_label = "Queued"
            class_name = "rf-agent"

        rows.append(
            f"""
<div class="{class_name}">
    <div class="rf-agent-icon">{icons[key]}</div>
    <div>
        <div class="rf-agent-name">{name}</div>
        <div class="rf-agent-detail">{detail}</div>
    </div>
    <div class="rf-agent-state">{state_label}</div>
</div>
"""
        )

    st.markdown(
        '<div class="rf-pipeline">' + "".join(rows) + "</div>",
        unsafe_allow_html=True,
    )


def render_report(report: dict[str, Any]) -> None:
    """Render the final structured research report."""

    if not report:
        return

    st.markdown('<div class="rf-section-label">Research result</div>', unsafe_allow_html=True)

    title = html.escape(str(report.get("title", "Research report")))

    st.markdown(
        f"""
<div class="rf-report">
    <h2 class="rf-report-title">{title}</h2>
</div>
""",
        unsafe_allow_html=True,
    )

    summary = html.escape(str(report.get("executive_summary", "")).strip())

    if summary:
        st.markdown(
            f'<div class="rf-summary">{summary}</div>',
            unsafe_allow_html=True,
        )

    for section in report.get("sections", []):
        heading = html.escape(str(section.get("heading", "Section")))
        content = html.escape(str(section.get("content", "")))

        st.markdown(
            f"""
<div class="rf-section-title">{heading}</div>
<div class="rf-section-content">{content}</div>
""",
            unsafe_allow_html=True,
        )

    caveats = report.get("caveats", [])

    if caveats:
        st.markdown(
            '<div class="rf-section-title">Caveats</div>',
            unsafe_allow_html=True,
        )

        for caveat in caveats:
            st.markdown(
                f'<div class="rf-caveat">• {html.escape(str(caveat))}</div>',
                unsafe_allow_html=True,
            )


def render_sources(sources: list[dict[str, Any]]) -> None:
    """Render deduplicated evidence sources."""

    st.markdown(
        '<div class="rf-section-label">Evidence sources</div>',
        unsafe_allow_html=True,
    )

    if not sources:
        st.markdown(
            '<div class="rf-empty">No source metadata was returned.</div>',
            unsafe_allow_html=True,
        )
        return

    with st.expander(f"{len(sources)} sources retrieved", expanded=False):
        for index, source in enumerate(sources, start=1):
            title = html.escape(str(source.get("title") or source.get("url") or "Untitled source"))
            url = html.escape(str(source.get("url") or ""))
            source_type = html.escape(str(source.get("source_type") or "unknown"))
            score = float(source.get("relevance_score") or 0.0)

            st.markdown(
                f"""
<div class="rf-source">
    <div class="rf-source-title">{index}. {title}</div>
    <div class="rf-source-meta">
        {source_type} &nbsp;•&nbsp; relevance {score:.3f}
    </div>
    <div class="rf-source-url">{url}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_event_log(event_log: list[str]) -> None:
    """Render a compact execution trace."""

    if not event_log:
        return

    with st.expander("Execution trace", expanded=False):
        for event in event_log:
            st.caption(event)


def run_research(question: str) -> None:
    """Execute the ResearchForge graph once and render its live progress."""

    llm = create_llm()
    graph = build_graph(llm)

    statuses = {key: "waiting" for key, _, _ in AGENT_ORDER}
    event_log: list[str] = []

    pipeline_placeholder = st.empty()
    report_placeholder = st.empty()
    source_placeholder = st.empty()
    trace_placeholder = st.empty()

    statuses["planner"] = "active"
    with pipeline_placeholder.container():
        st.markdown(
            '<div class="rf-section-label">Live agent pipeline</div>',
            unsafe_allow_html=True,
        )
        render_pipeline(statuses)

    initial_state: dict[str, Any] = {
        "question": question,
        "plan": [],
        "findings": [],
        "analysis": {},
        "critique": {},
        "report": {},
        "sources": [],
        "events": [],
    }

    final_report: dict[str, Any] = {}
    final_sources: list[dict[str, Any]] = []

    try:
        for update in graph.stream(
            initial_state,
            stream_mode="updates",
        ):
            if not isinstance(update, dict):
                continue

            for node_name, node_update in update.items():
                if not isinstance(node_update, dict):
                    continue

                normalized_node = str(node_name).lower()

                if normalized_node in statuses:
                    statuses[normalized_node] = "done"

                if normalized_node == "planner":
                    statuses["scout"] = "active"

                elif normalized_node == "scout":
                    statuses["scout"] = "active"

                    finding_count = len(
                        node_update.get("findings", [])
                    )

                    if finding_count:
                        event_log.append(
                            f"Scout completed: {finding_count} finding returned."
                        )

                elif normalized_node == "analyst":
                    statuses["scout"] = "done"
                    statuses["analyst"] = "done"
                    statuses["critic"] = "active"

                elif normalized_node == "critic":
                    statuses["analyst"] = "done"
                    statuses["critic"] = "done"
                    statuses["synthesizer"] = "active"

                elif normalized_node == "synthesizer":
                    statuses["critic"] = "done"
                    statuses["synthesizer"] = "done"

                    final_report = node_update.get(
                        "report",
                        {},
                    )

                    final_sources = node_update.get(
                        "sources",
                        [],
                    )

                for event in node_update.get("events", []):
                    if not isinstance(event, dict):
                        continue

                    detail = event.get("detail")
                    agent = event.get("agent")

                    if detail:
                        event_log.append(
                            f"{str(agent).title() if agent else 'Agent'}: {detail}"
                        )

                with pipeline_placeholder.container():
                    st.markdown(
                        '<div class="rf-section-label">Live agent pipeline</div>',
                        unsafe_allow_html=True,
                    )
                    render_pipeline(statuses)

        statuses = {
            key: "done"
            for key, _, _ in AGENT_ORDER
        }

        with pipeline_placeholder.container():
            st.markdown(
                '<div class="rf-section-label">Live agent pipeline</div>',
                unsafe_allow_html=True,
            )
            render_pipeline(statuses)

        st.session_state["research_result"] = {
            "report": final_report,
            "sources": final_sources,
            "events": list(event_log),
        }

        with report_placeholder.container():
            render_report(final_report)

        with source_placeholder.container():
            render_sources(final_sources)

        with trace_placeholder.container():
            render_event_log(event_log)

    except Exception as error:
        statuses = {
            key: ("done" if value == "done" else "waiting")
            for key, value in statuses.items()
        }

        with pipeline_placeholder.container():
            st.markdown(
                '<div class="rf-section-label">Pipeline stopped</div>',
                unsafe_allow_html=True,
            )
            render_pipeline(statuses)

        st.error(
            f"Research failed: {error}"
        )


def main() -> None:
    """Render the ResearchForge application."""

    st.set_page_config(
        page_title="ResearchForge",
        page_icon="🔨",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    font_scale = render_header()
    inject_styles(font_scale)
    render_hero()

    st.markdown(
        '<div class="rf-section-label">Ask ResearchForge</div>',
        unsafe_allow_html=True,
    )

    with st.form("research_form", clear_on_submit=False):
        question = st.text_area(
            "Research question",
            placeholder=(
                "Ask a complex question. ResearchForge will decompose it, "
                "investigate independent angles, audit the evidence, and "
                "forge a final brief."
            ),
            height=110,
            label_visibility="collapsed",
            help="",
        )

        submitted = st.form_submit_button(
            "Research →",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        normalized_question = question.strip()

        if not normalized_question:
            st.warning("Enter a research question first.")
            return

        st.session_state.pop("research_result", None)

        st.markdown(
            f"""
<div class="rf-section-label">Researching</div>
<div class="rf-source-title" style="font-size:1rem; margin-bottom:1rem;">
    {html.escape(normalized_question)}
</div>
""",
            unsafe_allow_html=True,
        )

        run_research(normalized_question)

    elif st.session_state.get("research_result"):
        saved_result = st.session_state["research_result"]
        render_report(saved_result.get("report", {}))
        render_sources(saved_result.get("sources", []))
        render_event_log(saved_result.get("events", []))


if __name__ == "__main__":
    main()
