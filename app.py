"""Streamlit app: browse the results of the six experiments, read straight from the saved notebooks."""
import base64
import json
import re
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="Deep Learning Experiments", page_icon="🧠", layout="wide")
st.title("Deep Learning & Generative Models: Experiments")
st.caption("Six experiments from my MS in AI, each starting from the reference code in Generative Deep Learning "
           "(2nd ed., David Foster) with one change measured. The models were trained offline with "
           "TensorFlow; this app shows their saved results. Source: github.com/Akshu577/deep-learning-experiments")


@st.cache_data
def experiments():
    """Experiment folder -> (title, data, what changed), from the README table."""
    out = {}
    for line in (ROOT / "README.md").read_text().splitlines():
        m = re.match(r"\| (\d\d) \| \[(.+?)\]\((.+?)\) \| (.+?) \| (.+) \|$", line)
        if m:
            text = re.sub(r"\s*\(\[sample output\].*?\)\)?", "", m.group(5))  # drop the relative link
            out[m.group(3)] = (f"{m.group(1)} · {m.group(2)}", m.group(4), text)
    return out


@st.cache_data
def figures(folder):
    """Every image a notebook produced, captioned with the nearest markdown heading or line above it."""
    nb = next((ROOT / folder).glob("*.ipynb"))
    caption, figs = "", []
    for cell in json.loads(nb.read_text())["cells"]:
        if cell["cell_type"] == "markdown":
            src = re.sub(r"<[^>]+>", "", "".join(cell["source"]))  # drop <a name=...> anchors
            lines = [l.strip("# ").strip() for l in src.splitlines() if l.strip()]
            if lines:
                caption = lines[0]
        for o in cell.get("outputs", []):
            if "image/png" in o.get("data", {}):
                figs.append((caption, base64.b64decode(o["data"]["image/png"])))
    return figs


exps = experiments()
folder = st.selectbox("Experiment", list(exps), format_func=lambda f: exps[f][0])
title, data, finding = exps[folder]
st.markdown(f"**Data:** {data}")
st.info(finding)

if folder.startswith("04"):
    text = (ROOT / folder / "generated_text.txt").read_text()
    parts = re.split(r"=+\nTEMPERATURE = ([\d.]+)\n=+\n", text)
    samples = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    temp = st.radio("Sampling temperature", list(samples), horizontal=True,
                    help="Lower temperature picks safer, more repetitive words; higher is more varied and riskier.")
    st.text_area("Generated recipes", samples[temp], height=400)
else:
    figs = figures(folder)
    cols = st.columns(2)
    for n, (caption, img) in enumerate(figs):
        cols[n % 2].image(img, caption=caption or None, width="stretch")
