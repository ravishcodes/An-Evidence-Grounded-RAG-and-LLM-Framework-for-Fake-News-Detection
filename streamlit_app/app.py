import json
import sys
from pathlib import Path

import joblib
import streamlit as st

st.set_page_config(page_title="Evidence-Grounded Fake News Detection", page_icon="🔎", layout="wide")
BASE = Path(__file__).resolve().parent
ROOT = BASE.parent
sys.path.insert(0, str(ROOT / "rag_backend"))

@st.cache_resource
def load_models():
    return (joblib.load(BASE / "binary_classifier.joblib"),
            joblib.load(BASE / "six_class_classifier.joblib"))

@st.cache_data
def load_saved_evidence():
    path = BASE / "liar_targeted_evidence.json"
    if path.exists():
        with path.open(encoding="utf-8") as f:
            return json.load(f).get("claims", {})
    return {}

@st.cache_resource(show_spinner="Loading evidence ranking models (first run may take a few minutes)...")
def load_live_rankers():
    from live_rag import load_rankers
    return load_rankers()

st.title("🔎 Evidence-Grounded Fake News Detection")
st.caption("Live evidence-grounded fact checking | Research prototype")
claim = st.text_area("Enter a claim to verify", height=130, placeholder="Example: Modi is the president of India")

if st.button("🔎 Verify with Live Evidence", type="primary"):
    if not claim.strip():
        st.warning("Please enter a claim.")
    elif len(claim) > 1500:
        st.warning("Please keep your claim under 1,500 characters.")
    elif not st.secrets.get("TAVILY_API_KEY") or not st.secrets.get("GROQ_API_KEY"):
        st.error("TAVILY_API_KEY and GROQ_API_KEY must be configured in Streamlit Secrets.")
    else:
        try:
            from live_rag import run_live_rag
            with st.spinner("Searching live sources, ranking evidence, and verifying the claim..."):
                embedder, reranker = load_live_rankers()
                result = run_live_rag(claim.strip(), st.secrets["TAVILY_API_KEY"],
                                      st.secrets["GROQ_API_KEY"], embedder, reranker)
            st.session_state["rag_result"] = result
            st.session_state["rag_claim"] = claim.strip()
        except Exception as exc:
            st.error(f"Live verification failed: {type(exc).__name__}: {exc}")

if "rag_result" in st.session_state:
    result = st.session_state["rag_result"]
    st.subheader("Evidence-grounded verdict")
    st.caption(f"Claim: {st.session_state.get('rag_claim', '')}")
    st.metric("Verdict", result["verdict"].replace("_", " "))
    st.write(result["explanation"])
    st.caption(f"Hosted inference model: {result['model']}")
    evidence = result.get("evidence", [])
    cited = set(result.get("citations", []))
    st.subheader("Retrieved web evidence")
    if not evidence:
        st.warning("No usable web evidence was retrieved.")
    for item in evidence:
        marker = " • Cited by model" if item["id"] in cited else " • Retrieved, not cited"
        with st.expander(f"{item['id']}: {item['title']}{marker}"):
            st.write(item["snippet"])
            st.link_button("Open original source", item["url"])
    st.caption("Citations are model-selected and ID-checked, but not independently validated for factual support. Verify important claims against original sources.")
    
    # Download the latest fact-checking result
    st.download_button(
        label="📥 Download Fact-Check Result (JSON)",
        data=json.dumps(result, indent=2, ensure_ascii=False, default=str),
        file_name="fact_check_result.json",
        mime="application/json",
        key="download_fact_check"
    )

with st.expander("Previous ML classifiers (claim-only baseline)"):
    st.write("These models predict labels from the claim text without live evidence verification.")
    if st.button("Run baseline classifiers"):
        if not claim.strip():
            st.warning("Enter a claim first.")
        else:
            try:
                binary_model, six_model = load_models()
                a, b = st.columns(2)
                a.metric("Binary prediction", str(binary_model.predict([claim])[0]))
                b.metric("Six-class prediction", str(six_model.predict([claim])[0]))
                index = {" ".join(str(v.get("claim", "")).lower().split()): (k, v)
                         for k, v in load_saved_evidence().items() if v.get("claim")}
                match = index.get(" ".join(claim.lower().split()))
                if match:
                    st.write("Saved evidence for", match[0])
                    for item in match[1].get("evidence", [])[:5]:
                        st.write(item.get("title", "Untitled"), item.get("snippet", ""))
                        if item.get("url", "").startswith(("https://", "http://")):
                            st.link_button("Open saved source", item["url"])
            except Exception as exc:
                st.error(f"Baseline prediction failed: {type(exc).__name__}: {exc}")

st.divider()
st.caption("Research prototype. A generated verdict is not a definitive fact-check. Search results and external model output can be incomplete or misleading.")
