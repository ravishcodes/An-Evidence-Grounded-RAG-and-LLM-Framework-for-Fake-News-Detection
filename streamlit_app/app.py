"""Streamlit research dashboard: live fact-checking plus transparent evidence diagnostics."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Evidence-Grounded Fake News Detection", page_icon="🔎", layout="wide")
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "rag_backend"))

from live_rag import load_rankers, run_live_rag, result_diagnostics, VERDICTS

LABELS = ["TRUE", "FALSE", "PARTIALLY_TRUE", "MISLEADING", "UNVERIFIED"]

@st.cache_resource
def load_baselines():
    return joblib.load(BASE / "binary_classifier.joblib"), joblib.load(BASE / "six_class_classifier.joblib")

@st.cache_resource(show_spinner="Loading MiniLM and cross-encoder (first run may take a few minutes)...")
def cached_rankers():
    return load_rankers()

@st.cache_data
def load_saved_evidence():
    p = BASE / "liar_targeted_evidence.json"
    if p.exists():
        with p.open(encoding="utf-8") as f:
            return json.load(f).get("claims", {})
    return {}

st.title("🔎 Evidence-Grounded Fake News Detection")
st.caption("Live RAG research dashboard • Tavily + BM25 + MiniLM + cross-encoder + Groq")
claim = st.text_area("Enter a factual claim", height=120, placeholder="Example: The Eiffel Tower is located in Paris, France.")

if st.button("🔎 Verify with Live Evidence", type="primary"):
    if not claim.strip():
        st.warning("Please enter a claim.")
    elif len(claim) > 1500:
        st.warning("Please keep your claim under 1,500 characters.")
    elif not st.secrets.get("TAVILY_API_KEY") or not st.secrets.get("GROQ_API_KEY"):
        st.error("Configure TAVILY_API_KEY and GROQ_API_KEY in Streamlit Secrets.")
    else:
        try:
            with st.spinner("Retrieving evidence, ranking passages, auditing citations, and verifying..."):
                embedder, reranker = cached_rankers()
                result = run_live_rag(claim.strip(), st.secrets["TAVILY_API_KEY"],
                                      st.secrets["GROQ_API_KEY"], embedder, reranker)
            st.session_state["rag_result"] = result
            st.session_state["rag_claim"] = claim.strip()
        except Exception as exc:
            st.error(f"Live verification failed: {type(exc).__name__}: {exc}")

if "rag_result" in st.session_state:
    result = st.session_state["rag_result"]
    diagnostics = result.get("diagnostics") or result_diagnostics(result)
    st.subheader("Evidence-grounded verdict")
    st.caption("Claim: " + st.session_state.get("rag_claim", ""))
    a, b, c, d = st.columns(4)
    a.metric("Verdict", result.get("verdict", "UNVERIFIED").replace("_", " "))
    b.metric("Evidence strength index", f"{diagnostics['evidence_strength_index']}/100")
    c.metric("Audited / retrieved", f"{diagnostics['audited_sources']}/{diagnostics['retrieved_sources']}")
    d.metric("Cited sources", diagnostics["cited_sources"])
    st.write(result.get("explanation", "No explanation supplied."))
    st.warning("Evidence strength is a heuristic indicator, NOT a calibrated confidence percentage, factual accuracy, or probability that the verdict is correct.")
    if result.get("verdict") == "UNVERIFIED":
        st.info("The system abstained: this is not evidence that the claim is true or false.")
    st.caption("Hosted model: " + result.get("model", "unknown") + " | Retrieval date: " + result.get("retrieved_at_utc", "not recorded"))

    st.subheader("Evidence and citation diagnostics")
    x, y, z = st.columns(3)
    x.metric("Supporting cited passages", diagnostics["supporting_citations"])
    y.metric("Refuting cited passages", diagnostics["refuting_citations"])
    z.metric("Citation coverage", f"{diagnostics['citation_coverage_percent']}%")
    st.caption("Citation stances are produced by an LLM audit and checked for verbatim quotations; they are not independent human verification.")
    st.caption(diagnostics["temporal_coverage"])
    evidence = result.get("evidence", [])
    cited = set(result.get("citations", []))
    audited = {item.get("id"): item for item in result.get("audit", []) if isinstance(item, dict)}
    if not evidence:
        st.warning("No usable evidence was retrieved.")
    for item in evidence:
        eid = item.get("id", "?")
        with st.expander(f"{eid}: {item.get('title', 'Untitled')} — {'CITED' if eid in cited else 'not cited'}"):
            st.write(item.get("snippet", ""))
            if item.get("url", "").startswith(("http://", "https://")):
                st.link_button("Open original source", item["url"])
            st.caption("Domain: " + item.get("domain", "unknown"))
            st.caption("Source type: " + item.get("source_category", "Not categorized"))
            audit = audited.get(eid)
            if audit:
                st.write("**Automated stance:**", audit.get("stance", "Unknown"))
                st.write("**Exact cited passage:**", audit.get("quote", ""))
            else:
                st.caption("No accepted audit quotation for this source.")
            st.write("**Retrieval diagnostics (not probabilities)**")
            scores = {
                "BM25 raw": item.get("bm25_raw"),
                "MiniLM cosine similarity": item.get("semantic_similarity"),
                "Hybrid normalized": item.get("hybrid_normalized"),
                "Cross-encoder raw": item.get("cross_encoder_raw"),
                "Final relative ranking score": item.get("retrieval_score"),
            }
            st.json({k: v for k, v in scores.items() if v is not None})
    st.download_button("📥 Download Fact-Check Result (JSON)",
                       data=json.dumps(result, ensure_ascii=False, indent=2, default=str),
                       file_name="fact_check_result.json", mime="application/json")

with st.expander("📊 Research evaluation (upload labeled predictions)"):
    st.write("Upload a CSV containing **actual_label** and **predicted_label** for independent, unseen test claims. No metrics are invented from single examples.")
    template = "claim,actual_label,predicted_label\n"
    st.download_button("Download blank evaluation template", template,
                       file_name="evaluation_template.csv", mime="text/csv")
    uploaded = st.file_uploader("Evaluation CSV", type=["csv"], key="evaluation_csv")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            if not {"actual_label", "predicted_label"}.issubset(df.columns):
                st.error("CSV must contain actual_label and predicted_label columns.")
            elif df.empty:
                st.warning("Add independently labeled rows before evaluation.")
            else:
                y_true = df["actual_label"].astype(str).str.strip().str.upper()
                y_pred = df["predicted_label"].astype(str).str.strip().str.upper()
                bad = ~(y_true.isin(VERDICTS) & y_pred.isin(VERDICTS))
                if bad.any():
                    st.error(f"{int(bad.sum())} row(s) have invalid or missing labels. Use: {', '.join(LABELS)}")
                else:
                    report = classification_report(y_true, y_pred, labels=LABELS,
                                                   output_dict=True, zero_division=0)
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Evaluated claims", len(df))
                    m2.metric("Accuracy", f"{accuracy_score(y_true, y_pred):.1%}")
                    m3.metric("Macro-F1", f"{report['macro avg']['f1-score']:.3f}")
                    st.write("**Per-class precision, recall, F1**")
                    st.dataframe(pd.DataFrame(report).T.loc[LABELS, ["precision", "recall", "f1-score", "support"]], use_container_width=True)
                    st.write("**Confusion matrix (rows = actual, columns = predicted)**")
                    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
                    st.dataframe(pd.DataFrame(matrix, index=LABELS, columns=LABELS), use_container_width=True)
                    st.metric("UNVERIFIED prediction rate", f"{(y_pred == 'UNVERIFIED').mean():.1%}")
                    st.download_button("Download evaluation metrics (JSON)",
                                       json.dumps(report, indent=2), file_name="evaluation_metrics.json",
                                       mime="application/json")
        except Exception as exc:
            st.error(f"Could not evaluate CSV: {type(exc).__name__}: {exc}")

with st.expander("🧪 Previous ML classifiers (claim-only baselines)"):
    st.caption("These baseline labels are not evidence-grounded and are not directly comparable to the five-class RAG labels without a documented mapping.")
    if st.button("Run baseline classifiers"):
        if not claim.strip():
            st.warning("Enter a claim first.")
        else:
            try:
                binary_model, six_model = load_baselines()
                c1, c2 = st.columns(2)
                c1.metric("Binary", str(binary_model.predict([claim])[0]))
                c2.metric("Six-class", str(six_model.predict([claim])[0]))
                saved = load_saved_evidence()
                idx = {" ".join(str(v.get("claim", "")).lower().split()): (k, v)
                       for k, v in saved.items() if v.get("claim")}
                match = idx.get(" ".join(claim.lower().split()))
                if match:
                    st.write("Saved evidence for", match[0])
                    for item in match[1].get("evidence", [])[:5]:
                        st.write(item.get("title", "Untitled"), item.get("snippet", ""))
                        url = item.get("url", "")
                        if url.startswith(("https://", "http://")):
                            st.link_button("Open saved source", url)
            except Exception as exc:
                st.error(f"Baseline prediction failed: {type(exc).__name__}: {exc}")

st.divider()
st.caption("Research prototype. Source type is a domain heuristic, not a credibility guarantee. Publication-date validity, calibration, and independent evaluation remain open research tasks.")
