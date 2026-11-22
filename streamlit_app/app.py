"""Streamlit research dashboard: live fact-checking plus transparent evidence diagnostics."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Verity AI | Evidence Intelligence", page_icon="🔎", layout="wide", initial_sidebar_state="expanded")
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

# Presentation-ready styling. Backend inference and research calculations are unchanged.
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root {color-scheme: dark;}
html, body, [class*="css"], [data-testid="stApp"] {font-family: 'DM Sans', sans-serif;}
[data-testid="stApp"] {background: radial-gradient(ellipse at 82% 0%, #172752 0%, #0a1022 39%, #080d1b 85%); color: #eaf0ff;}
[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1250px; padding-top: 2.2rem; padding-bottom: 3rem;}
h1, h2, h3 {font-family: 'Space Grotesk', sans-serif; letter-spacing: -.035em;}
h1 {font-size: 2.6rem !important;}
h3 {color: #dfe8ff;}
[data-testid="stSidebar"] {background: #0b1329; border-right: 1px solid #253354;}
[data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}
[data-testid="stMetric"] {background: linear-gradient(145deg,#152340,#101a31); border:1px solid #2b3d60; border-radius: 16px; padding: 18px 17px; min-height: 110px;}
[data-testid="stMetricLabel"] {color: #9caed0; font-size: .84rem;}
[data-testid="stMetricValue"] {font-family:'Space Grotesk',sans-serif; color:#f5f7ff; font-size: 1.65rem;}
[data-testid="stTextArea"] textarea {background:#111e37 !important; border:1px solid #3c5176 !important; border-radius: 14px !important; color:#f5f8ff !important; padding:18px !important; font-size:1rem !important;}
[data-testid="stTextArea"] textarea:focus {border-color:#8b8cff !important; box-shadow:0 0 0 1px #8b8cff !important;}
.stButton > button[kind="primary"] {background:linear-gradient(100deg,#7161f3,#4c9af5); border:0; border-radius:12px; color:white; font-weight:700; padding:.75rem 1.6rem; box-shadow:0 8px 28px #5557e633;}
.stButton > button[kind="primary"]:hover {background:linear-gradient(100deg,#8375ff,#60a9ff); color:white;}
.stButton > button:not([kind="primary"]), [data-testid="stDownloadButton"] button {border-radius:11px; border:1px solid #354768; background:#15233c; color:#e4ebff;}
[data-testid="stExpander"] {background:#111c33; border:1px solid #293b5a; border-radius:14px; overflow:hidden;}
[data-testid="stAlert"] {border-radius:12px;}
[data-testid="stDataFrame"] {border-radius:12px; overflow:hidden;}
a {color:#a5b4ff !important;}
.hero {padding:30px 34px; border:1px solid #344571; border-radius:22px; background:linear-gradient(115deg,rgba(92,77,191,.25),rgba(27,62,108,.18) 55%,rgba(10,20,42,.7)); margin-bottom:25px;}
.eyebrow {font-size:.73rem; font-weight:800; letter-spacing:.18em; color:#a9b5ff; margin-bottom:12px;}
.hero-title {font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:clamp(2rem,4vw,3.1rem); letter-spacing:-.045em; line-height:1.12; margin:0 0 12px; color:#fff;}
.hero-copy {color:#aebddd; font-size:1rem; line-height:1.6; max-width:760px;}
.pill {display:inline-block; background:#1c3053; border:1px solid #3a537c; border-radius:50px; padding:6px 12px; color:#cbd9ff; font-size:.76rem; margin:12px 6px 0 0;}
.section-kicker {font-size:.73rem; letter-spacing:.14em; color:#8eabff; font-weight:800; text-transform:uppercase; margin:20px 0 4px;}
.verdict {padding:18px 22px; border-radius:16px; border:1px solid #43557a; background:#15213a; margin:10px 0 20px;}
.verdict-label {font-size:.72rem; letter-spacing:.14em; color:#a6b8d7; font-weight:800;}
.verdict-name {font-family:'Space Grotesk',sans-serif; font-size:1.8rem; font-weight:700; margin-top:4px;}
.verdict.true {border-color:#246d65;background:#10332f;} .verdict.true .verdict-name {color:#75e2bb;}
.verdict.false {border-color:#88475b;background:#351c2b;} .verdict.false .verdict-name {color:#ffa2b4;}
.verdict.partial {border-color:#886a38;background:#332a1d;} .verdict.partial .verdict-name {color:#f9d38b;}
.verdict.unknown {border-color:#4a587b;background:#1a2740;} .verdict.unknown .verdict-name {color:#b5c9ff;}
.footer-note {color:#8fa3c4; font-size:.78rem; line-height:1.6;}
@media(max-width:700px){.hero{padding:23px 19px}.block-container{padding-top:1rem}h1{font-size:2rem!important}}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ◈ VERITY AI")
    st.caption("EVIDENCE INTELLIGENCE LAB")
    st.divider()
    st.markdown("**LIVE RESEARCH STACK**")
    st.markdown("🌐 **Tavily** · Web evidence")
    st.markdown("🔤 **BM25** · Keyword retrieval")
    st.markdown("🧠 **MiniLM** · Semantic search")
    st.markdown("🎯 **Cross-encoder** · Reranking")
    st.markdown("⚡ **Groq** · LLM verification")
    st.divider()
    st.markdown("**VERDICT CLASSES**")
    st.caption("TRUE · FALSE · PARTIALLY TRUE · MISLEADING · UNVERIFIED")
    st.divider()
    st.caption("Research prototype • Results require independent validation")

st.markdown("""<div class="hero">
  <div class="eyebrow">RESEARCH DASHBOARD / NLP FACT VERIFICATION</div>
  <div class="hero-title">Verify claims.<br>Follow the evidence.</div>
  <div class="hero-copy">An explainable fact-checking workspace combining live web retrieval, semantic search, evidence ranking and language-model reasoning.</div>
  <span class="pill">✦ Live evidence retrieval</span><span class="pill">◈ Source diagnostics</span><span class="pill">◎ Research evaluation</span>
</div>""", unsafe_allow_html=True)
st.markdown('<div class="section-kicker">01 / Claim verification</div>', unsafe_allow_html=True)
st.subheader("What would you like to fact-check?")
claim = st.text_area("Enter a factual claim", height=140, placeholder="Example: The Eiffel Tower is located in Paris, France.", help="Enter one specific, verifiable claim (maximum 1,500 characters).")
st.caption("Enter one specific claim. Live verification requires Tavily and Groq API keys configured in Streamlit Secrets.")

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
    st.markdown('<div class="section-kicker">02 / Verification result</div>', unsafe_allow_html=True)
    st.subheader("Evidence-grounded verdict")
    verdict = result.get("verdict", "UNVERIFIED")
    verdict_class = "true" if verdict == "TRUE" else "false" if verdict == "FALSE" else "partial" if verdict in ("PARTIALLY_TRUE", "MISLEADING") else "unknown"
    st.markdown(f'<div class="verdict {verdict_class}"><div class="verdict-label">AI VERIFICATION VERDICT</div><div class="verdict-name">{verdict.replace("_", " ")}</div></div>', unsafe_allow_html=True)
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

    st.markdown('<div class="section-kicker">03 / Evidence transparency</div>', unsafe_allow_html=True)
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

st.markdown('<div class="section-kicker">04 / Research laboratory</div>', unsafe_allow_html=True)
st.subheader("Evaluation & model analysis")
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

with st.expander("🔬 Run a real labeled RAG evaluation (live API calls)"):
    st.write("Upload an independently labeled CSV with columns **claim,actual_label**. This runs your deployed RAG on each claim and calculates real metrics. API usage may incur costs and take several minutes.")
    st.download_button("Download evaluation input template", "claim,actual_label\n", "labeled_claims_template.csv", "text/csv", key="download_input_template")
    batch_file = st.file_uploader("Labeled test claims CSV", type=["csv"], key="batch_input")
    max_cases = st.number_input("Maximum claims this run", min_value=1, max_value=100, value=10, step=1)
    if batch_file is not None:
        try:
            candidate = pd.read_csv(batch_file)
            if not {"claim", "actual_label"}.issubset(candidate.columns):
                st.error("CSV needs claim and actual_label columns")
            elif candidate.empty:
                st.warning("No claims supplied")
            else:
                candidate["actual_label"] = candidate["actual_label"].astype(str).str.strip().str.upper()
                candidate["claim"] = candidate["claim"].fillna("").astype(str).str.strip()
                if not candidate["actual_label"].isin(LABELS).all() or not candidate["claim"].str.len().between(1,1500).all():
                    st.error("Fix invalid labels or empty/oversized claims before running")
                else:
                    st.caption(f"Ready: {len(candidate)} labeled claims; first {min(len(candidate),int(max_cases))} will run")
                    if st.button("Run live evaluation", type="primary"):
                        if not st.secrets.get("TAVILY_API_KEY") or not st.secrets.get("GROQ_API_KEY"):
                            st.error("Configure both API keys in Streamlit Secrets")
                        else:
                            from research_eval import summarize_runs
                            embedder, reranker = cached_rankers()
                            runs=[]
                            progress=st.progress(0)
                            selected=candidate.head(int(max_cases))
                            for i, row in enumerate(selected.itertuples(index=False)):
                                try:
                                    outcome=run_live_rag(row.claim,st.secrets["TAVILY_API_KEY"],st.secrets["GROQ_API_KEY"],embedder,reranker)
                                    runs.append({"claim":row.claim,"actual_label":row.actual_label,"predicted_label":outcome["verdict"],"error":"","result_json":json.dumps(outcome,ensure_ascii=False)})
                                except Exception as exc:
                                    runs.append({"claim":row.claim,"actual_label":row.actual_label,"predicted_label":"","error":f"{type(exc).__name__}: {exc}","result_json":""})
                                progress.progress((i+1)/len(selected))
                            st.session_state["research_runs"] = runs
        except Exception as exc:
            st.error(f"Could not read input: {type(exc).__name__}: {exc}")
    if st.session_state.get("research_runs"):
        from research_eval import summarize_runs
        all_runs=pd.DataFrame(st.session_state["research_runs"])
        successful=all_runs[all_runs["error"]==""].copy()
        st.metric("Successful / attempted", f"{len(successful)}/{len(all_runs)}")
        if len(successful):
            results, stats=summarize_runs(successful.to_dict("records"))
            c1,c2,c3,c4=st.columns(4)
            c1.metric("Accuracy",f"{stats['accuracy']:.1%}")
            c2.metric("Macro F1",f"{stats['macro_f1']:.3f}")
            c3.metric("Macro precision",f"{stats['macro_precision']:.3f}")
            c4.metric("Macro recall",f"{stats['macro_recall']:.3f}")
            st.metric("Abstention rate",f"{stats['abstention_rate']:.1%}")
            st.dataframe(pd.DataFrame(stats["classification_report"]).T.loc[LABELS,["precision","recall","f1-score","support"]],use_container_width=True)
            st.write("Confusion matrix (rows=actual, columns=predicted)")
            st.dataframe(pd.DataFrame(stats["confusion_matrix"],index=LABELS,columns=LABELS),use_container_width=True)
            st.write("Citation audit proxies (not human faithfulness assessment)")
            st.json({k:v for k,v in stats.items() if k.endswith("_mean")})
            st.download_button("Download evaluation metrics JSON",json.dumps(stats,indent=2),"research_metrics.json","application/json",key="download_research_metrics")
        st.download_button("Download all run records CSV",all_runs.to_csv(index=False),"research_predictions.csv","text/csv",key="download_research_predictions")
        if (all_runs["error"]!="").any():
            st.warning("Failed API requests were excluded from accuracy calculations. Download the CSV to inspect them.")
    st.caption("S1-S5 comparisons, retrieval Recall@k/MRR, confidence calibration, and temporal ablations require separate ground truth and controlled experimental runs; they are not claimed by this dashboard.")

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
