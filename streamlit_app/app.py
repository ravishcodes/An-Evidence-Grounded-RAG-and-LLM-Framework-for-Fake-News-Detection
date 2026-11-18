
import streamlit as st
import joblib
import json
from pathlib import Path

st.set_page_config(
    page_title="Evidence-Grounded Fake News Detection",
    page_icon="🔎",
    layout="wide"
)

BASE = Path(__file__).parent

@st.cache_resource
def load_models():
    binary = joblib.load(
        BASE / "binary_classifier.joblib"
    )
    six = joblib.load(
        BASE / "six_class_classifier.joblib"
    )
    return binary, six

binary_model, six_model = load_models()

@st.cache_data
def load_evidence():
    path = BASE / "liar_targeted_evidence.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("claims", {})
    return {}

known_evidence = load_evidence()

def normalize(text):
    return " ".join(str(text).lower().split())

claim_index = {
    normalize(record.get("claim", "")): (cid, record)
    for cid, record in known_evidence.items()
    if record.get("claim")
}

st.title("🔎 Evidence-Grounded Fake News Detection")
st.caption(
    "Research prototype | LIAR claim classification "
    "and saved evidence retrieval"
)

st.markdown(
    "Enter a news claim to obtain a binary and "
    "six-class classification."
)

claim = st.text_area(
    "Enter your claim",
    height=130,
    placeholder="Example: Marijuana is less toxic than alcohol."
)

if st.button("Analyze Claim", type="primary"):
    if not claim.strip():
        st.warning("Please enter a claim.")
    else:
        binary = binary_model.predict([claim])[0]
        six = six_model.predict([claim])[0]

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Binary prediction", binary)

        with col2:
            st.metric("Six-class prediction", six)

        st.info(
            "These are claim-only model predictions. "
            "They are not independently verified facts."
        )

        match = claim_index.get(normalize(claim))

        st.subheader("Evidence")

        if match:
            cid, record = match

            st.success(
                f"Saved evidence found for {cid}"
            )

            for i, item in enumerate(
                record.get("evidence", [])[:5], 1
            ):
                with st.expander(
                    f"Source {i}: {item.get('title', 'Untitled')}"
                ):
                    st.write(
                        item.get("snippet", "")
                    )
                    url = item.get("url", "")
                    if url:
                        st.markdown(
                            f"[Open source]({url})"
                        )
        else:
            st.warning(
                "No saved evidence exists for this claim. "
                "Live evidence retrieval has not yet "
                "been integrated."
            )

st.divider()
st.caption(
    "Research prototype — predictions may be incorrect. "
    "Do not treat them as definitive fact-checks."
)
