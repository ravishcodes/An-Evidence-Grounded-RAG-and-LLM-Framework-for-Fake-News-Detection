
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
    "Research prototype | Claim classification "
    "and evidence-grounded fact verification"
)

mode = st.radio(
    "Select analysis mode",
    [
        "Existing ML Classifiers",
        "Live RAG Fact Verification"
    ],
    horizontal=True
)

claim = st.text_area(
    "Enter your news claim",
    height=130,
    placeholder="Example: The Eiffel Tower is located in Berlin."
)


if mode == "Existing ML Classifiers":

    st.info(
        "This mode uses trained claim-only classifiers "
        "and previously saved evidence."
    )

    if st.button("Analyze Claim", type="primary"):

        if not claim.strip():
            st.warning("Please enter a claim.")

        else:
            binary = binary_model.predict([claim])[0]
            six = six_model.predict([claim])[0]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Binary prediction", str(binary))

            with col2:
                st.metric("Six-class prediction", str(six))

            st.info(
                "These predictions are based on the claim text. "
                "They are not independently verified facts."
            )

            match = claim_index.get(normalize(claim))

            st.subheader("Saved Evidence")

            if match:
                cid, record = match

                st.success(
                    f"Saved evidence found for {cid}"
                )

                for i, item in enumerate(
                    record.get("evidence", [])[:5], 1
                ):
                    with st.expander(
                        f"Source {i}: "
                        f"{item.get('title', 'Untitled')}"
                    ):
                        st.write(item.get("snippet", ""))

                        url = item.get("url", "")
                        if url:
                            st.link_button(
                                "Open source",
                                url
                            )

            else:
                st.warning(
                    "No saved evidence exists for this claim."
                )


else:

    st.subheader("Live Evidence-Grounded Verification")

    st.markdown(
        "This mode will retrieve current web evidence, "
        "rank relevant sources, and generate a "
        "five-class fact-checking verdict."
    )

    st.info(
        "Live RAG inference is being integrated. "
        "The backend is not connected yet."
    )

    if st.button(
        "Verify with Live Evidence",
        type="primary",
        disabled=True
    ):
        pass


st.divider()

st.caption(
    "Research prototype. Automated predictions and "
    "generated explanations may be incorrect. "
    "Verify important claims using original sources."
)




# Temporary test: Discover and test Groq models
with st.expander("🔧 Test Groq LLM Connection"):

    if st.button("Find Working Groq Model"):

        try:
            from groq import Groq

            token = st.secrets.get("GROQ_API_KEY")

            if not token:
                st.error("GROQ_API_KEY is missing.")

            else:
                client = Groq(api_key=token)

                with st.spinner("Fetching Groq models..."):
                    models = client.models.list()

                available = [
                    model.id
                    for model in models.data
                    if getattr(model, "active", True)
                ]

                st.success(
                    f"Found {len(available)} active models."
                )

                with st.expander("Available model IDs"):
                    for model_id in available:
                        st.write(model_id)

                # Prefer smaller general-purpose chat models.
                preferred = [
                    "llama-3.1-8b-instant",
                    "llama-3.3-70b-versatile",
                    "openai/gpt-oss-20b",
                    "openai/gpt-oss-120b"
                ]

                ordered = [
                    name for name in preferred
                    if name in available
                ]

                if not ordered:
                    st.warning(
                        "No preferred chat model was found. "
                        "Check the available model IDs."
                    )

                else:
                    for model_id in ordered:

                        try:
                            with st.spinner(
                                f"Testing {model_id}..."
                            ):
                                response = (
                                    client.chat.completions.create(
                                        model=model_id,
                                        messages=[
                                            {
                                                "role": "user",
                                                "content": (
                                                    "Which city is the "
                                                    "Eiffel Tower located in? "
                                                    "Answer briefly."
                                                )
                                            }
                                        ],
                                        temperature=0,
                                        max_tokens=100
                                    )
                                )

                            answer = (
                                response.choices[0]
                                .message.content
                            )

                            st.success(
                                f"WORKING MODEL: {model_id}"
                            )

                            st.write(answer)
                            break

                        except Exception as error:
                            st.warning(
                                f"{model_id}: "
                                f"{type(error).__name__}: "
                                f"{str(error)[:200]}"
                            )

        except Exception as error:
            st.error(
                f"{type(error).__name__}: {str(error)}"
            )



        except Exception as error:
            st.error(
                f"{type(error).__name__}: {error}"
            )
