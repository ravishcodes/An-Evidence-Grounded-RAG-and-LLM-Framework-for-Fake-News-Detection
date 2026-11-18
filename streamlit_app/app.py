
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


# Temporary test: discover available hosted LLMs
with st.expander("🔧 Test Hosted LLM Connection"):

    if st.button("Find Available Models"):

        try:
            from huggingface_hub import HfApi, InferenceClient

            token = st.secrets.get("HF_TOKEN")

            if not token:
                st.error("HF_TOKEN is missing.")
            else:
                api = HfApi(token=token)

                candidates = [
                    "Qwen/Qwen2.5-1.5B-Instruct",
                    "Qwen/Qwen2.5-7B-Instruct",
                    "Qwen/Qwen3-4B-Instruct-2507",
                    "Qwen/Qwen3-8B",
                    "meta-llama/Llama-3.1-8B-Instruct"
                ]

                available = []

                with st.spinner("Checking inference providers..."):
                    for model in candidates:
                        try:
                            info = api.model_info(
                                model,
                                expand=["inferenceProviderMapping"]
                            )

                            mapping = getattr(
                                info,
                                "inference_provider_mapping",
                                None
                            )

                            if mapping:
                                available.append(model)
                                st.write(
                                    f"Available: {model}"
                                )

                        except Exception as error:
                            st.warning(
                                f"Could not check {model}: "
                                f"{type(error).__name__}"
                            )

                if available:
                    st.success(
                        f"Found {len(available)} models "
                        "with provider mappings."
                    )

                    client = InferenceClient(token=token)

                    for model in available:
                        try:
                            response = client.chat_completion(
                                model=model,
                                messages=[
                                    {
                                        "role": "user",
                                        "content": (
                                            "Reply with only the city "
                                            "where the Eiffel Tower is located."
                                        )
                                    }
                                ],
                                max_tokens=80
                            )

                            st.success(
                                f"Working model: {model}"
                            )

                            st.write(
                                response.choices[0].message.content
                            )

                            break

                        except Exception as error:
                            st.warning(
                                f"{model}: "
                                f"{type(error).__name__}: "
                                f"{str(error)[:250]}"
                            )
                else:
                    st.error(
                        "No provider mappings found for "
                        "the candidate models."
                    )

        except Exception as error:
            st.error(
                f"{type(error).__name__}: {error}"
            )
