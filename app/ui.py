import os

import requests
import streamlit as st


API_URL = os.getenv("VULNLOCATOR_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(page_title="VulnLocator", page_icon="🛡️", layout="wide")
st.title("🛡️ VulnLocator")
st.markdown(
    "Paste a function or code snippet to check it with the fine-tuned CodeBERT model. "
    "This tool is a triage aid, not a substitute for security review."
)

code = st.text_area(
    "Code input",
    height=320,
    placeholder="Paste code here...",
)
include_localization = st.checkbox("Show attention-based suspicious tokens", value=True)

if st.button("Analyze", type="primary"):
    if not code.strip():
        st.warning("Please paste some code first.")
    else:
        with st.spinner("Analyzing code..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "code": code,
                        "include_localization": include_localization,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                st.error(f"Could not get a prediction from the API: {exc}")
            else:
                confidence = result["confidence"]
                probability = result["vulnerable_probability"]
                if result["vulnerable"]:
                    st.error(
                        f"⚠️ **Potentially vulnerable** "
                        f"(confidence: {confidence:.1%}; vulnerable probability: {probability:.1%})"
                    )
                else:
                    st.success(
                        f"✅ **No vulnerability detected** "
                        f"(confidence: {confidence:.1%}; vulnerable probability: {probability:.1%})"
                    )

                st.caption(
                    f"Decision threshold: {result['threshold']:.2f} · "
                    f"Device: {result['device']}"
                )
                if include_localization:
                    findings = result["findings"]
                    st.subheader("Most attended code tokens")
                    st.caption(
                        "Attention rollout is a heuristic for review prioritization; "
                        "it does not prove that a token causes a vulnerability."
                    )
                    if findings:
                        st.dataframe(
                            [
                                {
                                    "Line": finding["line"],
                                    "Token": finding["text"],
                                    "Relative attention": round(finding["score"], 3),
                                    "Source line": finding["line_text"],
                                }
                                for finding in findings
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No code tokens were available to localize.")