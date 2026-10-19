import os
from pathlib import Path
import requests
import streamlit as st


API_URL = os.getenv("VULNLOCATOR_API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

st.set_page_config(page_title="VulnLocator", page_icon="🛡️", layout="wide")
st.title("🛡️ VulnLocator")
st.markdown(
    "Paste a function or code snippet to check it with the fine-tuned CodeBERT model. "
    "Attention Rollout visualizes the most attended tokens to explain the model's focus."
)

# Load sample examples from examples directory if available
sample_options = {
    "Select an example...": "",
    "🚨 Vulnerable: FUSE Driver Memory Handling (PrimeVul CVE)": (
        (EXAMPLES_DIR / "vulnerable_fuse.c").read_text()
        if (EXAMPLES_DIR / "vulnerable_fuse.c").is_file()
        else ""
    ),
    "🚨 Vulnerable: XFS ACL Disk Deserialization (PrimeVul CVE)": (
        (EXAMPLES_DIR / "vulnerable_xfs.c").read_text()
        if (EXAMPLES_DIR / "vulnerable_xfs.c").is_file()
        else ""
    ),
    "⚠️ Vulnerable: SQL String Concatenation (Python)": (
        (EXAMPLES_DIR / "vulnerable_sql.py").read_text()
        if (EXAMPLES_DIR / "vulnerable_sql.py").is_file()
        else ""
    ),
    "✅ Safe: Bounds-Checked String Copy (C)": (
        (EXAMPLES_DIR / "safe_code.c").read_text()
        if (EXAMPLES_DIR / "safe_code.c").is_file()
        else ""
    ),
}

with st.sidebar:
    st.header("⚙️ Configuration")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Probability cutoff for classifying as Vulnerable. Lower thresholds increase recall for security triage.",
    )
    top_k = st.slider("Top Suspicious Tokens", min_value=3, max_value=25, value=10)
    st.markdown("---")
    st.markdown(
        "**API Server:** `" + API_URL + "`\n\n"
        "**Model:** `microsoft/codebert-base` (Fine-tuned on PrimeVul)"
    )

selected_example = st.selectbox("📂 Load a Pre-configured Code Snippet:", list(sample_options.keys()))

default_code = sample_options[selected_example] if selected_example != "Select an example..." else ""

code = st.text_area(
    "Code input",
    value=default_code,
    height=280,
    placeholder="Paste C, C++, or Python code here...",
)

col1, col2 = st.columns([1, 4])
with col1:
    analyze_clicked = st.button("🔍 Analyze Code", type="primary", use_container_width=True)
with col2:
    include_localization = st.checkbox("Show attention-based suspicious tokens", value=True)

if analyze_clicked:
    if not code.strip():
        st.warning("Please paste some code first.")
    else:
        with st.spinner("Analyzing code representations and attention matrices..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "code": code,
                        "top_k": top_k,
                        "include_localization": include_localization,
                        "threshold": threshold,
                    },
                    timeout=120,
                )
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                st.error(f"Could not connect to the API: {exc}")
            else:
                confidence = result["confidence"]
                probability = result["vulnerable_probability"]
                res_threshold = result["threshold"]

                if result["vulnerable"]:
                    st.error(
                        f"🚨 **POTENTIALLY VULNERABLE** "
                        f"(Vulnerable Probability: **{probability:.1%}** | Threshold: **{res_threshold:.2f}**)"
                    )
                else:
                    st.success(
                        f"✅ **NO VULNERABILITY DETECTED** "
                        f"(Safe Confidence: **{confidence:.1%}** | Vulnerable Probability: **{probability:.1%}**)"
                    )

                st.caption(
                    f"Decision threshold: {res_threshold:.2f} · "
                    f"Inference device: {result['device']}"
                )

                if include_localization:
                    findings = result["findings"]
                    st.subheader("🔍 Attention Rollout Localization (Top Suspicious Tokens)")
                    st.caption(
                        "Attention rollout traces information flow across all 12 CodeBERT transformer layers "
                        "with residual connections to highlight the most influential tokens."
                    )
                    if findings:
                        st.dataframe(
                            [
                                {
                                    "Line": finding["line"],
                                    "Token": finding["text"],
                                    "Relative Attention": round(finding["score"], 3),
                                    "Source Line": finding["line_text"],
                                }
                                for finding in findings
                            ],
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        st.info("No code tokens were available to localize.")