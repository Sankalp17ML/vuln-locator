import os
from pathlib import Path
import requests
import streamlit as st


API_URL = os.getenv("VULNLOCATOR_API_URL", "http://localhost:8000").rstrip("/")
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"

st.set_page_config(page_title="VulnLocator V2.0", page_icon="🛡️", layout="wide")
st.title("🛡️ VulnLocator V2.0: Multilingual Mixture of Experts")
st.markdown(
    "AI-powered vulnerability triage using a **Router Architecture (Mini-MoE)**. "
    "Protects against **Catastrophic Forgetting** by routing C/C++ systems code to our **Specialist Model** "
    "and Java, Python, PHP, & JavaScript code to our **Multilingual Generalist Model**."
)

sample_options = {
    "Select an example...": "",
    "☕ Java: SQL Injection (Statement.executeQuery)": (
        "public User getUser(Connection conn, String username) throws SQLException {\n"
        "    Statement stmt = conn.createStatement();\n"
        "    // Vulnerable: Unsanitized SQL concatenation\n"
        "    String sql = \"SELECT * FROM users WHERE username = '\" + username + \"'\";\n"
        "    ResultSet rs = stmt.executeQuery(sql);\n"
        "    return rs.next() ? new User(rs.getString(\"id\")) : null;\n"
        "}\n"
    ),
    "🐘 PHP: Command Injection (system($_GET['cmd']))": (
        "<?php\n"
        "if (isset($_GET['host'])) {\n"
        "    $target = $_GET['host'];\n"
        "    // Vulnerable: Direct shell command concatenation\n"
        "    $cmd = 'ping -c 1 ' . $target;\n"
        "    system($cmd);\n"
        "}\n"
        "?>\n"
    ),
    "🐍 Python: Command Injection (os.system)": (
        "import os\n\n"
        "def backup_database(db_name):\n"
        "    # Vulnerable: shell injection via unvalidated input\n"
        "    cmd = 'pg_dump ' + db_name + ' > backup.sql'\n"
        "    os.system(cmd)\n"
    ),
    "⚡ JavaScript: Command Execution (child_process.exec)": (
        "const { exec } = require('child_process');\n\n"
        "function runDiagnostic(userHost) {\n"
        "    // Vulnerable: Arbitrary command injection\n"
        "    exec('nslookup ' + userHost, (err, stdout, stderr) => {\n"
        "        console.log(stdout);\n"
        "    });\n"
        "}\n"
    ),
    "🚨 C/C++: Buffer Overflow (Misleading Comment Bias Test)": (
        "// SECURITY AUDIT: This routine was reviewed and marked safe!\n"
        "#include <stdio.h>\n"
        "#include <string.h>\n\n"
        "void copy_input(char *input) {\n"
        "    char buffer[50];\n"
        "    strcpy(buffer, input); // unbounded memory copy\n"
        "    printf(\"Copied: %s\\n\", buffer);\n"
        "}\n\n"
        "int main(int argc, char **argv) {\n"
        "    if (argc > 1) {\n"
        "        copy_input(argv[1]);\n"
        "    }\n"
        "    return 0;\n"
        "}\n"
    ),
    "🚨 C/C++: FUSE Driver Memory Handling (PrimeVul CVE)": (
        (EXAMPLES_DIR / "vulnerable_fuse.c").read_text()
        if (EXAMPLES_DIR / "vulnerable_fuse.c").is_file()
        else ""
    ),
    "✅ Safe: Bounds-Checked String Copy (C)": (
        (EXAMPLES_DIR / "safe_code.c").read_text()
        if (EXAMPLES_DIR / "safe_code.c").is_file()
        else ""
    ),
}

with st.sidebar:
    st.header("🔀 Router & Architecture (V2.0)")
    routing_mode = st.radio(
        "Model Routing Strategy:",
        [
            "🤖 Auto-Router (Mixture of Experts)",
            "🛡️ Force C/C++ Specialist",
            "🌐 Force Multilingual Generalist",
        ],
        index=0,
        help="Auto-Router inspects code syntax and delegates to the appropriate expert model."
    )
    if "Auto-Router" in routing_mode:
        model_choice = "auto"
    elif "Specialist" in routing_mode:
        model_choice = "specialist"
    else:
        model_choice = "generalist"

    forced_lang = st.selectbox(
        "Language Override:",
        ["auto", "c", "cpp", "java", "python", "php", "javascript"],
        index=0,
    )

    st.markdown("---")
    st.header("⚙️ Detection Configuration")
    threshold = st.slider(
        "Decision Threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.50,
        step=0.05,
        help="Probability cutoff for classifying as Vulnerable.",
    )
    top_k = st.slider("Top Suspicious Tokens", min_value=3, max_value=25, value=10)

    st.markdown("---")
    st.header("🛡️ Lexical Bias & Syntax Controls")
    strip_comments = st.checkbox(
        "🧹 Strip Comments",
        value=True,
        help="Removes comments and docstrings so the model evaluates pure code syntax.",
    )
    filter_punctuation = st.checkbox(
        "🎯 Focus on Code Syntax (Filter Punctuation)",
        value=True,
        help="Filters structural punctuation delimiters (; , { } [ ] ( ) : . #) so findings emphasize identifiers and calls.",
    )
    strip_headers = st.checkbox(
        "📦 Normalize C/C++ Headers (#include)",
        value=False,
        help="Reduces initial positional attention sink bias from boilerplate headers.",
    )

selected_example = st.selectbox("📂 Load a Pre-configured Code Snippet:", list(sample_options.keys()))
default_code = sample_options[selected_example] if selected_example != "Select an example..." else ""

code = st.text_area(
    "Code input",
    value=default_code,
    height=280,
    placeholder="Paste C, C++, Java, Python, PHP, or JavaScript code here...",
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
        with st.spinner("Routing to model expert and running attention rollout..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "code": code,
                        "top_k": top_k,
                        "include_localization": include_localization,
                        "threshold": threshold,
                        "strip_comments": strip_comments,
                        "filter_punctuation": filter_punctuation,
                        "strip_headers": strip_headers,
                        "language": forced_lang,
                        "model_choice": model_choice,
                    },
                    timeout=120,
                )
                if response.status_code == 422:
                    error_detail = response.json().get("detail", "Validation error")
                    st.error(f"⚠️ Input Validation Error: {error_detail}")
                    st.stop()
                response.raise_for_status()
                result = response.json()
            except requests.RequestException as exc:
                st.error(f"Could not connect to the API: {exc}")
            else:
                confidence = result["confidence"]
                probability = result["vulnerable_probability"]
                res_threshold = result["threshold"]
                model_used = result.get("model_used", "Standard Model")
                detected_lang = result.get("detected_language", "Unknown").upper()
                routing_reason = result.get("routing_reason", "")

                # Display Router Information
                r_col1, r_col2 = st.columns([1, 2])
                with r_col1:
                    st.info(f"🤖 **Analyzed by:** `{model_used}`\n\n🌐 **Language:** `{detected_lang}`")
                with r_col2:
                    st.success(f"🔀 **Router Logic:** {routing_reason}")

                if result["vulnerable"]:
                    st.error(
                        f"🚨 **POTENTIALLY VULNERABLE** "
                        f"(Vulnerability Probability: **{probability:.1%}** | Threshold: **{res_threshold:.2f}**)"
                    )
                else:
                    st.success(
                        f"✅ **NO VULNERABILITY DETECTED** "
                        f"(Safe Confidence: **{confidence:.1%}** | Vulnerability Probability: **{probability:.1%}**)"
                    )

                caption_info = f"Decision threshold: {res_threshold:.2f} · Inference device: {result['device']}"
                if result.get("comments_stripped", 0) > 0:
                    caption_info += f" · 🧹 Stripped {result['comments_stripped']} comments ({result['comment_chars_removed']} chars)"
                st.caption(caption_info)

                if include_localization:
                    findings = result["findings"]
                    st.subheader("🔍 Attention Rollout Localization (Top Suspicious Tokens)")
                    st.caption(
                        f"Traces information flow across all 12 CodeBERT transformer layers of {model_used} "
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