import io
import json
import re
import random
import string
from datetime import datetime
from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html as st_html  # inline HTML viewer
from PIL import Image
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from openai import AzureOpenAI
import boto3  # S3 upload

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(
    page_title="Notes → OCR → Quiz → AMP",
    page_icon="🧠",
    layout="centered"
)
st.title("🧠 Notes/Quiz OCR → GPT Structuring → AMP Web Story")
st.caption("Upload notes/quiz as image(s) or PDF(s), or provide a ready JSON + an AMP HTML template → download timestamped final HTML.")

# ---------------------------
# Secrets / Config (from st.secrets)
# ---------------------------
try:
    # Azure
    AZURE_DI_ENDPOINT = st.secrets["AZURE_DI_ENDPOINT"]      # https://<your-di>.cognitiveservices.azure.com/
    AZURE_API_KEY = st.secrets["AZURE_API_KEY"]

    AZURE_OPENAI_ENDPOINT = st.secrets["AZURE_OPENAI_ENDPOINT"]  # https://<your-openai>.openai.azure.com/
    AZURE_OPENAI_API_VERSION = st.secrets.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    AZURE_OPENAI_API_KEY = st.secrets.get("AZURE_OPENAI_API_KEY", AZURE_API_KEY)  # reuse if same key
    GPT_DEPLOYMENT = st.secrets.get("GPT_DEPLOYMENT", "gpt-4")  # your deployment name

    # AWS / S3
    AWS_ACCESS_KEY_ID     = st.secrets["AWS_ACCESS_KEY_ID"]
    AWS_SECRET_ACCESS_KEY = st.secrets["AWS_SECRET_ACCESS_KEY"]
    AWS_REGION            = st.secrets.get("AWS_REGION", "ap-south-1")
    AWS_BUCKET            = st.secrets.get("AWS_BUCKET", "suvichaarapp")

    # HTML upload path + CDN
    HTML_S3_PREFIX = st.secrets.get("HTML_S3_PREFIX", "webstory-html")
    CDN_HTML_BASE  = st.secrets.get("CDN_HTML_BASE", "https://stories.suvichaar.org/")
except Exception:
    st.error("Missing secrets. Please set required Azure and AWS keys in .streamlit/secrets.toml")
    st.stop()

# ---------------------------
# Clients
# ---------------------------
di_client = DocumentIntelligenceClient(
    endpoint=AZURE_DI_ENDPOINT,
    credential=AzureKeyCredential(AZURE_API_KEY)
)

gpt_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
)

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    )

# ---------------------------
# Prompts
# ---------------------------
SYSTEM_PROMPT_OCR_TO_QA = """
You receive OCR text that already contains multiple-choice questions in Hindi or English.
Each question has options (A)-(D), a single correct answer, and ideally an explanation.

Return a JSON object:
{
  "questions": [
    {
      "question": "...",
      "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
      "correct_option": "A" | "B" | "C" | "D",
      "explanation": "..."
    }
  ]
}

- If explanations are missing, write a concise 1–2 sentence explanation grounded in the text.
- Preserve the original language (Hindi stays Hindi, English stays English).
- Ensure valid JSON only.
"""

SYSTEM_PROMPT_NOTES_TO_QA_BASE = """
You are given raw study notes text (could be Hindi or English). Generate exactly FIVE high-quality
multiple-choice questions (MCQs) that are strictly grounded in these notes.

For each question:
- Provide four options labeled A–D.
- Ensure exactly one correct option.
- Add a 1–2 sentence explanation that justifies the correct answer using the notes.

Respond ONLY with valid JSON in this schema:
{
  "questions": [
    {
      "question": "...",
      "options": {"A":"...", "B":"...", "C":"...", "D":"..."},
      "correct_option": "A" | "B" | "C" | "D",
      "explanation": "..."
    }
  ]
}

Language: Use the same language as the notes (auto-detect). Keep questions concise and unambiguous.
"""

SYSTEM_PROMPT_QA_TO_PLACEHOLDERS = """
You are given a JSON object with key "questions": a list where each item has:
- question (string)
- options: {"A":..., "B":..., "C":..., "D":...}
- correct_option (A/B/C/D)
- explanation (string)

Produce a single flat JSON object with EXACTLY these keys (sensible short defaults if missing).
Use the SAME language as the input questions (auto-detect; Hindi → Hindi, English → English).

pagetitle, storytitle, typeofquiz, potraitcoverurl,
s1title1, s1text1,

s2questionHeading, s2question1,
s2option1, s2option1attr, s2option2, s2option2attr,
s2option3, s2option3attr, s2option4, s2option4attr,
s2attachment1,

s3questionHeading, s3question1,
s3option1, s3option1attr, s3option2, s3option2attr,
s3option3, s3option3attr, s3option4, s3option4attr,
s3attachment1,

s4questionHeading, s4question1,
s4option1, s4option1attr, s4option2, s4option2attr,
s4option3, s4option3attr, s4option4, s4option4attr,
s4attachment1,

s5questionHeading, s5question1,
s5option1, s5option1attr, s5option2, s5option2attr,
s5option3, s5option3attr, s5option4, s5option4attr,
s5attachment1,

s6questionHeading, s6question1,
s6option1, s6option1attr, s6option2, s6option2attr,
s6option3, s6option3attr, s6option4, s6option4attr,
s6attachment1,

results_bg_image, results_prompt_text, results1_text, results2_text, results3_text

Mapping rules:
- We only need FIVE questions in the AMP template: map questions[0] → s2*, questions[1] → s3*, … questions[4] → s6*.
- sNquestion1 ← questions[N-2].question  (N=2..6)
- sNoption1..4 ← options A..D text
- For the correct option, set sNoptionKattr to the string "correct"; for others set "".
- sNattachment1 ← explanation for that question
- sNquestionHeading ← "Question {N-1}" (or the language-appropriate equivalent; e.g., Hindi: "प्रश्न {N-1}")
- pagetitle/storytitle: create short, relevant titles from the overall content.
- typeofquiz: "Educational" (or "शैक्षिक" in Hindi) if unknown.
- s1title1: a 2–5 word intro title; s1text1: 1–2 sentence intro.
- results_*: short friendly strings in the same language. results_bg_image: "" if none.

Return only the JSON object.
""".strip()

# ---------------------------
# Helpers
# ---------------------------
def clean_model_json(txt: str) -> str:
    """Remove code fences if model returns ```json ... ``` or ``` ... ```."""
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", txt, flags=re.DOTALL)
    if fenced:
        return fenced[0].strip()
    return txt.strip()

def _variation_token(n: int = 6) -> str:
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=n))

def _variation_style() -> str:
    return random.choice([
        "exam-focused neutral tone",
        "student-friendly tone",
        "teacher’s concise quiz tone",
        "competitive test prep tone",
        "flashcard-style phrasing"
    ])

def ocr_azure_read(bytes_blob: bytes) -> dict:
    """
    Call Azure DI prebuilt-read and return a dict:
    {
      "text": "<full text>",
      "pages": [
         {"pageNumber": 1, "text": "..."},
         ...
      ]
    }
    Works for images and PDFs.
    """
    poller = di_client.begin_analyze_document(
        model_id="prebuilt-read",
        body=bytes_blob
    )
    result = poller.result()

    pages_out = []
    full_lines = []

    # Prefer paragraphs with page grouping if available
    # Fallback to per-page lines
    if getattr(result, "pages", None):
        for p in result.pages:
            page_lines = []
            for line in getattr(p, "lines", []) or []:
                if getattr(line, "content", None):
                    page_lines.append(line.content)
            page_text = "\n".join(page_lines).strip()
            pages_out.append({"pageNumber": getattr(p, "page_number", len(pages_out) + 1), "text": page_text})
            if page_text:
                full_lines.append(f"[[PAGE {p.page_number}]]\n{page_text}")
    elif getattr(result, "paragraphs", None):
        # single lump with no page info—still return text
        allp = "\n".join([pp.content for pp in result.paragraphs if getattr(pp, "content", None)]).strip()
        pages_out.append({"pageNumber": 1, "text": allp})
        if allp:
            full_lines.append(f"[[PAGE 1]]\n{allp}")
    else:
        # last resort: use result.content if set
        raw = getattr(result, "content", "") or ""
        pages_out.append({"pageNumber": 1, "text": raw.strip()})
        if raw:
            full_lines.append(f"[[PAGE 1]]\n{raw.strip()}")

    return {
        "text": "\n\n".join(full_lines).strip(),
        "pages": pages_out
    }

def ocr_extract_many_mixed(files) -> str:
    """
    Accepts a mixed list of Streamlit UploadedFile (images and PDFs).
    Returns concatenated text with [[PAGE n]] markers per file/page.
    """
    chunks = []
    for fidx, f in enumerate(files, start=1):
        b = f.getvalue()
        out = ocr_azure_read(b)
        if out["text"]:
            # add file header so multi-file inputs are traceable
            chunks.append(f"[[FILE {fidx}: {f.name}]]\n{out['text']}")
    return "\n\n".join(chunks).strip()

def gpt_ocr_text_to_questions(raw_text: str, diversify: bool = False) -> dict:
    """
    Convert OCR text that already contains questions into structured JSON.
    If diversify=True, add a random token/style + slightly higher temperature.
    """
    temp = 0.0 if not diversify else 0.6
    sys = SYSTEM_PROMPT_OCR_TO_QA
    if diversify:
        sys = sys + f"\n\nVARIATION_TOKEN: { _variation_token() }\nStyle hint: { _variation_style() }"
    resp = gpt_client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        temperature=temp,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": raw_text}
        ],
    )
    content = clean_model_json(resp.choices[0].message.content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

def gpt_notes_to_questions(notes_text: str, diversify: bool = False) -> dict:
    """
    Generate 5 MCQs from notes. If diversify=True, we inject a random token/style
    and increase temperature so repeated inputs produce varied outputs.
    """
    temp = 0.0 if not diversify else 0.7
    sys = SYSTEM_PROMPT_NOTES_TO_QA_BASE
    if diversify:
        sys = sys + f"\n\nVARIATION_TOKEN: { _variation_token() }\nStyle hint: { _variation_style() }"
    resp = gpt_client.chat.completions.create(
        model=GPT_DEPLOYMENT,
        temperature=temp,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": notes_text}
        ],
    )
    content = clean_model_json(resp.choices[0].message.content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

def gpt_questions_to_placeholders(questions_data: dict) -> dict:
    """Map structured questions JSON into flat placeholder JSON for AMP template."""
    q = questions_data.get("questions", [])
    if len(q) > 5:
        questions_data = {"questions": q[:5]}

    # Prefer the modern call; keep backward-compatible fallback
    if hasattr(gpt_client, "chat_completions"):
        resp = gpt_client.chat_completions.create(
            model=GPT_DEPLOYMENT,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_QA_TO_PLACEHOLDERS},
                {"role": "user", "content": json.dumps(questions_data, ensure_ascii=False)}
            ],
        )
        choice_msg = getattr(resp.choices[0], "message", None)
        content = clean_model_json(choice_msg.content if choice_msg else "")
    else:
        resp = gpt_client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_QA_TO_PLACEHOLDERS},
                {"role": "user", "content": json.dumps(questions_data, ensure_ascii=False)}
            ],
        )
        content = clean_model_json(resp.choices[0].message.content)

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))

def build_attr_value(key: str, val: str) -> str:
    """s2option3attr = 'correct' → 'option-3-correct'; else passthrough/empty."""
    if not key.endswith("attr") or not val:
        return ""
    m = re.match(r"s(\d+)option(\d)attr$", key)
    if m and val.strip().lower() == "correct":
        return f"option-{m.group(2)}-correct"
    return val

def fill_template(template: str, data: dict) -> str:
    """Replace {{key}} and {{key|safe}} using placeholder data, handling *attr keys specially."""
    rendered = {}
    for k, v in data.items():
        if k.endswith("attr"):
            rendered[k] = build_attr_value(k, str(v))
        else:
            rendered[k] = "" if v is None else str(v)
    html = template
    for k, v in rendered.items():
        html = html.replace(f"{{{{{k}}}}}", v)
        html = html.replace(f"{{{{{k}|safe}}}}", v)
    return html

def upload_html_to_s3(html_text: str, filename: str):
    """
    Upload HTML to S3 and return (s3_key, cdn_url).
    No ACL is set (works with CloudFront + Origin Access).
    """
    if not filename.lower().endswith(".html"):
        filename = f"{filename}.html"

    s3_key = f"{HTML_S3_PREFIX.strip('/')}/{filename}" if HTML_S3_PREFIX else filename

    s3 = get_s3_client()
    s3.put_object(
        Bucket=AWS_BUCKET,
        Key=s3_key,
        Body=html_text.encode("utf-8"),
        ContentType="text/html; charset=utf-8",
        CacheControl="public, max-age=300",
        ContentDisposition=f'inline; filename="{filename}"',
    )

    cdn_url = f"{CDN_HTML_BASE.rstrip('/')}/{s3_key}"
    return s3_key, cdn_url

# ---------------------------
# 🧩 Builder UI
# ---------------------------
tab_all, = st.tabs(["All-in-one Builder"])

with tab_all:
    st.subheader("Build final AMP HTML from image(s)/PDF(s) or structured JSON")
    st.caption("Pick an input source, upload AMP HTML template, and download the final HTML with a timestamped filename.")

    mode = st.radio(
        "Choose input",
        [
            "Notes image(s)/PDF(s) (OCR → generate quiz JSON)",
            "Quiz image(s)/PDF(s) (OCR → parse existing MCQs)",
            "Structured JSON (skip OCR)"
        ],
        horizontal=False
    )

    up_tpl = st.file_uploader("📎 Upload AMP HTML template (.html)", type=["html", "htm"], key="tpl")
    show_debug = st.toggle("Show OCR / JSON previews", value=False)
    diversify = st.toggle("Always vary output (even for identical inputs)", value=True,
                          help="Adds a random style token and higher temperature so repeated inputs yield different phrasing/questions.")

    questions_data = None

    if mode == "Notes image(s)/PDF(s) (OCR → generate quiz JSON)":
        up_files = st.file_uploader(
            "📎 Upload notes files — images (JPG/PNG/WebP/TIFF) and/or PDFs (multiple allowed)",
            type=["jpg", "jpeg", "png", "webp", "tiff", "pdf"],
            accept_multiple_files=True,
            key="notes_files"
        )
        if up_files:
            try:
                if show_debug:
                    for i, f in enumerate(up_files, start=1):
                        if f.type.startswith("image/"):
                            try:
                                st.image(Image.open(io.BytesIO(f.getvalue())).convert("RGB"),
                                         caption=f"Notes file {i}: {f.name}", use_container_width=True)
                            except Exception:
                                pass
                        else:
                            st.write(f"📄 Notes file {i}: {f.name} (PDF)")

                with st.spinner("🔍 OCR (Azure Document Intelligence) on all inputs…"):
                    notes_text = ocr_extract_many_mixed(up_files)

                if not notes_text.strip():
                    st.error("OCR returned empty text. Try clearer scans or different files.")
                    st.stop()

                if show_debug:
                    with st.expander("📄 OCR Notes Text"):
                        st.text(notes_text[:8000] if len(notes_text) > 8000 else notes_text)

                with st.spinner("📝 Generating 5 MCQs from notes…"):
                    questions_data = gpt_notes_to_questions(notes_text, diversify=diversify)

                if show_debug:
                    with st.expander("🧱 Generated Questions JSON"):
                        st.code(json.dumps(questions_data, ensure_ascii=False, indent=2)[:12000], language="json")

            except Exception as e:
                st.error(f"Failed to process notes → quiz JSON: {e}")
                st.stop()

    elif mode == "Quiz image(s)/PDF(s) (OCR → parse existing MCQs)":
        up_files = st.file_uploader(
            "📎 Upload quiz files — images (JPG/PNG) and/or PDFs (multiple allowed)",
            type=["jpg", "jpeg", "png", "pdf"],
            accept_multiple_files=True,
            key="quiz_files"
        )
        if up_files:
            try:
                if show_debug:
                    for i, f in enumerate(up_files, start=1):
                        if f.type.startswith("image/"):
                            try:
                                st.image(Image.open(io.BytesIO(f.getvalue())).convert("RGB"),
                                         caption=f"Quiz file {i}: {f.name}", use_container_width=True)
                            except Exception:
                                pass
                        else:
                            st.write(f"📄 Quiz file {i}: {f.name} (PDF)")

                with st.spinner("🔍 OCR (Azure Document Intelligence)…"):
                    raw_text = ocr_extract_many_mixed(up_files)

                if not raw_text.strip():
                    st.error("OCR returned empty text. Try clearer scans or different files.")
                    st.stop()

                if show_debug:
                    with st.expander("📄 OCR Text"):
                        st.text(raw_text[:8000] if len(raw_text) > 8000 else raw_text)

                with st.spinner("🤖 Parsing OCR into questions JSON…"):
                    questions_data = gpt_ocr_text_to_questions(raw_text, diversify=diversify)

                if show_debug:
                    with st.expander("🧱 Structured Questions JSON"):
                        st.code(json.dumps(questions_data, ensure_ascii=False, indent=2)[:12000], language="json")

            except Exception as e:
                st.error(f"Failed to process quiz files → JSON: {e}")
                st.stop()

    else:  # Structured JSON
        up_json = st.file_uploader("📎 Upload structured questions JSON", type=["json"], key="json")
        if up_json:
            try:
                questions_data = json.loads(up_json.getvalue().decode("utf-8"))
                if show_debug:
                    with st.expander("🧱 Structured Questions JSON"):
                        st.code(json.dumps(questions_data, ensure_ascii=False, indent=2)[:12000], language="json")
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                st.stop()

    build = st.button("🛠️ Build final HTML", disabled=not (questions_data and up_tpl))

    if build and questions_data and up_tpl:
        try:
            # → placeholders
            with st.spinner("🧩 Generating placeholders…"):
                placeholders = gpt_questions_to_placeholders(questions_data)
                if show_debug:
                    with st.expander("🧩 Placeholder JSON"):
                        st.code(json.dumps(placeholders, ensure_ascii=False, indent=2)[:12000], language="json")

            # read template
            template_html = up_tpl.getvalue().decode("utf-8")

            # merge
            final_html = fill_template(template_html, placeholders)

            # save timestamped file locally
            ts_name = f"final_quiz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            Path(ts_name).write_text(final_html, encoding="utf-8")

            # upload to S3 (NO ACL)
            with st.spinner("☁️ Uploading to S3…"):
                s3_key, cdn_url = upload_html_to_s3(final_html, ts_name)

            st.success(f"✅ Final HTML generated and uploaded to S3: s3://{AWS_BUCKET}/{s3_key}")
            st.markdown(f"**CDN URL:** {cdn_url}")

            with st.expander("🔍 HTML Preview (source)"):
                st.code(final_html[:120000], language="html")

            st.download_button(
                "⬇️ Download final HTML",
                data=final_html.encode("utf-8"),
                file_name=ts_name,
                mime="text/html"
            )

            # 👀 Live HTML Viewer (inline)
            st.markdown("### 👀 Live HTML Preview")
            h = st.slider("Preview height (px)", min_value=400, max_value=1600, value=900, step=50)
            full_width = st.checkbox("Force full viewport width (100vw)", value=True,
                                     help="Overrides container width so the preview stretches edge-to-edge.")
            style = f"width: {'100vw' if full_width else '100%'}; height: {h}px; border: 0; margin: 0; padding: 0;"
            st_html(final_html, height=h, scrolling=True) if not full_width else st_html(
                f'<div style="position:relative;left:50%;right:50%;margin-left:-50vw;margin-right:-50vw;{style}">{final_html}</div>',
                height=h,
                scrolling=True
            )

            st.info("AMP pages may not fully render inside Streamlit due to CSP/sandbox. For a faithful view, download and open in a real browser or deploy to a server.")
        except Exception as e:
            st.error(f"Build failed: {e}")
    elif not (questions_data and up_tpl):
        st.info("Upload an input (notes/quiz files or JSON) **and** a template to enable the Build button.")
