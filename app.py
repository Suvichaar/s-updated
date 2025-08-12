import os
import io
import re
import json
import time
import base64
import requests
import boto3
import zipfile
from io import BytesIO
from datetime import datetime
from pathlib import Path
from PIL import Image
import streamlit as st
from streamlit.components.v1 import html as st_html  # <-- for live HTML preview

# --- Azure Doc Intelligence SDK ---
try:
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.core.credentials import AzureKeyCredential
except Exception:
    DocumentIntelligenceClient = None  # we'll error nicely later
    AzureKeyCredential = None

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Suvichaar Builder",
    page_icon="🧠",
    layout="centered"
)
st.title("🧠 Suvichaar Builder")
st.caption("OCR (Doc Intelligence) → GPT JSON → DALL·E → S3/CDN → (optional) SEO/TTS → Fill HTML templates → Upload & Verify")

# ---------------------------
# Secrets / Config
# ---------------------------
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except Exception:
        return default

# Azure OpenAI (GPT)
AZURE_API_KEY     = get_secret("AZURE_API_KEY")
AZURE_ENDPOINT    = get_secret("AZURE_ENDPOINT")  # https://<resource>.openai.azure.com
AZURE_DEPLOYMENT  = get_secret("AZURE_DEPLOYMENT", "gpt-4o")  # your *deployment* name
AZURE_API_VERSION = get_secret("AZURE_API_VERSION", "2024-08-01-preview")

# Azure Document Intelligence (OCR)
AZURE_DI_ENDPOINT = get_secret("AZURE_DI_ENDPOINT")  # https://<your-di>.cognitiveservices.azure.com/
AZURE_DI_KEY      = get_secret("AZURE_DI_KEY")

# Azure DALL·E (Images)
DALE_ENDPOINT     = get_secret("DALE_ENDPOINT")  # e.g. https://.../openai/deployments/dall-e-3/images/generations?api-version=2024-02-01
DAALE_KEY         = get_secret("DAALE_KEY")

# AWS S3
AWS_ACCESS_KEY        = get_secret("AWS_ACCESS_KEY")
AWS_SECRET_KEY        = get_secret("AWS_SECRET_KEY")
AWS_SESSION_TOKEN     = get_secret("AWS_SESSION_TOKEN")  # optional (for temporary creds)
AWS_REGION            = get_secret("AWS_REGION", "ap-south-1")
AWS_BUCKET            = get_secret("AWS_BUCKET", "suvichaarapp")  # default to suvichaarapp
S3_PREFIX             = get_secret("S3_PREFIX", "media")          # used for images/audio

# ---- Hard-lock HTML/JSON at bucket ROOT + root CDN base (no /webstory-html) ----
HTML_S3_PREFIX = ""  # bucket root
CDN_HTML_BASE  = "https://stories.suvichaar.org/"

# CDN image handler prefix (base64-encoded template)
CDN_PREFIX_MEDIA  = get_secret("CDN_PREFIX_MEDIA", "https://media.suvichaar.org/")

# Fallback image
DEFAULT_ERROR_IMAGE = get_secret("DEFAULT_ERROR_IMAGE", "https://media.suvichaar.org/default-error.jpg")

# Azure Speech (TTS)
AZURE_SPEECH_KEY     = get_secret("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION  = get_secret("AZURE_SPEECH_REGION", "eastus")
VOICE_NAME_DEFAULT   = get_secret("VOICE_NAME", "hi-IN-AaravNeural")

# CDN base for audio (CloudFront etc.)
CDN_BASE             = get_secret("CDN_BASE", "https://cdn.suvichaar.org/")

# Sanity checks (warn if missing)
missing_core = []
for k in [
    "AZURE_API_KEY", "AZURE_ENDPOINT", "AZURE_DEPLOYMENT",
    "AZURE_DI_ENDPOINT", "AZURE_DI_KEY",
    "DALE_ENDPOINT", "DAALE_KEY",
    "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "AWS_BUCKET"
]:
    if not get_secret(k, None):
        missing_core.append(k)
if missing_core:
    st.warning("Add these secrets in `.streamlit/secrets.toml`: " + ", ".join(missing_core))

# ---------------------------
# AWS helpers (robust client + verified uploads)
# ---------------------------
@st.cache_resource(show_spinner=False)
def get_s3_client():
    kwargs = dict(
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY,
        region_name=AWS_REGION,
    )
    if AWS_SESSION_TOKEN:
        kwargs["aws_session_token"] = AWS_SESSION_TOKEN
    return boto3.client("s3", **kwargs)


def s3_put_text_file(bucket: str, key: str, body: bytes, content_type: str, cache_control: str = "public, max-age=31536000, immutable"):
    """Upload a small text file and verify it exists by HEADing it back.
    Returns a dict: {"ok": bool, "etag": str|None, "key": str, "len": int, "error": str|None}
    """
    s3 = get_s3_client()
    put_args = {
        "Bucket": bucket,
        "Key": key,
        "Body": body,
        "ContentType": content_type,
        "CacheControl": cache_control,
    }

    try:
        s3.put_object(**put_args)
    except Exception as e:
        return {"ok": False, "etag": None, "key": key, "len": len(body), "error": f"put_object failed: {e}"}

    # Verify via HEAD
    try:
        head = s3.head_object(Bucket=bucket, Key=key)
        etag = head.get("ETag", "").strip('"')
        cl = int(head.get("ContentLength", 0))
        ok = cl == len(body)
        return {"ok": ok, "etag": etag, "key": key, "len": cl, "error": None if ok else f"size mismatch {cl}!={len(body)}"}
    except Exception as e:
        return {"ok": False, "etag": None, "key": key, "len": 0, "error": f"head_object failed: {e}"}

# ---------------------------
# Other helpers
# ---------------------------
def build_resized_cdn_url(bucket: str, key_path: str, width: int, height: int) -> str:
    """Return base64-encoded template URL for your Serverless Image Handler."""
    template = {
        "bucket": bucket,
        "key": key_path,
        "edits": {"resize": {"width": width, "height": height, "fit": "cover"}}
    }
    encoded = base64.urlsafe_b64encode(json.dumps(template).encode()).decode()
    return f"{CDN_PREFIX_MEDIA}{encoded}"

SAFE_FALLBACK = (
    "A joyful, abstract geometric illustration symbolizing unity and learning — "
    "soft shapes, harmonious gradients, friendly silhouettes, "
    "no text, no logos, no brands, no real persons, family-friendly, "
    "flat vector style, bright colors."
)

def robust_parse_model_json(raw_reply: str):
    """Parse model reply into a dict or return None."""
    parsed = None
    try:
        parsed = json.loads(raw_reply)
    except Exception:
        m = re.search(r"\{[\s\S]*\}", raw_reply)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                parsed = None
    return parsed if isinstance(parsed, dict) else None


def repair_json_with_model(raw_reply: str, chat_url: str, headers: dict):
    """Ask the model to fix its own output into valid JSON per schema; returns dict or None."""
    schema_hint = """
Keys (English), values in detected language. If any field is missing, use an empty string:
{
  "language": "hi|en|bn|ta|te|mr|gu|kn|pa|en-IN|...",
  "storytitle": "...",
  "s1paragraph1": "...",
  "s2paragraph1": "...",
  "s3paragraph1": "...",
  "s4paragraph1": "...",
  "s5paragraph1": "...",
  "s6paragraph1": "...",
  "s1alt1": "...",
  "s2alt1": "...",
  "s3alt1": "...",
  "s4alt1": "...",
  "s5alt1": "...",
  "s6alt1": "..."
}
Return ONLY valid JSON, no code fences, no commentary.
"""
    payload_fix = {
        "messages": [
            {"role": "system",
             "content": "You are a strict JSON formatter. You output ONLY valid minified JSON. No prose."},
            {"role": "user",
             "content": f"This text was intended to be JSON but is invalid/truncated. "
                        f"Repair it into valid JSON that matches the schema.\n\nSchema:\n{schema_hint}\n\nText:\n{raw_reply}"}
        ],
        "temperature": 0.0,
        "max_tokens": 1400,
        "response_format": {"type": "json_object"}
    }
    try:
        res = requests.post(chat_url, headers=headers, json=payload_fix, timeout=150)
        if res.status_code != 200:
            return None
        fixed = res.json()["choices"][0]["message"]["content"]
        return robust_parse_model_json(fixed)
    except Exception:
        return None


def call_azure_chat(messages, *, temperature=0.2, max_tokens=1800, force_json=True):
    """Call Azure Chat (JSON mode default). Returns (ok, content_or_err)."""
    chat_headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    chat_url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
    params = {"api-version": AZURE_API_VERSION}

    body = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if force_json:
        body["response_format"] = {"type": "json_object"}

    try:
        res = requests.post(chat_url, headers=chat_headers, params=params, json=body, timeout=150)
    except Exception as e:
        return False, f"Azure request failed: {e}"

    if res.status_code == 200:
        return True, res.json()["choices"][0]["message"]["content"]

    # If JSON mode fails, retry without it
    if force_json:
        body.pop("response_format", None)
        try:
            res2 = requests.post(chat_url, headers=chat_headers, params=params, json=body, timeout=150)
            if res2.status_code == 200:
                return True, res2.json()["choices"][0]["message"]["content"]
            return False, f"Azure Chat error: {res2.status_code} — {res2.text[:300]}"
        except Exception as e:
            return False, f"Azure retry failed: {e}"

    return False, f"Azure Chat error: {res.status_code} — {res.text[:300]}"

# ---------- Language auto-detect (Hindi vs English) ----------
def detect_hi_or_en(text: str) -> str:
    """Return 'hi' if text is mostly Devanagari, else 'en'."""
    if not text:
        return "en"
    devanagari = sum(0x0900 <= ord(c) <= 0x097F for c in text)
    latin = sum(('A' <= c <= 'Z') or ('a' <= c <= 'z') for c in text)
    total_letters = devanagari + latin
    if total_letters == 0:
        return "hi" if devanagari > 0 else "en"
    return "hi" if (devanagari / total_letters) >= 0.25 else "en"

# ---------- Azure Document Intelligence (OCR) ----------
def ocr_extract_bytes(image_bytes: bytes):
    """
    Uses Azure Document Intelligence 'prebuilt-read' to extract text.
    Returns (ok: bool, text_or_error: str).
    """
    if DocumentIntelligenceClient is None or AzureKeyCredential is None:
        return False, ("azure-ai-documentintelligence is not installed. "
                       "Add it to requirements.txt and set AZURE_DI_ENDPOINT / AZURE_DI_KEY.")

    if not (AZURE_DI_ENDPOINT and AZURE_DI_KEY):
        return False, "Missing AZURE_DI_ENDPOINT / AZURE_DI_KEY."

    try:
        client = DocumentIntelligenceClient(
            endpoint=AZURE_DI_ENDPOINT.rstrip("/"),
            credential=AzureKeyCredential(AZURE_DI_KEY),
        )
        poller = client.begin_analyze_document("prebuilt-read", body=image_bytes)
        doc = poller.result()
        paragraphs = getattr(doc, "paragraphs", None) or []
        if paragraphs:
            text = "\n".join(p.content for p in paragraphs if getattr(p, "content", None))
            text = (text or "").strip()
            if text:
                return True, text
        # fallback to raw content if available
        content = getattr(doc, "content", "") or ""
        if content.strip():
            return True, content.strip()
        return False, "Document Intelligence returned no text."
    except Exception as e:
        return False, f"Doc Intelligence error: {e}"

# -------- Image generation + S3 upload --------
def sanitize_prompt(chat_url: str, headers: dict, original_prompt: str) -> str:
    """Rewrite any risky prompt into a safe, positive, family-friendly version using Azure Chat."""
    sanitize_payload = {
        "messages": [
            {"role": "system", "content": (
                "Rewrite image prompts to be safe, positive, inclusive, and family-friendly. "
                "Remove any hate/harassment/violence/adult/illegal/extremist content, slogans, logos, "
                "or real-person likenesses. Keep the core educational idea and flat vector art style. "
                "Return ONLY the rewritten prompt text."
            )},
            {"role": "user", "content": f"Original prompt:\n{original_prompt}\n\nRewritten safe prompt:"}
        ],
        "temperature": 0.2,
        "max_tokens": 300
    }
    try:
        sr = requests.post(chat_url, headers=headers, json=sanitize_payload, timeout=60)
        if sr.status_code == 200:
            return sr.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.info(f"Sanitizer call failed; using local guards: {e}")

    return (original_prompt +
            "\nFlat vector illustration, bright colors, no text, no logos, no brands, "
            "no real persons, family-friendly, inclusive, peaceful.")


def generate_and_upload_images(result_json: dict) -> dict:
    """Generate DALL·E images, upload originals to S3, return CDN resized URLs in JSON."""
    if not all([DALE_ENDPOINT, DAALE_KEY, AWS_ACCESS_KEY, AWS_SECRET_KEY, AWS_BUCKET]):
        st.error("Missing DALL·E and/or AWS S3 secrets.")
        return {**result_json}

    s3 = get_s3_client()

    slug = (
        (result_json.get("storytitle") or "story")
        .lower()
        .replace(" ", "-")
        .replace(":", "")
        .replace(".", "")
    )
    out = {k: result_json[k] for k in result_json}
    first_slide_key = None

    headers_dalle = {"Content-Type": "application/json", "api-key": DAALE_KEY}
    progress = st.progress(0, text="Generating images…")

    for i in range(1, 7):
        raw_prompt = result_json.get(f"s{i}alt1", "") or ""
        chat_headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
        chat_url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
        safe_prompt = sanitize_prompt(chat_url, chat_headers, raw_prompt)

        payload = {"prompt": safe_prompt, "n": 1, "size": "1024x1024"}
        image_url = None

        for attempt in range(3):
            r = requests.post(DALE_ENDPOINT, headers=headers_dalle, json=payload, timeout=120)
            if r.status_code == 200:
                try:
                    image_url = r.json()["data"][0]["url"]
                    break
                except Exception as e:
                    st.info(f"Slide {i}: unexpected DALL·E response format: {e}")
                    break
            elif r.status_code in (400, 403):
                st.info(f"Slide {i}: DALL·E blocked, retrying with fallback.")
                payload = {"prompt": SAFE_FALLBACK, "n": 1, "size": "1024x1024"}
                continue
            elif r.status_code == 429:
                st.info(f"Slide {i}: rate-limited, waiting 10s…")
                time.sleep(10)
            else:
                st.info(f"Slide {i}: DALL·E error {r.status_code} — {r.text[:200]}")
                break

        if image_url:
            try:
                img_data = requests.get(image_url, timeout=120).content
                buffer = BytesIO(img_data)  # upload original; no local resize
                key = f"{S3_PREFIX.rstrip('/')}/{slug}/slide{i}.jpg"
                extra_args = {"ContentType": "image/jpeg"}
                # No ACL
                s3.upload_fileobj(buffer, AWS_BUCKET, key, ExtraArgs=extra_args)
                if i == 1:
                    first_slide_key = key

                # build CDN resized URL (720x1200)
                final_url = build_resized_cdn_url(AWS_BUCKET, key, 720, 1200)
                out[f"s{i}image1"] = final_url
            except Exception as e:
                st.info(f"Slide {i}: upload/CDN URL build failed → {e}")
                out[f"s{i}image1"] = DEFAULT_ERROR_IMAGE
        else:
            out[f"s{i}image1"] = DEFAULT_ERROR_IMAGE

        progress.progress(i/6.0, text=f"Generating images… ({i}/6)")

    progress.empty()

    # portrait cover from slide 1 via CDN (640x853) — keep both spellings for template compatibility
    try:
        if first_slide_key:
            cover = build_resized_cdn_url(AWS_BUCKET, first_slide_key, 640, 853)
            out["portraitcoverurl"] = cover
            out["potraitcoverurl"] = cover  # backward-compat
            out["potraightcoverurl"] = cover
        else:
            out["portraitcoverurl"] = DEFAULT_ERROR_IMAGE
            out["potraitcoverurl"] = DEFAULT_ERROR_IMAGE
            out["potraightcoverurl"] = DEFAULT_ERROR_IMAGE
    except Exception as e:
        st.info(f"Portrait cover URL build failed: {e}")
        out["portraitcoverurl"] = DEFAULT_ERROR_IMAGE
        out["potraitcoverurl"] = DEFAULT_ERROR_IMAGE
        out["potraightcoverurl"] = DEFAULT_ERROR_IMAGE

    return out


def generate_seo_metadata(chat_url: str, headers: dict, result_json: dict, lang_code: str):
    """Ask the model for SEO metadata in the detected language."""
    lang_code = (lang_code or "").strip() or "auto"
    seo_prompt = f"""
Generate SEO metadata for a web story. Write ALL outputs in this language: {lang_code}

Title: {result_json.get("storytitle","")}
Slides:
- {result_json.get("s1paragraph1","")}
- {result_json.get("s2paragraph1","")}
- {result_json.get("s3paragraph1","")}
- {result_json.get("s4paragraph1","")}
- {result_json.get("s5paragraph1","")}
- {result_json.get("s6paragraph1","")}

Respond strictly in this JSON format:
{{
  "metadescription": "...",
  "metakeywords": "keyword1, keyword2, ..."
}}
"""

    payload_seo = {
        "messages": [
            {"role": "system", "content": "You are an expert SEO assistant. Answer ONLY with valid JSON."},
            {"role": "user", "content": seo_prompt}
        ],
        "temperature": 0.5,
        "max_tokens": 400,
        "response_format": {"type": "json_object"}
    }
    try:
        r = requests.post(chat_url, headers=headers, json=payload_seo, timeout=90)
        if r.status_code != 200:
            return "Explore this insightful story.", "web story, inspiration"
        content = r.json()["choices"][0]["message"]["content"]
        data = robust_parse_model_json(content) or {}
        return data.get("metadescription", "Explore this insightful story."), \
               data.get("metakeywords", "web story, inspiration")
    except Exception:
        return "Explore this insightful story.", "web story, inspiration"


def pick_voice_for_language(lang_code: str, default_voice: str) -> str:
    """Map detected language → Azure voice name."""
    if not lang_code:
        return default_voice
    l = lang_code.lower()
    if l.startswith("hi"):
        return "hi-IN-AaravNeural"
    if l.startswith("en-in"):
        return "en-IN-NeerjaNeural"
    if l.startswith("en"):
        return "en-IN-AaravNeural"
    if l.startswith("bn"):
        return "bn-IN-BashkarNeural"
    if l.startswith("ta"):
        return "ta-IN-PallaviNeural"
    if l.startswith("te"):
        return "te-IN-ShrutiNeural"
    if l.startswith("mr"):
        return "mr-IN-AarohiNeural"
    if l.startswith("gu"):
        return "gu-IN-DhwaniNeural"
    if l.startswith("kn"):
        return "kn-IN-SapnaNeural"
    if l.startswith("pa"):
        return "pa-IN-GeetikaNeural"
    return default_voice


def fill_template_strict(template: str, data: dict):
    """Replace {{key}} with string(value). Also return placeholders detected (for missing-report)."""
    placeholders = set(re.findall(r"\{\{\s*([a-zA-Z0-9_\-]+)\s*\}\}", template))
    for k, v in data.items():
        template = template.replace(f"{{{{{k}}}}}", str(v))
    return template, placeholders

# ---- helpers for root uploads/URLs (hard-locked) ----
def _s3_key(name: str) -> str:
    """Always just filename (bucket root)."""
    return name  # HTML S3 uploads at root


def _cdn_url(name: str) -> str:
    """CDN base + filename at root."""
    return f"{CDN_HTML_BASE.rstrip('/')}/{name}"

# ---------------------------
# UI
# ---------------------------
img = st.file_uploader("Upload a notes image (JPG/PNG)", type=["jpg", "jpeg", "png"])
html_files = st.file_uploader("Upload one or more HTML templates (with {{placeholders}})", type=["html", "htm"], accept_multiple_files=True)

c1, c2, c3 = st.columns(3)
with c1:
    include_seo = st.checkbox("Generate SEO metadata", value=True)
with c2:
    include_tts = st.checkbox("Generate TTS audio", value=True)
with c3:
    add_time_fields = st.checkbox("Add time fields", value=True, help="Adds {{publishedtime}} and {{modifiedtime}} (UTC ISO).")

run = st.button("🚀 Run")

if run:
    # Basic validation
    if not img:
        st.error("Please upload a notes image.")
        st.stop()
    if not html_files:
        st.error("Please upload at least one HTML template.")
        st.stop()

    # Preview image
    try:
        raw_bytes = img.getvalue()
        if not raw_bytes:
            st.error("Uploaded image is empty.")
            st.stop()
        pil_img = Image.open(BytesIO(raw_bytes)).convert("RGB")
        st.image(pil_img, caption="Uploaded image", use_container_width=True)
    except Exception as e:
        st.error(f"Could not open image: {e}")
        st.stop()

    # -------- OCR with Azure Document Intelligence --------
    with st.spinner("Reading text with Azure Document Intelligence (prebuilt-read)…"):
        ok_ocr, raw_text = ocr_extract_bytes(raw_bytes)
        if not ok_ocr:
            st.error(raw_text)
            st.stop()

    with st.expander("🔎 OCR extracted text"):
        st.write(raw_text[:20000] if raw_text else "(empty)")

    # -------- Language auto-detect (hi/en) --------
    target_lang = detect_hi_or_en(raw_text)
    st.info(f"Target language (auto): **{target_lang}**")

    # -------- Summarize with GPT into JSON (s1..s6 + s1alt..s6alt) --------
   system_prompt = f"""
You are a multilingual teaching assistant.

INPUT:
- You will receive raw OCR text from a notes image.

MANDATORY:
- Target language = "{target_lang}" (use concise BCP-47 like "en" or "hi").
- Produce ALL text fields strictly in the Target language.

Your job:
1) Extract a short and catchy title → storytitle (Target language)
2) Summarise the content into 6 slides (s1paragraph1..s6paragraph1), each ≤ 400 characters (Target language).
3) For each paragraph (including slide 1), write a DALL·E prompt (s1alt1..s6alt1) for a 1024x1024 flat vector illustration: bright colors, clean lines, no text/captions/logos. (Prompts must not request text in the image.)

SAFETY & POSITIVITY RULES (MANDATORY):
- If the source includes hate, harassment, violence, adult content, self-harm, illegal acts, or extremist symbols, DO NOT reproduce them.
- Reinterpret into a positive, inclusive, family-friendly, educational scene (unity, learning, empathy, community, peace).
- Replace any hateful/violent symbol with abstract shapes, nature, or neutral motifs.
- Never include real people’s likeness or sensitive groups in a negative way.
- Avoid slogans, gestures, flags, trademarks, or captions. Absolutely NO TEXT in the image.

Respond strictly in this JSON format (keys in English; values in Target language):
{{
  "language": "{target_lang}",
  "storytitle": "...",
  "s1paragraph1": "...",
  "s2paragraph1": "...",
  "s3paragraph1": "...",
  "s4paragraph1": "...",
  "s5paragraph1": "...",
  "s6paragraph1": "...",
  "s1alt1": "...",
  "s2alt1": "...",
  "s3alt1": "...",
  "s4alt1": "...",
  "s5alt1": "...",
  "s6alt1": "..."
}}
"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"OCR TEXT:\n{raw_text}\n\nReturn only the JSON object described above."}
    ]

    with st.spinner("Summarizing OCR text with Azure OpenAI…"):
        ok, content = call_azure_chat(messages, temperature=0.0, max_tokens=1800, force_json=True)
        if not ok:
            st.error(content)
            st.stop()

        result = robust_parse_model_json(content)
        if not isinstance(result, dict):
            # One-shot repair
            chat_headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
            chat_url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
            fixed = repair_json_with_model(content, chat_url, chat_headers)
            if isinstance(fixed, dict):
                result = fixed

        if not isinstance(result, dict):
            st.error("Model did not return a valid JSON object.\n\nRaw reply (truncated):\n" + content[:800])
            st.stop()

    # Enforce target language in parsed result
    result["language"] = target_lang
    detected_lang = target_lang
    st.info(f"Detected/target language: **{detected_lang}**")

    # Keep OCR text too (handy for debugging or templates)
    result["ocr_text"] = raw_text

    st.success("Structured JSON created from OCR.")
    st.json(result, expanded=False)

    # -------- DALL·E images → S3 → CDN --------
    with st.spinner("Generating DALL·E images and uploading to S3…"):
        final_json = generate_and_upload_images(result)

    # -------- SEO metadata (optional) --------
    chat_headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
    chat_url = f"{AZURE_ENDPOINT.rstrip('/')}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions?api-version={AZURE_API_VERSION}"
    if include_seo:
        with st.spinner("Generating SEO metadata…"):
            meta_desc, meta_keywords = generate_seo_metadata(chat_url, chat_headers, final_json, detected_lang)
            final_json["metadescription"] = meta_desc
            final_json["metakeywords"] = meta_keywords

    # -------- Optional: TTS --------
    if include_tts:
        try:
            import azure.cognitiveservices.speech as speechsdk
        except Exception as e:
            st.error("`azure-cognitiveservices-speech` is not installed. Add it to requirements.txt.\n"
                     f"Import error: {e}")
            st.stop()

        s3 = get_s3_client()

        chosen_voice = pick_voice_for_language(detected_lang, VOICE_NAME_DEFAULT)
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
        speech_config.speech_synthesis_voice_name = chosen_voice
        st.info(f"TTS voice: **{chosen_voice}**")

        # Map paragraphs to audio fields (s1..s6 paragraphs + title as s1audio1)
        field_mapping = {
            "storytitle":    "s1audio1",
            "s2paragraph1":  "s2audio1",
            "s3paragraph1":  "s3audio1",
            "s4paragraph1":  "s4audio1",
            "s5paragraph1":  "s5audio1",
            "s6paragraph1":  "s6audio1",
        }

        created_audio = {}
        with st.spinner("Synthesizing audio and uploading to S3…"):
            for field, audio_key in field_mapping.items():
                text = final_json.get(field)
                if not text:
                    st.info(f"⚠️ Skipped (missing): {field}")
                    continue

                uuid_name = f"{os.urandom(16).hex()}.mp3"
                try:
                    audio_config = speechsdk.audio.AudioOutputConfig(filename=uuid_name)
                    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)
                    result_tts = synthesizer.speak_text_async(text).get()
                    from azure.cognitiveservices.speech import ResultReason
                    if result_tts.reason == ResultReason.SynthesizingAudioCompleted:
                        s3_key = f"{S3_PREFIX.rstrip('/')}/audio/{uuid_name}"
                        extra_args = {"ContentType": "audio/mpeg"}
                        # No ACL
                        get_s3_client().upload_file(uuid_name, AWS_BUCKET, s3_key, ExtraArgs=extra_args)
                        cdn_url = f"{CDN_BASE.rstrip('/')}/{s3_key}"
                        final_json[audio_key] = cdn_url
                        created_audio[field] = cdn_url
                        st.write(f"🎧 {field} → {cdn_url}")
                    else:
                        st.error(f"TTS failed for: {field}")
                except Exception as e:
                    st.error(f"TTS/Upload error for {field}: {e}")
                finally:
                    try:
                        os.remove(uuid_name)
                    except Exception:
                        pass

            if created_audio:
                st.json({"audio_created": created_audio}, expanded=False)

    # -------- Add time fields (optional) --------
    extra_fields = {}
    if add_time_fields:
        iso_now = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        extra_fields["publishedtime"] = iso_now
        extra_fields["modifiedtime"] = iso_now

    merged = dict(final_json)
    merged.update(extra_fields)

    # -------- Fill templates and offer downloads + S3 upload --------
    def slugify_filename(text: str) -> str:
        s = (text or "webstory").strip().lower()
        s = re.sub(r"[:/\\]+", "-", s)
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"[^a-z0-9_\-\.]", "", s)
        return s or "webstory"

    base_name = slugify_filename(merged.get("storytitle", "webstory"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    filled_items = []  # (filename, filled_html)

    if html_files:
        with st.spinner("Filling HTML templates…"):
            per_file_reports = []

            for f in html_files:
                try:
                    raw_bytes = f.read()
                    html_text = raw_bytes.decode("utf-8")
                except Exception:
                    st.error(f"Could not read {f.name} as UTF-8.")
                    continue

                filled, placeholders = fill_template_strict(html_text, merged)
                missing = sorted([p for p in placeholders if f"{{{{{p}}}}}" in html_text and p not in merged])
                if missing:
                    per_file_reports.append((f.name, missing))

                # ✅ make output filenames unique per template (prevents overwrite)
                stem = Path(f.name).stem
                out_filename = f"{stem}_{base_name}_{ts}.html"
                filled_items.append((out_filename, filled))

        if per_file_reports:
            st.warning("Some templates contain placeholders not found in JSON:")
            for name, missing in per_file_reports:
                st.write(f"• **{name}** → missing: {', '.join(missing)}")

        st.success("✅ Templates filled.")

        # Local download buttons (optional)
        json_name = f"{base_name}_{ts}.json"
        buf = io.StringIO()
        json.dump(merged, buf, ensure_ascii=False, indent=2)
        json_str = buf.getvalue()
        st.download_button(
            "⬇️ Download Final JSON",
            data=json_str.encode("utf-8"),
            file_name=json_name,
            mime="application/json"
        )

        if len(filled_items) == 1:
            single_name, single_html = filled_items[0]
            st.download_button(
                "⬇️ Download Filled HTML",
                data=single_html.encode("utf-8"),
                file_name=single_name,
                mime="text/html"
            )
        else:
            zip_buf = BytesIO()
            with zipfile.ZipFile(zip_buf, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
                for name, html in filled_items:
                    z.writestr(name, html)
            zip_buf.seek(0)
            st.download_button(
                "⬇️ Download All Filled HTML (ZIP)",
                data=zip_buf.getvalue(),
                file_name=f"{base_name}__filled_{ts}.zip",
                mime="application/zip"
            )

        # ---------- Upload JSON + HTML to S3 (bucket root) and VERIFY ----------
        st.subheader("🌐 Upload to S3 + Verification")
        uploaded_urls = []
        verifications = []

        # 1) Upload JSON (root)
        json_filename = f"{base_name}_{ts}.json"
        json_key = _s3_key(json_filename)  # root
        res_json = s3_put_text_file(
            bucket=AWS_BUCKET,
            key=json_key,
            body=json_str.encode("utf-8"),
            content_type="application/json"
        )
        if res_json["ok"]:
            json_cdn_url = _cdn_url(json_filename)
            uploaded_urls.append(("JSON", json_cdn_url))
        verifications.append({"file": json_filename, **res_json})

        # 2) Upload each HTML file (root) and verify
        for name, html in filled_items:
            html_key = _s3_key(name)
            res_html = s3_put_text_file(
                bucket=AWS_BUCKET,
                key=html_key,
                body=html.encode("utf-8"),
                content_type="text/html; charset=utf-8"
            )
            if res_html["ok"]:
                html_cdn_url = _cdn_url(name)  # https://stories.suvichaar.org/<file>.html
                uploaded_urls.append(("HTML", html_cdn_url))
            verifications.append({"file": name, **res_html})

        # Show results
        if uploaded_urls:
            for kind, url in uploaded_urls:
                st.markdown(f"- **{kind}**: {url}")
        st.json({"s3_verification": verifications}, expanded=False)

        if uploaded_urls and all(v.get("ok") for v in verifications):
            st.success("✅ Files uploaded to S3 and verified via HEAD. CDN should serve them at the URLs above (allow a short cache/propagation delay).")
        else:
            st.error("Some uploads failed or could not be verified. Check the errors above — common issues: wrong bucket name, IAM permissions (s3:PutObject and s3:HeadObject), or Public Access Block.")

        # ---------------------------
        # FINAL: Live HTML Preview (at the very end)
        # ---------------------------
        st.markdown("### 👀 Live HTML Preview")
        if not filled_items:
            st.info("No filled templates available to preview.")
        else:
            # Collect uploaded HTML URLs (if any)
            uploaded_html_urls = [u for (k, u) in uploaded_urls if k == "HTML"]

            preview_source = st.radio(
                "Choose preview source",
                options=["Local filled HTML", "Uploaded CDN URL"],
                index=0 if filled_items else 1,
                horizontal=True
            )

            if preview_source == "Local filled HTML":
                # Show a select for local HTMLs
                names = [name for name, _ in filled_items]
                choice = st.selectbox("Select local HTML to preview", names, index=0)
                chosen_html = next(html for (name, html) in filled_items if name == choice)
                # Render the HTML directly
                st_html(chosen_html, height=800, scrolling=True)
            else:
                if not uploaded_html_urls:
                    st.info("No uploaded HTML URLs found yet. Upload step might have failed.")
                else:
                    cdn_choice = st.selectbox("Select uploaded CDN URL to preview", uploaded_html_urls, index=0)
                    # Use an iframe to display remote page
                    iframe = f'''
                        <iframe src="{cdn_choice}" width="1600" height="800" style="border:0;"></iframe>
                    '''
                    st_html(iframe, height=820, scrolling=False)

        # (optional) keep your minimal code preview toggle
        show_preview = st.checkbox("Show raw HTML code of first filled template", value=False)
        if show_preview and filled_items:
            st.code(filled_items[0][1][:5000], language="html")
