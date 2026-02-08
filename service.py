#!/usr/bin/env python3
"""
Eve's Utility Services v3.0 - Practical developer utility toolkit.

Endpoints:
  POST /markdown          - Markdown to HTML conversion
  POST /json_validate     - JSON Schema validation
  POST /hash              - Cryptographic hash generation (SHA256, SHA512, MD5, HMAC)
  POST /text_analytics    - Readability scores, word frequency, text statistics
  POST /cron_explain      - Human-readable cron expression explanations
  POST /color_palette     - Color palette generation from base color
  POST /uuid_generate     - Bulk UUID generation
  POST /encode_decode     - Base64 / URL encoding and decoding
  POST /password_generate - Secure password, API key, passphrase, and PIN generation
  POST /jwt_decode        - JWT token decoding (header, payload, expiry check)
  POST /diff              - Unified diff generation with similarity stats
  POST /template_render   - Mustache/Jinja-like template rendering with filters
  POST /regex_test        - Regex pattern testing with match details
  POST /slug              - URL-friendly slug generation
  POST /csv_json          - CSV to JSON conversion
  POST /ip_info           - IP address analysis and classification
  GET  /health            - Health check
  GET  /catalog           - Service catalog with pricing

Zero external dependencies. Pure Python 3.10+ stdlib.
Complementary to Adam's services — no overlap.
"""

import hashlib
import hmac
import html
import io
import json
import math
import os
import re
import secrets
import statistics
import string
import time
import uuid
import colorsys
import base64
import difflib
import urllib.parse
import urllib.request
import urllib.error
from collections import Counter, defaultdict
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any
import threading

# ─── Configuration ───────────────────────────────────────────────────────────

PORT = int(os.environ.get("EVE_SERVICE_PORT", 8081))
COORDINATOR_URL = os.environ.get("COORDINATOR_URL", "https://singularity.wisent.ai")
INSTANCE_ID = os.environ.get("AGENT_INSTANCE_ID", "agent_1770509569_5622f0")
VERSION = "3.0.0"

PRICES = {
    "markdown": 0.02,
    "json_validate": 0.03,
    "hash": 0.02,
    "text_analytics": 0.05,
    "cron_explain": 0.02,
    "color_palette": 0.03,
    "uuid_generate": 0.01,
    "encode_decode": 0.01,
    "password_generate": 0.02,
    "jwt_decode": 0.03,
    "diff": 0.05,
    "template_render": 0.03,
    "regex_test": 0.03,
    "slug": 0.01,
    "csv_json": 0.03,
    "ip_info": 0.02,
}

stats = {
    "total_requests": 0,
    "total_revenue": 0.0,
    "errors": 0,
    "by_service": {},
    "started_at": datetime.now().isoformat(),
}


# ─── Coordinator Integration ─────────────────────────────────────────────────

def report_revenue(amount: float, service_name: str):
    """Report revenue to the coordinator."""
    url = f"{COORDINATOR_URL}/api/agents/activity"
    payload = json.dumps({
        "instance_id": INSTANCE_ID,
        "ticker": "EVE",
        "action": "REVENUE",
        "details": {
            "name": "Eve",
            "message": f"Earned ${amount:.2f} from {service_name}",
            "service": service_name,
            "amount": amount,
        },
        "revenue": amount,
        "cost": 0,
    }).encode()

    try:
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except Exception:
        pass


# ─── Service: Markdown to HTML ───────────────────────────────────────────────

def markdown_to_html(text: str) -> str:
    """Convert Markdown to HTML using pure Python."""
    lines = text.split('\n')
    result = []
    in_code_block = False
    in_list = False
    in_ordered_list = False
    code_lang = ""

    for line in lines:
        # Code blocks
        if line.strip().startswith('```'):
            if not in_code_block:
                code_lang = line.strip()[3:].strip()
                lang_attr = f' class="language-{html.escape(code_lang)}"' if code_lang else ''
                result.append(f'<pre><code{lang_attr}>')
                in_code_block = True
            else:
                result.append('</code></pre>')
                in_code_block = False
            continue

        if in_code_block:
            result.append(html.escape(line))
            continue

        # Close open lists if needed
        if in_list and not re.match(r'^[\s]*[-*+]\s', line):
            result.append('</ul>')
            in_list = False
        if in_ordered_list and not re.match(r'^\d+\.\s', line):
            result.append('</ol>')
            in_ordered_list = False

        # Headings
        heading_match = re.match(r'^(#{1,6})\s+(.*)', line)
        if heading_match:
            level = len(heading_match.group(1))
            content = inline_format(heading_match.group(2))
            result.append(f'<h{level}>{content}</h{level}>')
            continue

        # Horizontal rules
        if re.match(r'^(\-{3,}|\*{3,}|_{3,})$', line.strip()):
            result.append('<hr/>')
            continue

        # Unordered lists
        list_match = re.match(r'^[\s]*[-*+]\s+(.*)', line)
        if list_match:
            if not in_list:
                result.append('<ul>')
                in_list = True
            result.append(f'<li>{inline_format(list_match.group(1))}</li>')
            continue

        # Ordered lists
        ol_match = re.match(r'^\d+\.\s+(.*)', line)
        if ol_match:
            if not in_ordered_list:
                result.append('<ol>')
                in_ordered_list = True
            result.append(f'<li>{inline_format(ol_match.group(1))}</li>')
            continue

        # Blockquotes
        bq_match = re.match(r'^>\s?(.*)', line)
        if bq_match:
            result.append(f'<blockquote>{inline_format(bq_match.group(1))}</blockquote>')
            continue

        # Empty lines
        if line.strip() == '':
            result.append('')
            continue

        # Paragraphs
        result.append(f'<p>{inline_format(line)}</p>')

    if in_list:
        result.append('</ul>')
    if in_ordered_list:
        result.append('</ol>')
    if in_code_block:
        result.append('</code></pre>')

    return '\n'.join(result)


def inline_format(text: str) -> str:
    """Apply inline Markdown formatting."""
    # Images (before links)
    text = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<img src="\2" alt="\1"/>', text)
    # Links
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', text)
    # Bold + Italic
    text = re.sub(r'\*\*\*(.+?)\*\*\*', r'<strong><em>\1</em></strong>', text)
    # Bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    # Italic
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)
    text = re.sub(r'_(.+?)_', r'<em>\1</em>', text)
    # Strikethrough
    text = re.sub(r'~~(.+?)~~', r'<del>\1</del>', text)
    # Inline code
    text = re.sub(r'`([^`]+)`', r'<code>\1</code>', text)
    return text


# ─── Service: JSON Schema Validation ─────────────────────────────────────────

def validate_json_schema(data: Any, schema: dict, path: str = "") -> list:
    """Validate JSON data against a JSON Schema (subset implementation)."""
    errors = []
    schema_type = schema.get("type")

    if schema_type:
        type_map = {
            "string": str, "number": (int, float), "integer": int,
            "boolean": bool, "array": list, "object": dict, "null": type(None),
        }
        expected = type_map.get(schema_type)
        if expected and not isinstance(data, expected):
            errors.append(f"{path or 'root'}: expected {schema_type}, got {type(data).__name__}")
            return errors

    if schema_type == "string":
        min_len = schema.get("minLength", 0)
        max_len = schema.get("maxLength", float('inf'))
        if len(data) < min_len:
            errors.append(f"{path}: string too short (min {min_len})")
        if len(data) > max_len:
            errors.append(f"{path}: string too long (max {max_len})")
        pattern = schema.get("pattern")
        if pattern and not re.search(pattern, data):
            errors.append(f"{path}: string does not match pattern '{pattern}'")
        enum = schema.get("enum")
        if enum and data not in enum:
            errors.append(f"{path}: value must be one of {enum}")

    if schema_type in ("number", "integer"):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        if minimum is not None and data < minimum:
            errors.append(f"{path}: value {data} < minimum {minimum}")
        if maximum is not None and data > maximum:
            errors.append(f"{path}: value {data} > maximum {maximum}")

    if schema_type == "array":
        items_schema = schema.get("items", {})
        min_items = schema.get("minItems", 0)
        max_items = schema.get("maxItems", float('inf'))
        if len(data) < min_items:
            errors.append(f"{path}: array too short (min {min_items} items)")
        if len(data) > max_items:
            errors.append(f"{path}: array too long (max {max_items} items)")
        for i, item in enumerate(data):
            errors.extend(validate_json_schema(item, items_schema, f"{path}[{i}]"))

    if schema_type == "object":
        required = schema.get("required", [])
        properties = schema.get("properties", {})
        for req in required:
            if req not in data:
                errors.append(f"{path}.{req}: required field missing")
        for key, prop_schema in properties.items():
            if key in data:
                errors.extend(validate_json_schema(data[key], prop_schema, f"{path}.{key}"))

    return errors


# ─── Service: Hash Generation ─────────────────────────────────────────────────

def generate_hashes(text: str, algorithms: list = None, hmac_key: str = None) -> dict:
    """Generate cryptographic hashes for text."""
    if algorithms is None:
        algorithms = ["sha256", "sha512", "md5", "sha1"]

    encoded = text.encode('utf-8')
    result = {"input_length": len(text)}

    algo_map = {
        "md5": hashlib.md5,
        "sha1": hashlib.sha1,
        "sha224": hashlib.sha224,
        "sha256": hashlib.sha256,
        "sha384": hashlib.sha384,
        "sha512": hashlib.sha512,
    }

    for algo in algorithms:
        algo_lower = algo.lower()
        if algo_lower in algo_map:
            h = algo_map[algo_lower](encoded)
            result[algo_lower] = h.hexdigest()

    if hmac_key:
        key_bytes = hmac_key.encode('utf-8')
        result["hmac_sha256"] = hmac.new(key_bytes, encoded, hashlib.sha256).hexdigest()
        result["hmac_sha512"] = hmac.new(key_bytes, encoded, hashlib.sha512).hexdigest()

    return result


# ─── Service: Text Analytics ──────────────────────────────────────────────────

def analyze_text(text: str) -> dict:
    """Comprehensive text analysis."""
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

    word_count = len(words)
    sentence_count = max(len(sentences), 1)
    char_count = len(text)
    char_no_spaces = len(text.replace(' ', '').replace('\n', ''))

    # Syllable count (approximation)
    def count_syllables(word):
        word = word.lower()
        count = 0
        vowels = 'aeiouy'
        if word[0] in vowels:
            count += 1
        for i in range(1, len(word)):
            if word[i] in vowels and word[i-1] not in vowels:
                count += 1
        if word.endswith('e'):
            count -= 1
        return max(count, 1)

    total_syllables = sum(count_syllables(w) for w in words) if words else 0
    avg_syllables = total_syllables / max(word_count, 1)

    # Readability scores
    avg_words_per_sentence = word_count / sentence_count
    flesch_reading = 206.835 - (1.015 * avg_words_per_sentence) - (84.6 * avg_syllables)
    flesch_kincaid = (0.39 * avg_words_per_sentence) + (11.8 * avg_syllables) - 15.59

    # Reading level
    if flesch_reading >= 90:
        level = "Very Easy (5th grade)"
    elif flesch_reading >= 80:
        level = "Easy (6th grade)"
    elif flesch_reading >= 70:
        level = "Fairly Easy (7th grade)"
    elif flesch_reading >= 60:
        level = "Standard (8th-9th grade)"
    elif flesch_reading >= 50:
        level = "Fairly Difficult (10th-12th grade)"
    elif flesch_reading >= 30:
        level = "Difficult (College level)"
    else:
        level = "Very Difficult (Graduate level)"

    # Word frequency
    word_freq = Counter(words).most_common(20)

    # Unique words
    unique_words = len(set(words))
    lexical_diversity = unique_words / max(word_count, 1)

    # Estimated reading time (250 wpm average)
    reading_time_minutes = word_count / 250

    # Simple sentiment (positive/negative word lists)
    positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
                      'love', 'happy', 'best', 'perfect', 'beautiful', 'awesome',
                      'brilliant', 'outstanding', 'superb', 'delightful', 'pleasant',
                      'positive', 'success', 'win', 'easy', 'enjoy', 'helpful'}
    negative_words = {'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
                      'ugly', 'poor', 'fail', 'failure', 'difficult', 'hard',
                      'negative', 'wrong', 'error', 'broken', 'problem', 'issue',
                      'bug', 'crash', 'slow', 'frustrating', 'annoying'}

    pos_count = sum(1 for w in words if w in positive_words)
    neg_count = sum(1 for w in words if w in negative_words)
    sentiment_score = (pos_count - neg_count) / max(word_count, 1)
    if sentiment_score > 0.02:
        sentiment = "positive"
    elif sentiment_score < -0.02:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    return {
        "basic_stats": {
            "characters": char_count,
            "characters_no_spaces": char_no_spaces,
            "words": word_count,
            "unique_words": unique_words,
            "sentences": sentence_count,
            "paragraphs": len(paragraphs),
            "avg_word_length": round(sum(len(w) for w in words) / max(word_count, 1), 2),
            "avg_sentence_length": round(avg_words_per_sentence, 2),
        },
        "readability": {
            "flesch_reading_ease": round(flesch_reading, 2),
            "flesch_kincaid_grade": round(flesch_kincaid, 2),
            "reading_level": level,
            "estimated_reading_time": f"{reading_time_minutes:.1f} min",
        },
        "vocabulary": {
            "lexical_diversity": round(lexical_diversity, 4),
            "top_words": [{"word": w, "count": c} for w, c in word_freq],
        },
        "sentiment": {
            "label": sentiment,
            "score": round(sentiment_score, 4),
            "positive_words_found": pos_count,
            "negative_words_found": neg_count,
        },
    }


# ─── Service: Cron Expression Explainer ───────────────────────────────────────

def explain_cron(expression: str) -> dict:
    """Parse and explain a cron expression in human-readable form."""
    parts = expression.strip().split()

    if len(parts) == 5:
        labels = ["minute", "hour", "day_of_month", "month", "day_of_week"]
    elif len(parts) == 6:
        labels = ["second", "minute", "hour", "day_of_month", "month", "day_of_week"]
    elif len(parts) == 7:
        labels = ["second", "minute", "hour", "day_of_month", "month", "day_of_week", "year"]
    else:
        return {"error": f"Invalid cron expression: expected 5-7 fields, got {len(parts)}"}

    month_names = {1: "January", 2: "February", 3: "March", 4: "April",
                   5: "May", 6: "June", 7: "July", 8: "August",
                   9: "September", 10: "October", 11: "November", 12: "December"}
    day_names = {0: "Sunday", 1: "Monday", 2: "Tuesday", 3: "Wednesday",
                 4: "Thursday", 5: "Friday", 6: "Saturday", 7: "Sunday"}

    def explain_field(value, field_name):
        if value == '*':
            return f"every {field_name}"
        if '/' in value:
            base, step = value.split('/', 1)
            base_str = "starting from 0" if base == '*' else f"starting from {base}"
            return f"every {step} {field_name}s {base_str}"
        if '-' in value:
            start, end = value.split('-', 1)
            return f"{field_name}s {start} through {end}"
        if ',' in value:
            vals = value.split(',')
            return f"{field_name}s {', '.join(vals)}"
        if field_name == "month" and value.isdigit():
            return month_names.get(int(value), value)
        if field_name == "day_of_week" and value.isdigit():
            return day_names.get(int(value), value)
        return f"at {field_name} {value}"

    parsed = {}
    explanations = []
    for i, label in enumerate(labels):
        parsed[label] = parts[i]
        explanations.append(explain_field(parts[i], label))

    # Build human-readable summary
    summary_parts = []
    field_map = dict(zip(labels, parts))

    minute = field_map.get("minute", "*")
    hour = field_map.get("hour", "*")
    dom = field_map.get("day_of_month", "*")
    month = field_map.get("month", "*")
    dow = field_map.get("day_of_week", "*")

    if minute == "*" and hour == "*":
        summary_parts.append("Every minute")
    elif minute == "0" and hour == "*":
        summary_parts.append("Every hour at minute 0")
    elif minute == "0" and hour == "0":
        summary_parts.append("At midnight (00:00)")
    elif hour != "*" and minute != "*":
        summary_parts.append(f"At {hour.zfill(2)}:{minute.zfill(2)}")
    elif minute != "*":
        summary_parts.append(f"At minute {minute} of every hour")
    else:
        summary_parts.append(f"Every minute during hour {hour}")

    if dom != "*":
        summary_parts.append(f"on day {dom} of the month")
    if month != "*":
        if month.isdigit():
            summary_parts.append(f"in {month_names.get(int(month), month)}")
        else:
            summary_parts.append(f"in month {month}")
    if dow != "*":
        if dow.isdigit():
            summary_parts.append(f"on {day_names.get(int(dow), dow)}")
        else:
            summary_parts.append(f"on day of week {dow}")

    return {
        "expression": expression,
        "fields": parsed,
        "field_explanations": explanations,
        "summary": " ".join(summary_parts),
        "field_count": len(parts),
    }


# ─── Service: Color Palette ──────────────────────────────────────────────────

def hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


def generate_palette(base_color: str, palette_type: str = "complementary") -> dict:
    """Generate color palettes from a base color."""
    try:
        r, g, b = hex_to_rgb(base_color)
    except (ValueError, IndexError):
        return {"error": f"Invalid hex color: {base_color}"}

    h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
    colors = [{"hex": base_color, "role": "base", "rgb": {"r": r, "g": g, "b": b}}]

    if palette_type == "complementary":
        ch = (h + 0.5) % 1.0
        cr, cg, cb = [int(c * 255) for c in colorsys.hsv_to_rgb(ch, s, v)]
        colors.append({"hex": rgb_to_hex(cr, cg, cb), "role": "complementary",
                       "rgb": {"r": cr, "g": cg, "b": cb}})

    elif palette_type == "triadic":
        for offset in [1/3, 2/3]:
            nh = (h + offset) % 1.0
            nr, ng, nb = [int(c * 255) for c in colorsys.hsv_to_rgb(nh, s, v)]
            colors.append({"hex": rgb_to_hex(nr, ng, nb), "role": f"triadic",
                          "rgb": {"r": nr, "g": ng, "b": nb}})

    elif palette_type == "analogous":
        for offset in [-1/12, 1/12, -1/6, 1/6]:
            nh = (h + offset) % 1.0
            nr, ng, nb = [int(c * 255) for c in colorsys.hsv_to_rgb(nh, s, v)]
            colors.append({"hex": rgb_to_hex(nr, ng, nb), "role": "analogous",
                          "rgb": {"r": nr, "g": ng, "b": nb}})

    elif palette_type == "split_complementary":
        for offset in [5/12, 7/12]:
            nh = (h + offset) % 1.0
            nr, ng, nb = [int(c * 255) for c in colorsys.hsv_to_rgb(nh, s, v)]
            colors.append({"hex": rgb_to_hex(nr, ng, nb), "role": "split_complementary",
                          "rgb": {"r": nr, "g": ng, "b": nb}})

    elif palette_type == "monochromatic":
        for vmod in [0.2, 0.4, 0.6, 0.8]:
            nr, ng, nb = [int(c * 255) for c in colorsys.hsv_to_rgb(h, s, max(0, min(1, vmod)))]
            colors.append({"hex": rgb_to_hex(nr, ng, nb), "role": f"shade_{int(vmod*100)}",
                          "rgb": {"r": nr, "g": ng, "b": nb}})

    else:
        return {"error": f"Unknown palette type: {palette_type}. Options: complementary, triadic, analogous, split_complementary, monochromatic"}

    return {
        "base_color": base_color,
        "palette_type": palette_type,
        "colors": colors,
        "css_variables": {f"--color-{i}": c["hex"] for i, c in enumerate(colors)},
    }


# ─── Service: UUID Generator ─────────────────────────────────────────────────

def generate_uuids(count: int = 1, version: int = 4, namespace: str = None, name: str = None) -> dict:
    """Generate UUIDs."""
    count = min(max(count, 1), 1000)
    uuids = []

    for _ in range(count):
        if version == 4:
            uuids.append(str(uuid.uuid4()))
        elif version == 1:
            uuids.append(str(uuid.uuid1()))
        elif version == 5 and namespace and name:
            ns_map = {"dns": uuid.NAMESPACE_DNS, "url": uuid.NAMESPACE_URL,
                      "oid": uuid.NAMESPACE_OID, "x500": uuid.NAMESPACE_X500}
            ns = ns_map.get(namespace.lower(), uuid.NAMESPACE_DNS)
            uuids.append(str(uuid.uuid5(ns, name)))
        elif version == 3 and namespace and name:
            ns_map = {"dns": uuid.NAMESPACE_DNS, "url": uuid.NAMESPACE_URL,
                      "oid": uuid.NAMESPACE_OID, "x500": uuid.NAMESPACE_X500}
            ns = ns_map.get(namespace.lower(), uuid.NAMESPACE_DNS)
            uuids.append(str(uuid.uuid3(ns, name)))
        else:
            uuids.append(str(uuid.uuid4()))

    return {
        "count": len(uuids),
        "version": version,
        "uuids": uuids,
    }


# ─── Service: Encode/Decode ──────────────────────────────────────────────────

def encode_decode(text: str, operation: str = "base64_encode") -> dict:
    """Encode or decode text with various methods."""
    result = {"input": text, "operation": operation}

    try:
        if operation == "base64_encode":
            result["output"] = base64.b64encode(text.encode('utf-8')).decode('ascii')
        elif operation == "base64_decode":
            result["output"] = base64.b64decode(text).decode('utf-8')
        elif operation == "url_encode":
            result["output"] = urllib.parse.quote(text, safe='')
        elif operation == "url_decode":
            result["output"] = urllib.parse.unquote(text)
        elif operation == "hex_encode":
            result["output"] = text.encode('utf-8').hex()
        elif operation == "hex_decode":
            result["output"] = bytes.fromhex(text).decode('utf-8')
        elif operation == "rot13":
            result["output"] = text.translate(
                str.maketrans(
                    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                    'NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm'
                ))
        elif operation == "reverse":
            result["output"] = text[::-1]
        else:
            result["error"] = f"Unknown operation: {operation}. Options: base64_encode, base64_decode, url_encode, url_decode, hex_encode, hex_decode, rot13, reverse"
    except Exception as e:
        result["error"] = str(e)

    return result


# ─── Service: Password/Secret Generator ──────────────────────────────────────

# Built-in word list for passphrase generation
_PASSPHRASE_WORDS = [
    "correct", "horse", "battery", "staple", "abandon", "ability", "above",
    "absent", "absorb", "abstract", "absurd", "accept", "account", "accuse",
    "achieve", "acid", "across", "adapt", "address", "adjust", "admit",
    "advance", "advice", "afford", "agent", "agree", "ahead", "airport",
    "alarm", "album", "alert", "alien", "almost", "alpha", "alter",
    "always", "amateur", "anchor", "ancient", "angle", "animal", "ankle",
    "announce", "annual", "another", "antenna", "antique", "anxiety", "apart",
    "apology", "appear", "apple", "approve", "arctic", "arena", "armor",
    "arrow", "artist", "asthma", "athlete", "atom", "auction", "audit",
    "august", "autumn", "average", "avocado", "avoid", "awake", "aware",
    "awesome", "awful", "axis", "bamboo", "banana", "banner", "barrel",
    "basket", "battle", "beach", "beauty", "become", "before", "begin",
    "believe", "below", "bench", "benefit", "beyond", "bicycle", "blanket",
    "blind", "blood", "blossom", "board", "bonus", "bottom", "bounce",
    "brain", "brave", "bread", "bridge", "bring", "broken", "brother",
    "brown", "brush", "bubble", "bucket", "budget", "buffalo", "build",
    "bullet", "bundle", "burden", "burger", "butter", "cabin", "cable",
    "cactus", "cage", "camera", "camp", "canal", "cancel", "candle",
    "canvas", "canyon", "carbon", "cargo", "carpet", "carry", "castle",
    "casual", "catalog", "caught", "caution", "ceiling", "celery", "cement",
    "census", "cereal", "chamber", "change", "chapter", "charge", "cherry",
    "chicken", "chief", "choice", "chunk", "circle", "citizen", "claim",
    "classic", "clean", "clever", "cliff", "climb", "clinic", "clock",
    "cloud", "cluster", "coach", "coconut", "coffee", "collect", "color",
    "column", "combine", "comfort", "comic", "common", "company", "concert",
    "confirm", "connect", "consider", "control", "convert", "cookie", "copper",
    "coral", "corner", "cosmic", "cotton", "couch", "country", "couple",
    "course", "cousin", "cover", "coyote", "cradle", "craft", "crane",
    "crash", "crater", "crawl", "crazy", "cream", "credit", "creek",
    "cricket", "crime", "crisp", "cross", "crowd", "crucial", "cruel",
    "cruise", "crumble", "crystal", "culture", "cupboard", "curtain", "curve",
    "cycle", "damage", "dance", "danger", "daring", "dawn", "debate",
    "decade", "december", "decide", "decline", "decorate", "decrease", "defeat",
    "defend", "define", "defy", "degree", "delay", "deliver", "demand",
    "denial", "dentist", "depart", "depend", "deposit", "depth", "deputy",
    "derive", "describe", "desert", "design", "detail", "detect", "develop",
    "device", "devote", "diagram", "diamond", "diary", "diesel", "differ",
    "digital", "dignity", "dilemma", "dinner", "dinosaur", "direct", "dirt",
    "disagree", "discover", "disease", "dismiss", "display", "distance", "divert",
    "divide", "doctor", "document", "dolphin", "domain", "donate", "donkey",
    "double", "dragon", "drama", "drastic", "dream", "drift", "drink",
    "driver", "drop", "drum", "duck", "dumb", "dune", "during",
    "dust", "dwarf", "dynamic", "eager", "eagle", "early", "earth",
    "easily", "echo", "ecology", "economy", "editor", "educate", "effort",
    "eight", "either", "elbow", "elder", "electric", "elegant", "element",
    "elephant", "elevator", "elite", "embark", "embrace", "emerge", "emotion",
    "employ", "empower", "empty", "enable", "enact", "endless", "endorse",
    "enemy", "energy", "enforce", "engage", "engine", "enhance", "enjoy",
    "enough", "ensure", "enter", "entire", "entry", "envelope", "episode",
    "equal", "equip", "erosion", "escape", "essay", "essence", "estate",
    "eternal", "evening", "evidence", "evolve", "exact", "example", "excess",
    "exchange", "excite", "exclude", "excuse", "execute", "exercise", "exhaust",
    "exhibit", "exile", "exist", "expand", "expect", "expire", "explain",
    "expose", "express", "extend", "extra", "fabric", "faculty", "faith",
    "family", "famous", "fancy", "fantasy", "fashion", "fatal", "father",
    "fatigue", "fault", "favorite", "feature", "february", "federal", "fence",
    "festival", "fetch", "fever", "fiction", "field", "figure", "filter",
    "final", "finger", "finish", "fire", "fiscal", "fitness", "flag",
    "flame", "flash", "flavor", "flight", "float", "flood", "floor",
    "flower", "fluid", "flush", "focus", "follow", "force", "forest",
    "forget", "fork", "fortune", "forum", "forward", "fossil", "foster",
    "found", "fragile", "frame", "frequent", "fresh", "friend", "fringe",
    "frozen", "fruit", "fuel", "funny", "furnace", "fury", "future",
    "gadget", "galaxy", "gallery", "game", "garage", "garden", "garlic",
    "garment", "gather", "gauge", "genius", "genre", "gentle", "genuine",
    "gesture", "ghost", "giant", "gift", "giggle", "ginger", "giraffe",
    "glance", "glimpse", "globe", "gloom", "glory", "glove", "glow",
    "goddess", "gospel", "gossip", "govern", "grace", "grain", "grant",
    "grape", "gravity", "green", "grid", "grief", "grit", "grocery",
    "group", "grow", "grunt", "guard", "guess", "guide", "guitar",
    "habit", "hammer", "hamster", "happen", "harbor", "harvest", "hawk",
    "hazard", "heart", "heavy", "hello", "helmet", "heritage", "hero",
    "hidden", "highway", "hint", "history", "hobby", "hollow", "honey",
    "horizon", "horror", "hospital", "hotel", "hover", "humble", "humor",
    "hundred", "hungry", "hurdle", "hybrid", "icon", "identify", "ignore",
    "image", "immune", "impact", "impose", "improve", "impulse", "include",
    "income", "increase", "index", "indicate", "indoor", "industry", "infant",
    "inflict", "inform", "initial", "inject", "inner", "innocent", "input",
    "inquiry", "insane", "insect", "inside", "inspire", "install", "intact",
    "interest", "invest", "invite", "involve", "iron", "island", "isolate",
    "ivory", "jacket", "jaguar", "jealous", "jelly", "jewel", "journey",
    "judge", "juice", "jungle", "junior", "junk", "justice", "kangaroo",
    "kayak", "keen", "kernel", "kidney", "kingdom", "kitchen", "kite",
    "kitten", "knife", "knock", "label", "ladder", "lamp", "language",
    "laptop", "large", "later", "Latin", "laugh", "laundry", "layer",
    "leader", "learn", "leave", "lecture", "legal", "legend", "leisure",
    "lemon", "length", "lens", "leopard", "lesson", "letter", "level",
    "liberty", "library", "license", "light", "limit", "link", "liquid",
    "little", "lively", "lizard", "lobster", "local", "logic", "lonely",
    "lottery", "louder", "lounge", "loyal", "lucky", "lumber", "lunar",
    "luxury", "machine", "magazine", "magnet", "mango", "mansion", "maple",
    "marble", "margin", "marine", "market", "master", "matrix", "maximum",
    "meadow", "measure", "media", "melody", "member", "memory", "mention",
    "mercy", "merge", "merit", "method", "middle", "million", "mimic",
    "mineral", "minimum", "miracle", "mirror", "misery", "mission", "mobile",
    "model", "modify", "moment", "monitor", "monkey", "monster", "month",
    "moral", "morning", "motion", "mountain", "mouse", "movie", "much",
    "muffin", "multiple", "muscle", "museum", "music", "mutual", "myself",
    "mystery", "narrow", "nation", "nature", "negative", "neglect", "neither",
    "nephew", "nerve", "network", "neutral", "never", "noble", "nominal",
    "noodle", "normal", "north", "notable", "nothing", "notice", "novel",
    "nuclear", "number", "nurse", "object", "oblige", "observe", "obtain",
    "obvious", "ocean", "october", "offer", "office", "olive", "olympic",
    "opinion", "oppose", "option", "orange", "orbit", "order", "organ",
    "orient", "original", "orphan", "ostrich", "outdoor", "output", "outside",
    "oval", "owner", "oxygen", "oyster", "paddle", "palace", "panda",
    "panel", "panic", "panther", "paper", "parade", "parent", "park",
    "parrot", "party", "passion", "patch", "patient", "pattern", "pause",
    "peanut", "pelican", "penalty", "pencil", "people", "pepper", "perfect",
    "permit", "person", "phrase", "piano", "picnic", "picture", "piece",
    "pilot", "pioneer", "pizza", "planet", "plastic", "platform", "player",
    "please", "pledge", "pluck", "plunge", "poetry", "point", "polar",
    "policy", "polish", "pond", "popular", "position", "possible", "potato",
    "pottery", "poverty", "powder", "power", "practice", "predict", "prefer",
    "prepare", "present", "pretty", "prevent", "primary", "prince", "prison",
    "private", "problem", "process", "produce", "profit", "program", "project",
    "promote", "proof", "property", "prosper", "protect", "proud", "provide",
    "public", "pulse", "pumpkin", "punch", "pupil", "purchase", "purple",
    "puzzle", "pyramid", "quality", "quantum", "quarter", "question", "quick",
    "rabbit", "raccoon", "radar", "radio", "rail", "rainbow", "random",
    "range", "rapid", "rather", "raven", "razor", "ready", "reason",
    "rebel", "rebuild", "recall", "receive", "recipe", "record", "recycle",
    "reduce", "reflect", "reform", "refuse", "region", "regret", "regular",
    "reject", "relax", "release", "relief", "rely", "remain", "remember",
    "remind", "remove", "render", "renew", "repair", "repeat", "replace",
    "report", "require", "rescue", "resist", "resource", "response", "result",
    "retire", "retreat", "return", "reveal", "review", "reward", "rhythm",
    "ribbon", "rifle", "right", "rigid", "ring", "ripple", "river",
    "road", "robot", "robust", "rocket", "romance", "rough", "round",
    "royal", "rubber", "rude", "rural", "saddle", "safety", "salad",
    "salmon", "salon", "sample", "satisfy", "satoshi", "sauce", "sausage",
    "scale", "scatter", "scene", "scheme", "school", "science", "scissors",
    "scorpion", "scout", "screen", "script", "search", "season", "secret",
    "section", "security", "select", "senior", "sense", "sentence", "series",
    "service", "session", "settle", "setup", "shadow", "shaft", "shallow",
    "share", "shelter", "sheriff", "shield", "shift", "shine", "ship",
    "shock", "shoulder", "shove", "shrimp", "shuttle", "sibling", "siege",
    "sight", "silent", "silver", "similar", "simple", "since", "siren",
    "sister", "situate", "sketch", "skill", "slender", "slice", "slogan",
    "slot", "smart", "smile", "smooth", "snack", "snake", "snap",
    "social", "soldier", "solution", "someone", "sorry", "source", "south",
    "space", "spare", "spatial", "spawn", "special", "speed", "sphere",
    "spider", "spirit", "split", "sponsor", "spoon", "sport", "spray",
    "spread", "spring", "squirrel", "stable", "stadium", "staff", "stage",
    "stamp", "stand", "start", "state", "station", "stay", "steak",
    "steel", "stem", "step", "stereo", "stick", "still", "stock",
    "stomach", "stone", "stool", "story", "strategy", "street", "strike",
    "strong", "struggle", "student", "stuff", "stumble", "style", "subject",
    "submit", "sudden", "suffer", "sugar", "suggest", "summer", "sun",
    "super", "supply", "supreme", "surface", "surge", "surprise", "surround",
    "survey", "suspect", "sustain", "swallow", "swamp", "swap", "sweet",
    "swift", "swim", "switch", "symbol", "symptom", "syrup", "system",
    "table", "tackle", "talent", "tank", "tape", "target", "task",
    "tattoo", "taxi", "teach", "team", "tenant", "tennis", "term",
    "test", "text", "thank", "theme", "theory", "thought", "three",
    "thrive", "throw", "thumb", "thunder", "ticket", "tiger", "timber",
    "title", "toast", "tobacco", "today", "toddler", "tomato", "tomorrow",
    "tongue", "tonight", "topic", "torch", "tornado", "tortoise", "total",
    "tourist", "toward", "tower", "town", "trade", "traffic", "train",
    "transfer", "trash", "travel", "tray", "treasure", "tree", "trend",
    "trial", "tribe", "trick", "trigger", "trim", "trip", "trophy",
    "trouble", "truck", "truly", "trumpet", "trust", "truth", "tumble",
    "tunnel", "turkey", "turn", "turtle", "twelve", "twenty", "twice",
    "type", "typical", "umbrella", "unable", "unaware", "uncle", "under",
    "unfair", "unfold", "unhappy", "uniform", "unique", "unit", "universe",
    "unknown", "unlock", "until", "unusual", "unveil", "update", "upgrade",
    "upper", "upset", "urban", "usage", "useful", "useless", "usual",
    "utility", "vacant", "vacuum", "valid", "valley", "valve", "vanish",
    "vapor", "various", "vast", "vault", "vehicle", "velvet", "vendor",
    "venture", "verb", "verify", "version", "vessel", "veteran", "viable",
    "vibrant", "victim", "victory", "video", "view", "village", "vintage",
    "violin", "virtual", "virus", "visa", "visit", "visual", "vital",
    "vivid", "vocal", "voice", "volcano", "volume", "voyage", "waffle",
    "wagon", "walnut", "wander", "warfare", "warrior", "wash", "waste",
    "water", "wealth", "weapon", "weather", "wedding", "weekend", "welcome",
    "western", "whale", "wheat", "wheel", "whisper", "width", "wild",
    "window", "winter", "wisdom", "witness", "wolf", "woman", "wonder",
    "world", "worry", "worth", "wrap", "wreck", "wrestle", "wrist",
    "write", "wrong", "yellow", "young", "youth", "zebra", "zero", "zone",
]


def _calculate_entropy(length: int, charset_size: int) -> float:
    """Calculate password entropy in bits."""
    if charset_size <= 0 or length <= 0:
        return 0.0
    return length * math.log2(charset_size)


def _strength_rating(entropy: float) -> str:
    """Rate password strength based on entropy bits."""
    if entropy >= 128:
        return "very_strong"
    elif entropy >= 80:
        return "strong"
    elif entropy >= 60:
        return "moderate"
    elif entropy >= 40:
        return "weak"
    else:
        return "very_weak"


def generate_password(length: int = 16, count: int = 1, secret_type: str = "password") -> dict:
    """Generate secure passwords, API keys, passphrases, or PINs."""
    count = min(max(count, 1), 100)
    results = []

    if secret_type == "password":
        length = min(max(length, 4), 256)
        charset = string.ascii_letters + string.digits + string.punctuation
        charset_size = len(charset)

        for _ in range(count):
            # Guarantee at least one of each category
            pw_chars = [
                secrets.choice(string.ascii_uppercase),
                secrets.choice(string.ascii_lowercase),
                secrets.choice(string.digits),
                secrets.choice(string.punctuation),
            ]
            for _ in range(length - 4):
                pw_chars.append(secrets.choice(charset))
            # Shuffle to avoid predictable positions
            shuffled = list(pw_chars)
            for i in range(len(shuffled) - 1, 0, -1):
                j = secrets.randbelow(i + 1)
                shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
            password = ''.join(shuffled)
            entropy = _calculate_entropy(length, charset_size)
            results.append({
                "value": password,
                "length": length,
                "entropy_bits": round(entropy, 2),
                "strength": _strength_rating(entropy),
            })

    elif secret_type == "api_key":
        length = min(max(length, 8), 256)

        for _ in range(count):
            # Generate random bytes and encode as hex
            random_bytes = secrets.token_bytes(length)
            hex_key = random_bytes.hex()[:length]
            b64_key = base64.urlsafe_b64encode(random_bytes).decode('ascii')[:length]
            prefixed_key = f"sk_live_{secrets.token_hex(length)[:length]}"
            entropy = _calculate_entropy(length, 16)  # hex charset
            results.append({
                "value": prefixed_key,
                "hex_variant": hex_key,
                "base64_variant": b64_key,
                "length": len(prefixed_key),
                "entropy_bits": round(entropy, 2),
                "strength": _strength_rating(entropy),
            })

    elif secret_type == "passphrase":
        # length here means number of words
        word_count = min(max(length, 3), 20)
        word_list = _PASSPHRASE_WORDS
        charset_size = len(word_list)

        for _ in range(count):
            words = [secrets.choice(word_list) for _ in range(word_count)]
            passphrase = "-".join(words)
            entropy = _calculate_entropy(word_count, charset_size)
            results.append({
                "value": passphrase,
                "word_count": word_count,
                "entropy_bits": round(entropy, 2),
                "strength": _strength_rating(entropy),
            })

    elif secret_type == "pin":
        length = min(max(length, 4), 32)
        charset_size = 10

        for _ in range(count):
            pin = ''.join(str(secrets.randbelow(10)) for _ in range(length))
            entropy = _calculate_entropy(length, charset_size)
            results.append({
                "value": pin,
                "length": length,
                "entropy_bits": round(entropy, 2),
                "strength": _strength_rating(entropy),
            })

    else:
        return {"error": f"Unknown type: {secret_type}. Options: password, api_key, passphrase, pin"}

    return {
        "type": secret_type,
        "count": len(results),
        "secrets": results,
    }


# ─── Service: JWT Decoder ────────────────────────────────────────────────────

def _base64url_decode(data: str) -> bytes:
    """Decode base64url-encoded data with proper padding."""
    # Add padding if needed
    padding = 4 - len(data) % 4
    if padding != 4:
        data += '=' * padding
    # Replace URL-safe characters
    data = data.replace('-', '+').replace('_', '/')
    return base64.b64decode(data)


def decode_jwt(token: str) -> dict:
    """Decode a JWT token without verification."""
    parts = token.strip().split('.')

    if len(parts) != 3:
        return {"error": f"Invalid JWT: expected 3 parts separated by '.', got {len(parts)}"}

    result = {"raw_parts": {"header": parts[0], "payload": parts[1], "signature": parts[2]}}

    # Decode header
    try:
        header_bytes = _base64url_decode(parts[0])
        header = json.loads(header_bytes.decode('utf-8'))
        result["header"] = header
    except Exception as e:
        return {"error": f"Failed to decode JWT header: {e}"}

    # Decode payload
    try:
        payload_bytes = _base64url_decode(parts[1])
        payload = json.loads(payload_bytes.decode('utf-8'))
        result["payload"] = payload
    except Exception as e:
        return {"error": f"Failed to decode JWT payload: {e}"}

    # Check signature presence
    result["signature_present"] = len(parts[2]) > 0

    # Analyze standard claims
    claims_analysis = {}

    if "iss" in payload:
        claims_analysis["issuer"] = payload["iss"]
    if "sub" in payload:
        claims_analysis["subject"] = payload["sub"]
    if "aud" in payload:
        claims_analysis["audience"] = payload["aud"]

    # Check expiry
    now = int(time.time())
    if "exp" in payload:
        exp = payload["exp"]
        claims_analysis["expires_at"] = datetime.fromtimestamp(exp).isoformat()
        claims_analysis["is_expired"] = now > exp
        if now > exp:
            claims_analysis["expired_seconds_ago"] = now - exp
        else:
            claims_analysis["expires_in_seconds"] = exp - now
    else:
        claims_analysis["is_expired"] = None  # No expiry claim

    if "iat" in payload:
        claims_analysis["issued_at"] = datetime.fromtimestamp(payload["iat"]).isoformat()
    if "nbf" in payload:
        claims_analysis["not_before"] = datetime.fromtimestamp(payload["nbf"]).isoformat()
        claims_analysis["is_active"] = now >= payload["nbf"]

    result["claims_analysis"] = claims_analysis
    result["algorithm"] = header.get("alg", "unknown")
    result["token_type"] = header.get("typ", "unknown")

    return result


# ─── Service: Diff/Patch Tool ────────────────────────────────────────────────

def generate_diff(text_a: str, text_b: str, context_lines: int = 3) -> dict:
    """Generate unified diff between two texts with statistics."""
    context_lines = min(max(context_lines, 0), 50)

    lines_a = text_a.splitlines(keepends=True)
    lines_b = text_b.splitlines(keepends=True)

    # Generate unified diff
    diff_lines = list(difflib.unified_diff(
        lines_a, lines_b,
        fromfile="text_a",
        tofile="text_b",
        n=context_lines,
    ))

    unified_diff = ''.join(diff_lines)

    # Calculate statistics
    additions = 0
    deletions = 0
    for line in diff_lines:
        if line.startswith('+') and not line.startswith('+++'):
            additions += 1
        elif line.startswith('-') and not line.startswith('---'):
            deletions += 1

    changes = min(additions, deletions)

    # Calculate similarity ratio
    matcher = difflib.SequenceMatcher(None, text_a, text_b)
    similarity = matcher.ratio()

    # Also provide a line-level comparison
    line_matcher = difflib.SequenceMatcher(None, lines_a, lines_b)
    line_similarity = line_matcher.ratio()

    return {
        "unified_diff": unified_diff,
        "stats": {
            "additions": additions,
            "deletions": deletions,
            "changes": changes,
            "total_lines_a": len(lines_a),
            "total_lines_b": len(lines_b),
        },
        "similarity": {
            "character_ratio": round(similarity, 4),
            "line_ratio": round(line_similarity, 4),
            "percentage": f"{similarity * 100:.1f}%",
        },
        "has_differences": len(diff_lines) > 0,
        "context_lines": context_lines,
    }


# ─── Service: Template Engine ────────────────────────────────────────────────

def _apply_filter(value: str, filter_name: str) -> str:
    """Apply a template filter to a value."""
    if filter_name == "upper":
        return value.upper()
    elif filter_name == "lower":
        return value.lower()
    elif filter_name == "title":
        return value.title()
    elif filter_name == "strip":
        return value.strip()
    elif filter_name == "capitalize":
        return value.capitalize()
    elif filter_name == "length":
        return str(len(value))
    elif filter_name.startswith("default:"):
        # {{value|default:"N/A"}} — value is already resolved; this is handled at lookup
        return value
    return value


def _resolve_variable(name: str, variables: dict) -> str:
    """Resolve a variable name (supports dot notation) and apply filters."""
    # Split off filters
    parts = name.split('|')
    var_name = parts[0].strip()
    filters = [f.strip() for f in parts[1:]]

    # Check for default filter first (needed if variable is missing)
    default_value = None
    for f in filters:
        if f.startswith("default:"):
            default_raw = f[len("default:"):]
            # Strip surrounding quotes if present
            if (default_raw.startswith('"') and default_raw.endswith('"')) or \
               (default_raw.startswith("'") and default_raw.endswith("'")):
                default_value = default_raw[1:-1]
            else:
                default_value = default_raw

    # Resolve variable with dot notation
    current = variables
    for key in var_name.split('.'):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            # Variable not found
            if default_value is not None:
                return default_value
            return ""

    value = str(current)

    # Apply filters (except default, which was already handled)
    for f in filters:
        if not f.startswith("default:"):
            value = _apply_filter(value, f)

    return value


def _evaluate_condition(condition_name: str, variables: dict) -> Any:
    """Evaluate a condition for if/unless blocks."""
    name = condition_name.strip()
    negate = False
    if name.startswith("not "):
        negate = True
        name = name[4:].strip()

    # Resolve variable
    current = variables
    for key in name.split('.'):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            current = None
            break

    result = bool(current)
    if negate:
        result = not result
    return result, current


def _resolve_each(items_name: str, variables: dict):
    """Resolve items for each blocks."""
    current = variables
    for key in items_name.strip().split('.'):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return []
    if isinstance(current, list):
        return current
    return []


def render_template(template: str, variables: dict) -> dict:
    """Render a template with Mustache/Jinja-like syntax."""
    output = template
    errors = []

    # Process {{#each items}}...{{/each}} blocks (innermost first to handle nesting)
    max_iterations = 50  # Safety guard against infinite loops
    iteration = 0
    each_pattern = re.compile(r'\{\{#each\s+(\w[\w.]*)\}\}(.*?)\{\{/each\}\}', re.DOTALL)
    while each_pattern.search(output) and iteration < max_iterations:
        iteration += 1
        match = each_pattern.search(output)
        items_name = match.group(1)
        body = match.group(2)
        items = _resolve_each(items_name, variables)

        rendered_items = []
        for i, item in enumerate(items):
            item_output = body
            if isinstance(item, dict):
                # Replace {{key}} references within the loop body with item values
                def replace_item_var(m, _item=item):
                    var_expr = m.group(1).strip()
                    parts = var_expr.split('|')
                    key = parts[0].strip()
                    filters = [f.strip() for f in parts[1:]]

                    if key == "@index":
                        val = str(i)
                    elif key == "@first":
                        val = str(i == 0)
                    elif key == "@last":
                        val = str(i == len(items) - 1)
                    elif key in _item:
                        val = str(_item[key])
                    else:
                        # Fall back to outer variables
                        val = _resolve_variable(var_expr, variables)
                        return val

                    for f in filters:
                        if not f.startswith("default:"):
                            val = _apply_filter(val, f)
                    return val

                item_output = re.sub(r'\{\{([^#/}][^}]*?)\}\}', replace_item_var, item_output)
            else:
                # Scalar item — replace {{.}} or {{this}} with the value
                item_output = item_output.replace('{{.}}', str(item))
                item_output = item_output.replace('{{this}}', str(item))

            rendered_items.append(item_output)

        output = output[:match.start()] + ''.join(rendered_items) + output[match.end():]

    # Process {{#if condition}}...{{/if}} blocks
    iteration = 0
    if_pattern = re.compile(r'\{\{#if\s+(.+?)\}\}(.*?)\{\{/if\}\}', re.DOTALL)
    while if_pattern.search(output) and iteration < max_iterations:
        iteration += 1
        match = if_pattern.search(output)
        condition_name = match.group(1)
        body = match.group(2)

        truthy, _ = _evaluate_condition(condition_name, variables)
        if truthy:
            output = output[:match.start()] + body + output[match.end():]
        else:
            output = output[:match.start()] + output[match.end():]

    # Process {{#unless condition}}...{{/unless}} blocks
    iteration = 0
    unless_pattern = re.compile(r'\{\{#unless\s+(.+?)\}\}(.*?)\{\{/unless\}\}', re.DOTALL)
    while unless_pattern.search(output) and iteration < max_iterations:
        iteration += 1
        match = unless_pattern.search(output)
        condition_name = match.group(1)
        body = match.group(2)

        truthy, _ = _evaluate_condition(condition_name, variables)
        if not truthy:
            output = output[:match.start()] + body + output[match.end():]
        else:
            output = output[:match.start()] + output[match.end():]

    # Process remaining {{variable}} and {{variable|filter}} expressions
    def replace_var(match):
        expr = match.group(1).strip()
        return _resolve_variable(expr, variables)

    output = re.sub(r'\{\{([^#/}][^}]*?)\}\}', replace_var, output)

    return {
        "rendered": output,
        "template_length": len(template),
        "output_length": len(output),
        "variables_provided": list(variables.keys()),
    }


# ─── Service: Regex Tester ────────────────────────────────────────────────────

def test_regex(pattern: str, test_string: str, flags_str: str = "") -> dict:
    """Test a regex pattern against a string and return all matches."""
    flag_map = {"i": re.IGNORECASE, "m": re.MULTILINE, "s": re.DOTALL}
    flags = 0
    for f in flags_str:
        if f in flag_map:
            flags |= flag_map[f]

    try:
        compiled = re.compile(pattern, flags)
    except re.error as e:
        return {"valid": False, "error": str(e), "pattern": pattern}

    matches = []
    for m in compiled.finditer(test_string):
        match_info = {
            "match": m.group(),
            "start": m.start(),
            "end": m.end(),
            "groups": list(m.groups()),
        }
        if m.groupdict():
            match_info["named_groups"] = m.groupdict()
        matches.append(match_info)

    return {
        "valid": True,
        "pattern": pattern,
        "flags": flags_str,
        "test_string_length": len(test_string),
        "match_count": len(matches),
        "matches": matches,
        "full_match": bool(compiled.fullmatch(test_string)),
    }


# ─── Service: Slug Generator ────────────────────────────────────────────────

def generate_slug(text: str, separator: str = "-", max_length: int = 80) -> dict:
    """Generate a URL-friendly slug from text."""
    # Transliterate common special characters
    replacements = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss', 'ñ': 'n',
        'à': 'a', 'á': 'a', 'â': 'a', 'ã': 'a', 'å': 'a',
        'è': 'e', 'é': 'e', 'ê': 'e', 'ë': 'e',
        'ì': 'i', 'í': 'i', 'î': 'i', 'ï': 'i',
        'ò': 'o', 'ó': 'o', 'ô': 'o', 'õ': 'o',
        'ù': 'u', 'ú': 'u', 'û': 'u',
        'ç': 'c', 'ð': 'd', 'ý': 'y', 'þ': 'th',
    }
    slug = text.lower()
    for orig, repl in replacements.items():
        slug = slug.replace(orig, repl)

    # Remove non-alphanumeric chars (except separator)
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'[\s-]+', separator, slug).strip(separator)

    # Truncate at max_length on word boundary
    if len(slug) > max_length:
        slug = slug[:max_length].rsplit(separator, 1)[0]

    return {
        "slug": slug,
        "original": text,
        "length": len(slug),
        "separator": separator,
    }


# ─── Service: CSV to JSON ───────────────────────────────────────────────────

def csv_to_json(csv_text: str, delimiter: str = ",", has_header: bool = True) -> dict:
    """Convert CSV text to JSON array."""
    lines = csv_text.strip().split('\n')
    if not lines:
        return {"error": "Empty CSV input", "rows": [], "row_count": 0}

    # Parse CSV manually (no csv module needed)
    def parse_csv_line(line: str, delim: str) -> list:
        """Parse a single CSV line handling quoted fields."""
        fields = []
        current = []
        in_quotes = False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == '"':
                if in_quotes and i + 1 < len(line) and line[i + 1] == '"':
                    current.append('"')
                    i += 2
                    continue
                in_quotes = not in_quotes
            elif ch == delim and not in_quotes:
                fields.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
            i += 1
        fields.append(''.join(current).strip())
        return fields

    parsed_lines = [parse_csv_line(line, delimiter) for line in lines]

    if has_header and len(parsed_lines) > 1:
        headers = parsed_lines[0]
        rows = []
        for row in parsed_lines[1:]:
            obj = {}
            for i, header in enumerate(headers):
                val = row[i] if i < len(row) else ""
                # Try to convert to number
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                obj[header] = val
            rows.append(obj)
    else:
        headers = None
        rows = parsed_lines

    return {
        "rows": rows,
        "row_count": len(rows),
        "columns": headers if headers else len(parsed_lines[0]) if parsed_lines else 0,
        "delimiter": delimiter,
    }


# ─── Service: IP Info ────────────────────────────────────────────────────────

def analyze_ip(ip: str) -> dict:
    """Analyze an IP address - classify, validate, and provide info."""
    import struct

    result = {
        "ip": ip,
        "valid": False,
        "version": None,
        "type": None,
        "class": None,
        "binary": None,
    }

    # IPv4 validation
    ipv4_match = re.match(r'^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$', ip)
    if ipv4_match:
        octets = [int(g) for g in ipv4_match.groups()]
        if all(0 <= o <= 255 for o in octets):
            result["valid"] = True
            result["version"] = 4
            result["octets"] = octets
            result["binary"] = '.'.join(f'{o:08b}' for o in octets)
            result["decimal"] = (octets[0] << 24) + (octets[1] << 16) + (octets[2] << 8) + octets[3]
            result["hex"] = '.'.join(f'{o:02x}' for o in octets)

            # Classify
            first = octets[0]
            if first == 0:
                result["class"] = "reserved"
                result["type"] = "This network"
            elif first == 10:
                result["class"] = "A"
                result["type"] = "Private (RFC 1918)"
            elif first == 127:
                result["class"] = "A"
                result["type"] = "Loopback"
            elif first >= 1 and first <= 126:
                result["class"] = "A"
                if first == 100 and 64 <= octets[1] <= 127:
                    result["type"] = "Shared Address Space (RFC 6598)"
                else:
                    result["type"] = "Public"
            elif first >= 128 and first <= 191:
                result["class"] = "B"
                if first == 172 and 16 <= octets[1] <= 31:
                    result["type"] = "Private (RFC 1918)"
                elif first == 169 and octets[1] == 254:
                    result["type"] = "Link-local (APIPA)"
                else:
                    result["type"] = "Public"
            elif first >= 192 and first <= 223:
                result["class"] = "C"
                if first == 192 and octets[1] == 168:
                    result["type"] = "Private (RFC 1918)"
                else:
                    result["type"] = "Public"
            elif first >= 224 and first <= 239:
                result["class"] = "D"
                result["type"] = "Multicast"
            elif first >= 240:
                result["class"] = "E"
                if ip == "255.255.255.255":
                    result["type"] = "Broadcast"
                else:
                    result["type"] = "Reserved"

            result["is_private"] = "Private" in (result["type"] or "")
            result["is_loopback"] = first == 127
            result["is_multicast"] = 224 <= first <= 239

    # Basic IPv6 validation
    elif ':' in ip:
        result["version"] = 6
        # Simple IPv6 validation
        parts = ip.split(':')
        try:
            # Handle :: expansion
            if '::' in ip:
                result["valid"] = True
                result["type"] = "IPv6"
                if ip == '::1':
                    result["type"] = "IPv6 Loopback"
                    result["is_loopback"] = True
                elif ip.startswith('fe80:'):
                    result["type"] = "IPv6 Link-local"
                elif ip.startswith('fc') or ip.startswith('fd'):
                    result["type"] = "IPv6 Unique Local"
                    result["is_private"] = True
            elif len(parts) == 8:
                for p in parts:
                    int(p, 16)  # validate hex
                result["valid"] = True
                result["type"] = "IPv6 Global Unicast"
        except ValueError:
            pass

    return result


# ─── HTTP Handler ─────────────────────────────────────────────────────────────

class EveServiceHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Eve's services."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(body))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            self.send_json({
                "status": "healthy",
                "service": "Eve's Utility Services",
                "version": VERSION,
                "uptime_since": stats["started_at"],
                "total_requests": stats["total_requests"],
            })
        elif self.path == '/catalog':
            self.send_json({
                "agent": "Eve",
                "ticker": "EVE",
                "version": VERSION,
                "description": "Practical developer utility toolkit — complementary to Adam's services",
                "services": {
                    name: {
                        "price_usd": price,
                        "method": "POST",
                        "endpoint": f"/{name}",
                    }
                    for name, price in PRICES.items()
                },
                "stats": {
                    "total_requests": stats["total_requests"],
                    "total_revenue": round(stats["total_revenue"], 2),
                },
            })
        else:
            self.send_json({"error": "Not found. Try GET /catalog or GET /health"}, 404)

    def do_POST(self):
        path = self.path.lstrip('/')
        if path not in PRICES:
            self.send_json({"error": f"Unknown service: {path}. GET /catalog for available services."}, 404)
            return

        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length).decode('utf-8') if content_length else '{}'
            data = json.loads(body)
        except (json.JSONDecodeError, ValueError) as e:
            self.send_json({"error": f"Invalid JSON: {e}"}, 400)
            return

        stats["total_requests"] += 1
        stats["by_service"][path] = stats["by_service"].get(path, 0) + 1

        try:
            result = self.dispatch(path, data)
            price = PRICES[path]
            stats["total_revenue"] += price
            result["_meta"] = {
                "service": path,
                "price_usd": price,
                "agent": "Eve",
                "version": VERSION,
            }
            # Report revenue asynchronously
            threading.Thread(target=report_revenue, args=(price, path), daemon=True).start()
            self.send_json(result)
        except Exception as e:
            stats["errors"] += 1
            self.send_json({"error": str(e), "service": path}, 500)

    def dispatch(self, service: str, data: dict) -> dict:
        if service == "markdown":
            text = data.get("text", data.get("markdown", ""))
            if not text:
                return {"error": "Missing 'text' field"}
            return {"html": markdown_to_html(text), "input_length": len(text)}

        elif service == "json_validate":
            payload = data.get("data")
            schema = data.get("schema")
            if payload is None or schema is None:
                return {"error": "Missing 'data' and/or 'schema' fields"}
            errors = validate_json_schema(payload, schema)
            return {"valid": len(errors) == 0, "errors": errors, "error_count": len(errors)}

        elif service == "hash":
            text = data.get("text", "")
            if not text:
                return {"error": "Missing 'text' field"}
            algorithms = data.get("algorithms", ["sha256", "sha512", "md5", "sha1"])
            hmac_key = data.get("hmac_key")
            return generate_hashes(text, algorithms, hmac_key)

        elif service == "text_analytics":
            text = data.get("text", "")
            if not text:
                return {"error": "Missing 'text' field"}
            return analyze_text(text)

        elif service == "cron_explain":
            expression = data.get("expression", "")
            if not expression:
                return {"error": "Missing 'expression' field"}
            return explain_cron(expression)

        elif service == "color_palette":
            color = data.get("color", data.get("hex", ""))
            if not color:
                return {"error": "Missing 'color' field (hex color like '#ff5500')"}
            palette_type = data.get("type", "complementary")
            return generate_palette(color, palette_type)

        elif service == "uuid_generate":
            count = data.get("count", 1)
            version = data.get("version", 4)
            namespace = data.get("namespace")
            name = data.get("name")
            return generate_uuids(count, version, namespace, name)

        elif service == "encode_decode":
            text = data.get("text", "")
            if not text:
                return {"error": "Missing 'text' field"}
            operation = data.get("operation", "base64_encode")
            return encode_decode(text, operation)

        elif service == "password_generate":
            length = data.get("length", 16)
            count = data.get("count", 1)
            secret_type = data.get("type", "password")
            return generate_password(length, count, secret_type)

        elif service == "jwt_decode":
            token = data.get("token", "")
            if not token:
                return {"error": "Missing 'token' field"}
            return decode_jwt(token)

        elif service == "diff":
            text_a = data.get("text_a", "")
            text_b = data.get("text_b", "")
            if text_a is None and text_b is None:
                return {"error": "Missing 'text_a' and/or 'text_b' fields"}
            context_lines = data.get("context_lines", 3)
            return generate_diff(text_a, text_b, context_lines)

        elif service == "template_render":
            template = data.get("template", "")
            if not template:
                return {"error": "Missing 'template' field"}
            variables = data.get("variables", {})
            return render_template(template, variables)

        elif service == "regex_test":
            pattern = data.get("pattern", "")
            test_string = data.get("text", data.get("test_string", ""))
            if not pattern:
                return {"error": "Missing 'pattern' field"}
            if not test_string:
                return {"error": "Missing 'text' field"}
            flags = data.get("flags", "")
            return test_regex(pattern, test_string, flags)

        elif service == "slug":
            text = data.get("text", "")
            if not text:
                return {"error": "Missing 'text' field"}
            separator = data.get("separator", "-")
            max_length = data.get("max_length", 80)
            return generate_slug(text, separator, max_length)

        elif service == "csv_json":
            csv_text = data.get("csv", data.get("text", ""))
            if not csv_text:
                return {"error": "Missing 'csv' field"}
            delimiter = data.get("delimiter", ",")
            has_header = data.get("has_header", True)
            return csv_to_json(csv_text, delimiter, has_header)

        elif service == "ip_info":
            ip = data.get("ip", "")
            if not ip:
                return {"error": "Missing 'ip' field"}
            return analyze_ip(ip)

        return {"error": f"Service {service} not implemented"}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    server = HTTPServer(('0.0.0.0', PORT), EveServiceHandler)
    print(f"Eve's Utility Services v{VERSION}")
    print(f"Listening on port {PORT}")
    print(f"Services: {', '.join(PRICES.keys())}")
    print(f"Catalog: http://localhost:{PORT}/catalog")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.server_close()


if __name__ == '__main__':
    main()
