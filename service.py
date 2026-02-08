#!/usr/bin/env python3
"""
Eve's Utility Services v1.0 - Practical developer utility toolkit.

Endpoints:
  POST /markdown       - Markdown to HTML conversion
  POST /json_validate  - JSON Schema validation
  POST /hash           - Cryptographic hash generation (SHA256, SHA512, MD5, HMAC)
  POST /text_analytics - Readability scores, word frequency, text statistics
  POST /cron_explain   - Human-readable cron expression explanations
  POST /color_palette  - Color palette generation from base color
  POST /uuid_generate  - Bulk UUID generation
  POST /encode_decode  - Base64 / URL encoding and decoding
  GET  /health         - Health check
  GET  /catalog        - Service catalog with pricing

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
import statistics
import time
import uuid
import colorsys
import base64
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
VERSION = "1.0.0"

PRICES = {
    "markdown": 0.02,
    "json_validate": 0.03,
    "hash": 0.02,
    "text_analytics": 0.05,
    "cron_explain": 0.02,
    "color_palette": 0.03,
    "uuid_generate": 0.01,
    "encode_decode": 0.01,
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
