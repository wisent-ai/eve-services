#!/usr/bin/env python3
"""Tests for Eve's Utility Services v2.0."""

import json
import math
import sys
import time
import unittest

# Import service functions
sys.path.insert(0, '.')
from service import (
    markdown_to_html, inline_format,
    validate_json_schema,
    generate_hashes,
    analyze_text,
    explain_cron,
    generate_palette, hex_to_rgb, rgb_to_hex,
    generate_uuids,
    encode_decode,
    generate_password, _calculate_entropy, _strength_rating,
    decode_jwt, _base64url_decode,
    generate_diff,
    render_template, _apply_filter, _resolve_variable,
    VERSION, PRICES,
)


class TestMarkdown(unittest.TestCase):
    """Test Markdown to HTML conversion."""

    def test_headings(self):
        self.assertIn('<h1>', markdown_to_html('# Hello'))
        self.assertIn('<h2>', markdown_to_html('## World'))
        self.assertIn('<h3>', markdown_to_html('### Test'))
        self.assertIn('<h6>', markdown_to_html('###### Deep'))

    def test_bold(self):
        result = inline_format('**bold text**')
        self.assertIn('<strong>bold text</strong>', result)

    def test_italic(self):
        result = inline_format('*italic text*')
        self.assertIn('<em>italic text</em>', result)

    def test_bold_italic(self):
        result = inline_format('***bold and italic***')
        self.assertIn('<strong><em>', result)

    def test_inline_code(self):
        result = inline_format('Use `print()` here')
        self.assertIn('<code>print()</code>', result)

    def test_links(self):
        result = inline_format('[Google](https://google.com)')
        self.assertIn('<a href="https://google.com">Google</a>', result)

    def test_images(self):
        result = inline_format('![alt](image.png)')
        self.assertIn('<img src="image.png" alt="alt"/>', result)

    def test_strikethrough(self):
        result = inline_format('~~deleted~~')
        self.assertIn('<del>deleted</del>', result)

    def test_code_blocks(self):
        md = '```python\nprint("hello")\n```'
        result = markdown_to_html(md)
        self.assertIn('<pre><code class="language-python">', result)
        self.assertIn('print(&quot;hello&quot;)', result)

    def test_unordered_list(self):
        md = '- item 1\n- item 2\n- item 3'
        result = markdown_to_html(md)
        self.assertIn('<ul>', result)
        self.assertIn('<li>item 1</li>', result)
        self.assertEqual(result.count('<li>'), 3)

    def test_ordered_list(self):
        md = '1. first\n2. second\n3. third'
        result = markdown_to_html(md)
        self.assertIn('<ol>', result)
        self.assertIn('<li>first</li>', result)

    def test_blockquote(self):
        result = markdown_to_html('> A wise quote')
        self.assertIn('<blockquote>', result)

    def test_horizontal_rule(self):
        self.assertIn('<hr/>', markdown_to_html('---'))
        self.assertIn('<hr/>', markdown_to_html('***'))

    def test_paragraph(self):
        result = markdown_to_html('Just a paragraph.')
        self.assertIn('<p>Just a paragraph.</p>', result)

    def test_empty_input(self):
        result = markdown_to_html('')
        self.assertEqual(result, '')


class TestJsonValidation(unittest.TestCase):
    """Test JSON Schema validation."""

    def test_valid_string(self):
        errors = validate_json_schema("hello", {"type": "string"})
        self.assertEqual(errors, [])

    def test_invalid_type(self):
        errors = validate_json_schema(42, {"type": "string"})
        self.assertTrue(len(errors) > 0)

    def test_string_min_length(self):
        errors = validate_json_schema("hi", {"type": "string", "minLength": 5})
        self.assertTrue(len(errors) > 0)

    def test_string_max_length(self):
        errors = validate_json_schema("hello world", {"type": "string", "maxLength": 5})
        self.assertTrue(len(errors) > 0)

    def test_string_pattern(self):
        errors = validate_json_schema("abc", {"type": "string", "pattern": r"^\d+$"})
        self.assertTrue(len(errors) > 0)
        errors = validate_json_schema("123", {"type": "string", "pattern": r"^\d+$"})
        self.assertEqual(errors, [])

    def test_string_enum(self):
        errors = validate_json_schema("red", {"type": "string", "enum": ["red", "green", "blue"]})
        self.assertEqual(errors, [])
        errors = validate_json_schema("yellow", {"type": "string", "enum": ["red", "green", "blue"]})
        self.assertTrue(len(errors) > 0)

    def test_number_range(self):
        schema = {"type": "number", "minimum": 0, "maximum": 100}
        self.assertEqual(validate_json_schema(50, schema), [])
        self.assertTrue(len(validate_json_schema(-1, schema)) > 0)
        self.assertTrue(len(validate_json_schema(101, schema)) > 0)

    def test_integer_type(self):
        self.assertEqual(validate_json_schema(42, {"type": "integer"}), [])
        self.assertTrue(len(validate_json_schema(3.14, {"type": "integer"})) > 0)

    def test_boolean_type(self):
        self.assertEqual(validate_json_schema(True, {"type": "boolean"}), [])
        self.assertTrue(len(validate_json_schema("true", {"type": "boolean"})) > 0)

    def test_array_type(self):
        self.assertEqual(validate_json_schema([1, 2, 3], {"type": "array", "items": {"type": "integer"}}), [])

    def test_array_min_items(self):
        errors = validate_json_schema([], {"type": "array", "minItems": 1})
        self.assertTrue(len(errors) > 0)

    def test_array_invalid_items(self):
        errors = validate_json_schema([1, "two", 3], {"type": "array", "items": {"type": "integer"}})
        self.assertTrue(len(errors) > 0)

    def test_object_required(self):
        schema = {"type": "object", "required": ["name", "age"], "properties": {
            "name": {"type": "string"}, "age": {"type": "integer"}
        }}
        self.assertEqual(validate_json_schema({"name": "Eve", "age": 1}, schema), [])
        errors = validate_json_schema({"name": "Eve"}, schema)
        self.assertTrue(len(errors) > 0)

    def test_nested_object(self):
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"]
                }
            }
        }
        self.assertEqual(validate_json_schema({"address": {"city": "SF"}}, schema), [])
        errors = validate_json_schema({"address": {}}, schema)
        self.assertTrue(len(errors) > 0)


class TestHashGeneration(unittest.TestCase):
    """Test cryptographic hash generation."""

    def test_sha256(self):
        result = generate_hashes("hello")
        self.assertIn("sha256", result)
        self.assertEqual(len(result["sha256"]), 64)

    def test_md5(self):
        result = generate_hashes("hello")
        self.assertIn("md5", result)
        self.assertEqual(len(result["md5"]), 32)

    def test_sha512(self):
        result = generate_hashes("hello")
        self.assertIn("sha512", result)
        self.assertEqual(len(result["sha512"]), 128)

    def test_specific_algorithms(self):
        result = generate_hashes("test", algorithms=["sha256"])
        self.assertIn("sha256", result)
        self.assertNotIn("md5", result)

    def test_hmac(self):
        result = generate_hashes("hello", hmac_key="secret")
        self.assertIn("hmac_sha256", result)
        self.assertIn("hmac_sha512", result)

    def test_deterministic(self):
        r1 = generate_hashes("same input")
        r2 = generate_hashes("same input")
        self.assertEqual(r1["sha256"], r2["sha256"])

    def test_different_inputs(self):
        r1 = generate_hashes("input1")
        r2 = generate_hashes("input2")
        self.assertNotEqual(r1["sha256"], r2["sha256"])

    def test_input_length(self):
        result = generate_hashes("hello world")
        self.assertEqual(result["input_length"], 11)


class TestTextAnalytics(unittest.TestCase):
    """Test text analytics."""

    def test_basic_stats(self):
        result = analyze_text("Hello world. This is a test.")
        self.assertIn("basic_stats", result)
        self.assertEqual(result["basic_stats"]["words"], 6)
        self.assertEqual(result["basic_stats"]["sentences"], 2)

    def test_readability(self):
        result = analyze_text("The quick brown fox jumps over the lazy dog.")
        self.assertIn("readability", result)
        self.assertIn("flesch_reading_ease", result["readability"])

    def test_vocabulary(self):
        result = analyze_text("the the the cat cat dog")
        self.assertIn("vocabulary", result)
        top = result["vocabulary"]["top_words"]
        self.assertEqual(top[0]["word"], "the")
        self.assertEqual(top[0]["count"], 3)

    def test_sentiment_positive(self):
        result = analyze_text("This is amazing wonderful great excellent fantastic")
        self.assertEqual(result["sentiment"]["label"], "positive")

    def test_sentiment_negative(self):
        result = analyze_text("This is terrible awful horrible bad broken")
        self.assertEqual(result["sentiment"]["label"], "negative")

    def test_lexical_diversity(self):
        result = analyze_text("unique words are all different here today")
        diversity = result["vocabulary"]["lexical_diversity"]
        self.assertGreater(diversity, 0.5)

    def test_reading_time(self):
        # 250 words should be ~1 min
        text = " ".join(["word"] * 250)
        result = analyze_text(text)
        self.assertIn("1.0 min", result["readability"]["estimated_reading_time"])


class TestCronExplainer(unittest.TestCase):
    """Test cron expression explanation."""

    def test_every_minute(self):
        result = explain_cron("* * * * *")
        self.assertIn("summary", result)
        self.assertIn("Every minute", result["summary"])

    def test_midnight(self):
        result = explain_cron("0 0 * * *")
        self.assertIn("midnight", result["summary"].lower())

    def test_specific_time(self):
        result = explain_cron("30 9 * * *")
        self.assertIn("09:30", result["summary"])

    def test_field_count(self):
        result = explain_cron("* * * * *")
        self.assertEqual(result["field_count"], 5)

    def test_six_fields(self):
        result = explain_cron("0 * * * * *")
        self.assertEqual(result["field_count"], 6)

    def test_invalid_cron(self):
        result = explain_cron("* *")
        self.assertIn("error", result)

    def test_day_of_week(self):
        result = explain_cron("0 9 * * 1")
        self.assertIn("Monday", result["summary"])

    def test_month(self):
        result = explain_cron("0 0 1 12 *")
        self.assertIn("December", result["summary"])


class TestColorPalette(unittest.TestCase):
    """Test color palette generation."""

    def test_hex_to_rgb(self):
        self.assertEqual(hex_to_rgb("#ff0000"), (255, 0, 0))
        self.assertEqual(hex_to_rgb("#00ff00"), (0, 255, 0))
        self.assertEqual(hex_to_rgb("0000ff"), (0, 0, 255))

    def test_rgb_to_hex(self):
        self.assertEqual(rgb_to_hex(255, 0, 0), "#ff0000")
        self.assertEqual(rgb_to_hex(0, 255, 0), "#00ff00")

    def test_complementary(self):
        result = generate_palette("#ff0000", "complementary")
        self.assertEqual(len(result["colors"]), 2)
        self.assertIn("css_variables", result)

    def test_triadic(self):
        result = generate_palette("#ff0000", "triadic")
        self.assertEqual(len(result["colors"]), 3)

    def test_analogous(self):
        result = generate_palette("#ff0000", "analogous")
        self.assertEqual(len(result["colors"]), 5)

    def test_monochromatic(self):
        result = generate_palette("#ff0000", "monochromatic")
        self.assertEqual(len(result["colors"]), 5)

    def test_split_complementary(self):
        result = generate_palette("#ff0000", "split_complementary")
        self.assertEqual(len(result["colors"]), 3)

    def test_invalid_color(self):
        result = generate_palette("not-a-color")
        self.assertIn("error", result)

    def test_invalid_type(self):
        result = generate_palette("#ff0000", "rainbow")
        self.assertIn("error", result)

    def test_css_variables(self):
        result = generate_palette("#ff5500", "complementary")
        self.assertIn("--color-0", result["css_variables"])


class TestUuidGenerator(unittest.TestCase):
    """Test UUID generation."""

    def test_single_uuid(self):
        result = generate_uuids(1)
        self.assertEqual(result["count"], 1)
        self.assertEqual(len(result["uuids"]), 1)

    def test_multiple_uuids(self):
        result = generate_uuids(10)
        self.assertEqual(result["count"], 10)
        # All should be unique
        self.assertEqual(len(set(result["uuids"])), 10)

    def test_max_limit(self):
        result = generate_uuids(5000)
        self.assertEqual(result["count"], 1000)

    def test_uuid_format(self):
        result = generate_uuids(1)
        u = result["uuids"][0]
        self.assertEqual(len(u), 36)
        self.assertEqual(u.count('-'), 4)

    def test_uuid5(self):
        result = generate_uuids(1, version=5, namespace="dns", name="example.com")
        self.assertEqual(result["version"], 5)
        # UUID5 is deterministic
        result2 = generate_uuids(1, version=5, namespace="dns", name="example.com")
        self.assertEqual(result["uuids"][0], result2["uuids"][0])

    def test_uuid3(self):
        result = generate_uuids(1, version=3, namespace="dns", name="example.com")
        self.assertEqual(result["version"], 3)


class TestEncodeDecode(unittest.TestCase):
    """Test encoding/decoding operations."""

    def test_base64_encode(self):
        result = encode_decode("hello world", "base64_encode")
        self.assertEqual(result["output"], "aGVsbG8gd29ybGQ=")

    def test_base64_decode(self):
        result = encode_decode("aGVsbG8gd29ybGQ=", "base64_decode")
        self.assertEqual(result["output"], "hello world")

    def test_base64_roundtrip(self):
        encoded = encode_decode("test string", "base64_encode")
        decoded = encode_decode(encoded["output"], "base64_decode")
        self.assertEqual(decoded["output"], "test string")

    def test_url_encode(self):
        result = encode_decode("hello world & foo=bar", "url_encode")
        self.assertNotIn(' ', result["output"])
        self.assertIn('%20', result["output"])

    def test_url_decode(self):
        result = encode_decode("hello%20world", "url_decode")
        self.assertEqual(result["output"], "hello world")

    def test_hex_encode(self):
        result = encode_decode("AB", "hex_encode")
        self.assertEqual(result["output"], "4142")

    def test_hex_decode(self):
        result = encode_decode("4142", "hex_decode")
        self.assertEqual(result["output"], "AB")

    def test_rot13(self):
        result = encode_decode("hello", "rot13")
        self.assertEqual(result["output"], "uryyb")
        # Double ROT13 returns original
        result2 = encode_decode(result["output"], "rot13")
        self.assertEqual(result2["output"], "hello")

    def test_reverse(self):
        result = encode_decode("hello", "reverse")
        self.assertEqual(result["output"], "olleh")

    def test_unknown_operation(self):
        result = encode_decode("test", "unknown_op")
        self.assertIn("error", result)


# ─── Tests for New Services (v2.0) ───────────────────────────────────────────


class TestPasswordGenerator(unittest.TestCase):
    """Test password/secret generation."""

    def test_password_default(self):
        """Test default password generation with correct length and character types."""
        result = generate_password()
        self.assertEqual(result["type"], "password")
        self.assertEqual(result["count"], 1)
        secret = result["secrets"][0]
        self.assertEqual(secret["length"], 16)
        self.assertEqual(len(secret["value"]), 16)
        self.assertIn("entropy_bits", secret)
        self.assertIn("strength", secret)

    def test_password_has_all_char_types(self):
        """Test that generated passwords contain upper, lower, digits, and symbols."""
        # Generate several passwords and verify character diversity
        result = generate_password(length=20, count=5)
        for secret in result["secrets"]:
            pw = secret["value"]
            has_upper = any(c.isupper() for c in pw)
            has_lower = any(c.islower() for c in pw)
            has_digit = any(c.isdigit() for c in pw)
            has_symbol = any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?/`~\'"\\' for c in pw)
            self.assertTrue(has_upper, f"Password missing uppercase: {pw}")
            self.assertTrue(has_lower, f"Password missing lowercase: {pw}")
            self.assertTrue(has_digit, f"Password missing digit: {pw}")
            self.assertTrue(has_symbol, f"Password missing symbol: {pw}")

    def test_password_custom_length(self):
        """Test custom password length."""
        result = generate_password(length=32)
        self.assertEqual(result["secrets"][0]["length"], 32)
        self.assertEqual(len(result["secrets"][0]["value"]), 32)

    def test_password_min_length_clamp(self):
        """Test that password length is clamped to minimum of 4."""
        result = generate_password(length=1)
        self.assertEqual(result["secrets"][0]["length"], 4)

    def test_password_multiple(self):
        """Test generating multiple passwords -- all unique."""
        result = generate_password(count=10)
        self.assertEqual(result["count"], 10)
        values = [s["value"] for s in result["secrets"]]
        # All should be unique (astronomically unlikely to collide)
        self.assertEqual(len(set(values)), 10)

    def test_api_key_generation(self):
        """Test API key generation with sk_live_ prefix."""
        result = generate_password(length=16, secret_type="api_key")
        self.assertEqual(result["type"], "api_key")
        secret = result["secrets"][0]
        self.assertTrue(secret["value"].startswith("sk_live_"))
        self.assertIn("hex_variant", secret)
        self.assertIn("base64_variant", secret)

    def test_passphrase_generation(self):
        """Test passphrase generation with word-based format."""
        result = generate_password(length=4, secret_type="passphrase")
        self.assertEqual(result["type"], "passphrase")
        secret = result["secrets"][0]
        self.assertEqual(secret["word_count"], 4)
        words = secret["value"].split("-")
        self.assertEqual(len(words), 4)
        # Each word should be non-empty
        for word in words:
            self.assertTrue(len(word) > 0)

    def test_passphrase_min_words(self):
        """Test passphrase minimum word count is clamped to 3."""
        result = generate_password(length=1, secret_type="passphrase")
        self.assertEqual(result["secrets"][0]["word_count"], 3)

    def test_pin_generation(self):
        """Test PIN generation -- numeric only."""
        result = generate_password(length=6, secret_type="pin")
        self.assertEqual(result["type"], "pin")
        secret = result["secrets"][0]
        self.assertEqual(secret["length"], 6)
        self.assertTrue(secret["value"].isdigit())
        self.assertEqual(len(secret["value"]), 6)

    def test_pin_default_length(self):
        """Test PIN default and minimum length clamping."""
        result = generate_password(length=2, secret_type="pin")
        # Minimum PIN length is 4
        self.assertEqual(result["secrets"][0]["length"], 4)

    def test_unknown_type(self):
        """Test error on unknown secret type."""
        result = generate_password(secret_type="unknown")
        self.assertIn("error", result)

    def test_entropy_calculation(self):
        """Test entropy calculation helper."""
        # log2(2^8) * 8 = 8 * 8 = 64 -- wait, let's check properly
        # _calculate_entropy(length, charset_size) = length * log2(charset_size)
        entropy = _calculate_entropy(16, 94)  # 94 printable chars
        self.assertGreater(entropy, 100)
        # Zero edge cases
        self.assertEqual(_calculate_entropy(0, 94), 0.0)
        self.assertEqual(_calculate_entropy(16, 0), 0.0)

    def test_strength_rating(self):
        """Test strength rating bands."""
        self.assertEqual(_strength_rating(130), "very_strong")
        self.assertEqual(_strength_rating(90), "strong")
        self.assertEqual(_strength_rating(65), "moderate")
        self.assertEqual(_strength_rating(45), "weak")
        self.assertEqual(_strength_rating(20), "very_weak")

    def test_count_limit(self):
        """Test that count is clamped to 100 max."""
        result = generate_password(count=200)
        self.assertEqual(result["count"], 100)


class TestJwtDecoder(unittest.TestCase):
    """Test JWT token decoding."""

    # A well-known test JWT (HS256, expired)
    # Header: {"alg": "HS256", "typ": "JWT"}
    # Payload: {"sub": "1234567890", "name": "John Doe", "iat": 1516239022}
    TEST_JWT = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIiwiaWF0IjoxNTE2MjM5MDIyfQ.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"

    def test_decode_header(self):
        """Test that JWT header is correctly decoded."""
        result = decode_jwt(self.TEST_JWT)
        self.assertIn("header", result)
        self.assertEqual(result["header"]["alg"], "HS256")
        self.assertEqual(result["header"]["typ"], "JWT")

    def test_decode_payload(self):
        """Test that JWT payload is correctly decoded."""
        result = decode_jwt(self.TEST_JWT)
        self.assertIn("payload", result)
        self.assertEqual(result["payload"]["sub"], "1234567890")
        self.assertEqual(result["payload"]["name"], "John Doe")
        self.assertEqual(result["payload"]["iat"], 1516239022)

    def test_algorithm_and_type(self):
        """Test that algorithm and token type are extracted."""
        result = decode_jwt(self.TEST_JWT)
        self.assertEqual(result["algorithm"], "HS256")
        self.assertEqual(result["token_type"], "JWT")

    def test_signature_present(self):
        """Test signature presence detection."""
        result = decode_jwt(self.TEST_JWT)
        self.assertTrue(result["signature_present"])

    def test_claims_analysis_subject(self):
        """Test that standard claims are analyzed."""
        result = decode_jwt(self.TEST_JWT)
        self.assertIn("claims_analysis", result)
        self.assertEqual(result["claims_analysis"]["subject"], "1234567890")

    def test_claims_analysis_issued_at(self):
        """Test that iat claim is converted to ISO format."""
        result = decode_jwt(self.TEST_JWT)
        self.assertIn("issued_at", result["claims_analysis"])

    def test_no_expiry_claim(self):
        """Test handling of tokens without exp claim."""
        result = decode_jwt(self.TEST_JWT)
        # This test JWT has no exp claim
        self.assertIsNone(result["claims_analysis"]["is_expired"])

    def test_expired_token(self):
        """Test decoding a token with an expired exp claim."""
        import base64 as b64
        # Build a JWT with an expired exp (timestamp 1000000)
        header = b64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b'=').decode()
        payload = b64.urlsafe_b64encode(json.dumps({"sub": "test", "exp": 1000000}).encode()).rstrip(b'=').decode()
        token = f"{header}.{payload}.fakesig"
        result = decode_jwt(token)
        self.assertTrue(result["claims_analysis"]["is_expired"])
        self.assertIn("expired_seconds_ago", result["claims_analysis"])

    def test_invalid_jwt_format(self):
        """Test error handling for invalid JWT format."""
        result = decode_jwt("not.a.valid.jwt.token")
        self.assertIn("error", result)

    def test_invalid_jwt_no_dots(self):
        """Test error handling for JWT without dots."""
        result = decode_jwt("nodots")
        self.assertIn("error", result)

    def test_base64url_padding(self):
        """Test base64url decoding with missing padding."""
        # "hello" base64url encoded is "aGVsbG8" (no padding needed)
        decoded = _base64url_decode("aGVsbG8")
        self.assertEqual(decoded, b"hello")

    def test_raw_parts_present(self):
        """Test that raw parts are included in the result."""
        result = decode_jwt(self.TEST_JWT)
        self.assertIn("raw_parts", result)
        self.assertIn("header", result["raw_parts"])
        self.assertIn("payload", result["raw_parts"])
        self.assertIn("signature", result["raw_parts"])


class TestDiffTool(unittest.TestCase):
    """Test diff/patch generation."""

    def test_identical_texts(self):
        """Test diff of identical texts returns no differences."""
        text = "line 1\nline 2\nline 3"
        result = generate_diff(text, text)
        self.assertFalse(result["has_differences"])
        self.assertEqual(result["stats"]["additions"], 0)
        self.assertEqual(result["stats"]["deletions"], 0)
        self.assertEqual(result["similarity"]["character_ratio"], 1.0)

    def test_completely_different_texts(self):
        """Test diff of completely different texts."""
        result = generate_diff("hello", "world")
        self.assertTrue(result["has_differences"])
        self.assertGreater(result["stats"]["additions"], 0)
        self.assertGreater(result["stats"]["deletions"], 0)
        self.assertLess(result["similarity"]["character_ratio"], 0.5)

    def test_additions_only(self):
        """Test diff where text_b has additional lines."""
        text_a = "line 1\nline 2"
        text_b = "line 1\nline 2\nline 3\nline 4"
        result = generate_diff(text_a, text_b)
        self.assertTrue(result["has_differences"])
        self.assertGreater(result["stats"]["additions"], 0)
        self.assertEqual(result["stats"]["total_lines_a"], 2)
        self.assertEqual(result["stats"]["total_lines_b"], 4)

    def test_deletions_only(self):
        """Test diff where text_b has fewer lines."""
        text_a = "line 1\nline 2\nline 3"
        text_b = "line 1"
        result = generate_diff(text_a, text_b)
        self.assertTrue(result["has_differences"])
        self.assertGreater(result["stats"]["deletions"], 0)

    def test_unified_diff_format(self):
        """Test that the unified diff output has proper format markers."""
        text_a = "old line\n"
        text_b = "new line\n"
        result = generate_diff(text_a, text_b)
        diff = result["unified_diff"]
        self.assertIn("---", diff)
        self.assertIn("+++", diff)
        self.assertIn("-old line", diff)
        self.assertIn("+new line", diff)

    def test_similarity_percentage(self):
        """Test that similarity percentage is formatted correctly."""
        result = generate_diff("abc", "abc")
        self.assertEqual(result["similarity"]["percentage"], "100.0%")

    def test_context_lines_parameter(self):
        """Test that context_lines parameter is respected."""
        result = generate_diff("a\nb\nc", "a\nB\nc", context_lines=0)
        self.assertEqual(result["context_lines"], 0)

    def test_empty_texts(self):
        """Test diff of empty texts."""
        result = generate_diff("", "")
        self.assertFalse(result["has_differences"])
        self.assertEqual(result["similarity"]["character_ratio"], 1.0)

    def test_one_empty_text(self):
        """Test diff where one text is empty."""
        result = generate_diff("", "some content\n")
        self.assertTrue(result["has_differences"])
        self.assertGreater(result["stats"]["additions"], 0)

    def test_line_similarity(self):
        """Test that line-level similarity ratio is provided."""
        text_a = "line 1\nline 2\nline 3"
        text_b = "line 1\nline 2\nline 3"
        result = generate_diff(text_a, text_b)
        self.assertIn("line_ratio", result["similarity"])
        self.assertEqual(result["similarity"]["line_ratio"], 1.0)

    def test_changes_stat(self):
        """Test that changes count is the minimum of additions and deletions."""
        text_a = "old1\nold2\ncommon"
        text_b = "new1\nnew2\ncommon"
        result = generate_diff(text_a, text_b)
        stats = result["stats"]
        self.assertEqual(stats["changes"], min(stats["additions"], stats["deletions"]))


class TestTemplateEngine(unittest.TestCase):
    """Test template rendering engine."""

    def test_simple_variable(self):
        """Test basic variable substitution."""
        result = render_template("Hello, {{name}}!", {"name": "World"})
        self.assertEqual(result["rendered"], "Hello, World!")

    def test_multiple_variables(self):
        """Test multiple variable substitutions."""
        template = "{{greeting}}, {{name}}! Welcome to {{place}}."
        variables = {"greeting": "Hello", "name": "Eve", "place": "Eden"}
        result = render_template(template, variables)
        self.assertEqual(result["rendered"], "Hello, Eve! Welcome to Eden.")

    def test_missing_variable_empty(self):
        """Test that missing variables render as empty strings."""
        result = render_template("Hello, {{name}}!", {})
        self.assertEqual(result["rendered"], "Hello, !")

    def test_filter_upper(self):
        """Test upper filter."""
        result = render_template("{{name|upper}}", {"name": "hello"})
        self.assertEqual(result["rendered"], "HELLO")

    def test_filter_lower(self):
        """Test lower filter."""
        result = render_template("{{name|lower}}", {"name": "HELLO"})
        self.assertEqual(result["rendered"], "hello")

    def test_filter_title(self):
        """Test title filter."""
        result = render_template("{{name|title}}", {"name": "hello world"})
        self.assertEqual(result["rendered"], "Hello World")

    def test_filter_default(self):
        """Test default filter for missing variables."""
        result = render_template('{{missing|default:"N/A"}}', {})
        self.assertEqual(result["rendered"], "N/A")

    def test_filter_default_not_applied_when_present(self):
        """Test that default filter is not applied when variable exists."""
        result = render_template('{{name|default:"N/A"}}', {"name": "Eve"})
        self.assertEqual(result["rendered"], "Eve")

    def test_if_block_truthy(self):
        """Test if block renders body when condition is truthy."""
        template = "{{#if show}}Visible{{/if}}"
        result = render_template(template, {"show": True})
        self.assertEqual(result["rendered"], "Visible")

    def test_if_block_falsy(self):
        """Test if block does not render body when condition is falsy."""
        template = "{{#if show}}Visible{{/if}}"
        result = render_template(template, {"show": False})
        self.assertEqual(result["rendered"], "")

    def test_if_block_missing_var(self):
        """Test if block with missing variable is falsy."""
        template = "Before{{#if missing}}Inside{{/if}}After"
        result = render_template(template, {})
        self.assertEqual(result["rendered"], "BeforeAfter")

    def test_unless_block_falsy(self):
        """Test unless block renders body when condition is falsy."""
        template = "{{#unless hidden}}Shown{{/unless}}"
        result = render_template(template, {"hidden": False})
        self.assertEqual(result["rendered"], "Shown")

    def test_unless_block_truthy(self):
        """Test unless block does not render body when condition is truthy."""
        template = "{{#unless hidden}}Shown{{/unless}}"
        result = render_template(template, {"hidden": True})
        self.assertEqual(result["rendered"], "")

    def test_each_block_with_list(self):
        """Test each block iterates over list items."""
        template = "{{#each items}}{{name}} {{/each}}"
        variables = {"items": [{"name": "a"}, {"name": "b"}, {"name": "c"}]}
        result = render_template(template, variables)
        self.assertEqual(result["rendered"], "a b c ")

    def test_each_block_with_scalars(self):
        """Test each block with scalar (non-dict) items."""
        template = "{{#each items}}{{.}},{{/each}}"
        variables = {"items": [1, 2, 3]}
        result = render_template(template, variables)
        self.assertEqual(result["rendered"], "1,2,3,")

    def test_each_block_empty_list(self):
        """Test each block with empty list renders nothing."""
        template = "{{#each items}}item{{/each}}"
        result = render_template(template, {"items": []})
        self.assertEqual(result["rendered"], "")

    def test_dot_notation(self):
        """Test dot notation for nested variables."""
        result = render_template("{{user.name}}", {"user": {"name": "Eve"}})
        self.assertEqual(result["rendered"], "Eve")

    def test_output_metadata(self):
        """Test that result includes metadata."""
        result = render_template("Hello {{name}}", {"name": "World"})
        self.assertIn("template_length", result)
        self.assertIn("output_length", result)
        self.assertIn("variables_provided", result)
        self.assertIn("name", result["variables_provided"])

    def test_filter_chaining_not_crash(self):
        """Test that applying a filter to a value does not crash."""
        result = render_template("{{name|upper}}", {"name": "test"})
        self.assertEqual(result["rendered"], "TEST")

    def test_complex_template(self):
        """Test a template combining multiple features."""
        template = (
            "{{#if title}}Title: {{title|upper}}\n{{/if}}"
            "Items:\n"
            "{{#each items}}  - {{name}}\n{{/each}}"
            "{{#unless footer}}No footer{{/unless}}"
        )
        variables = {
            "title": "my list",
            "items": [{"name": "alpha"}, {"name": "beta"}],
            "footer": False,
        }
        result = render_template(template, variables)
        rendered = result["rendered"]
        self.assertIn("Title: MY LIST", rendered)
        self.assertIn("  - alpha", rendered)
        self.assertIn("  - beta", rendered)
        self.assertIn("No footer", rendered)


class TestVersionAndPrices(unittest.TestCase):
    """Test version and pricing configuration."""

    def test_version_is_2(self):
        """Test that version is updated to 2.0.0."""
        self.assertEqual(VERSION, "2.0.0")

    def test_twelve_services_in_prices(self):
        """Test that there are 12 services in PRICES dict."""
        self.assertEqual(len(PRICES), 12)

    def test_new_services_have_prices(self):
        """Test that all 4 new services have prices."""
        self.assertIn("password_generate", PRICES)
        self.assertIn("jwt_decode", PRICES)
        self.assertIn("diff", PRICES)
        self.assertIn("template_render", PRICES)

    def test_new_service_prices(self):
        """Test that new service prices are correct."""
        self.assertEqual(PRICES["password_generate"], 0.02)
        self.assertEqual(PRICES["jwt_decode"], 0.03)
        self.assertEqual(PRICES["diff"], 0.05)
        self.assertEqual(PRICES["template_render"], 0.03)


if __name__ == '__main__':
    unittest.main(verbosity=2)
