#!/usr/bin/env python3
"""Tests for Eve's Utility Services v1.0."""

import json
import sys
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
