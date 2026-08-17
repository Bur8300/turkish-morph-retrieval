"""Small provider-facing schemas; trusted metadata is added by Python."""

from __future__ import annotations

from .taxonomy import HARD_SUBTYPES


CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "candidate_slot": {"type": "string", "pattern": "^(positive_01|hard_0[1-8]|easy_0[1-2])$"},
        "critical_sentence": {"type": "string", "minLength": 8},
        "critical_word": {"type": "string", "minLength": 1},
    },
    "required": ["candidate_slot", "critical_sentence", "critical_word"],
}


GENERATION_SCHEMA = {
    "name": "turkish_morph_contrast_family",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "semantic_frame_id": {"type": "string"},
            "critical_lemma": {"type": "string", "minLength": 2},
            "critical_word_query": {"type": "string", "minLength": 2},
            "query": {"type": "string", "minLength": 8},
            "context_sentences": {
                "type": "array",
                "maxItems": 3,
                "items": {"type": "string", "minLength": 8},
            },
            "candidates": {
                "type": "array",
                "minItems": 11,
                "maxItems": 11,
                "items": CANDIDATE_SCHEMA,
            },
        },
        "required": [
            "semantic_frame_id", "critical_lemma", "critical_word_query", "query",
            "context_sentences", "candidates",
        ],
    },
}


JUDGE_TYPES = ["positive", *HARD_SUBTYPES, "easy_negative", "unclear"]

JUDGE_SCHEMA = {
    "name": "blind_morph_family_judgment",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "answers_query": {"type": "array", "items": {"type": "string"}},
            "candidate_assessments": {
                "type": "array",
                "minItems": 11,
                "maxItems": 11,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "id": {"type": "string"},
                        "relevance": {"type": "string", "enum": ["relevant", "not_relevant"]},
                        "naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
                        "inferred_type": {"type": "string", "enum": JUDGE_TYPES},
                        "morphology_ok": {"type": "boolean"},
                        "supports_query": {"type": "boolean"},
                        "internally_consistent": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": [
                        "id", "relevance", "naturalness", "inferred_type", "morphology_ok",
                        "supports_query", "internally_consistent", "reason",
                    ],
                },
            },
            "length_or_style_artifact": {"type": "boolean"},
            "allomorph_treated_as_wrong": {"type": "boolean"},
            "family_naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
            "notes": {"type": "string"},
        },
        "required": [
            "answers_query", "candidate_assessments", "length_or_style_artifact",
            "allomorph_treated_as_wrong", "family_naturalness", "notes",
        ],
    },
}
