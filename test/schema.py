"""Provider-neutral JSON schemas and candidate vocabularies."""

from __future__ import annotations

from .taxonomy import HARD_SUBTYPES


CANDIDATE_SUBTYPES = ["equivalence_positive", *HARD_SUBTYPES, "easy_negative"]
MORPH_RELATIONS = [
    "target_preserved",
    "allomorph_equivalent",
    "feature_changed",
    "wrong_inflection",
    "same_feature_wrong_content",
    "chain_partial",
    "scope_changed",
    "unrelated",
]


CANDIDATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "role": {"type": "string", "enum": ["positive", "hard_negative", "easy_negative"]},
        "candidate_slot": {"type": "string", "pattern": "^(positive_01|hard_0[1-8]|easy_0[1-2])$"},
        "subtype": {"type": "string", "enum": CANDIDATE_SUBTYPES},
        "critical_sentence": {"type": "string", "minLength": 8},
        "critical_word": {"type": "string", "minLength": 1},
        "morph_relation": {"type": "string", "enum": MORPH_RELATIONS},
        "reason": {"type": "string", "minLength": 3},
    },
    "required": [
        "role", "candidate_slot", "subtype", "critical_sentence", "critical_word",
        "morph_relation", "reason"
    ],
}


GENERATION_SCHEMA = {
    "name": "turkish_morph_contrast_family",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "semantic_frame_id": {"type": "string"},
            "template_id": {"type": "string"},
            "critical_lemma": {"type": "string", "minLength": 2},
            "critical_word_query": {"type": "string", "minLength": 2},
            "critical_word_positive": {"type": "string", "minLength": 2},
            "feature_delta": {"type": "string", "minLength": 3},
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
            "generation_notes": {"type": "string"},
        },
        "required": [
            "semantic_frame_id", "template_id", "critical_lemma", "critical_word_query",
            "critical_word_positive", "feature_delta", "query", "context_sentences",
            "candidates", "generation_notes",
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
                        "relevance": {"type": "string", "enum": ["fully", "partially", "not_relevant"]},
                        "naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
                        "inferred_type": {"type": "string", "enum": JUDGE_TYPES},
                        "morphology_ok": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "relevance", "naturalness", "inferred_type", "morphology_ok", "reason"],
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
