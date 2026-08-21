"""Small provider-facing schemas; trusted metadata is added by Python."""

from __future__ import annotations

from .dataset_memory import SEMANTIC_PROFILE_SCHEMA


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
            "semantic_profile": SEMANTIC_PROFILE_SCHEMA,
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
            "semantic_frame_id", "semantic_profile", "critical_lemma", "critical_word_query", "query",
            "context_sentences", "candidates",
        ],
    },
}


SEMANTIC_ERROR_DIMENSIONS = [
    "none",
    "semantic_content_mismatch",
    "argument_role_error",
    "scope_error",
    "time_or_state_error",
    "internal_contradiction",
    "unnatural_turkish",
    "style_or_length_artifact",
]

MORPH_ERROR_DIMENSIONS = [
    "none",
    "morph_feature_mismatch",
    "wrong_inflection",
    "argument_role_error",
    "scope_error",
    "tense_or_modality_error",
    "possessor_number_error",
    "allomorph_function_error",
    "semantic_content_mismatch",
    "unclear",
]


SEMANTIC_JUDGE_SCHEMA = {
    "name": "blind_semantic_retrieval_judgment",
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
                        "supports_query": {"type": "boolean"},
                        "naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
                        "internally_consistent": {"type": "boolean"},
                        "error_dimensions": {
                            "type": "array",
                            "items": {"type": "string", "enum": SEMANTIC_ERROR_DIMENSIONS},
                            "uniqueItems": True,
                        },
                    },
                    "required": [
                        "id", "supports_query", "naturalness", "internally_consistent",
                        "error_dimensions",
                    ],
                },
            },
            "length_or_style_artifact": {"type": "boolean"},
            "family_naturalness": {"type": "integer", "minimum": 1, "maximum": 5},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "abstain": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": [
            "answers_query", "candidate_assessments", "length_or_style_artifact",
            "family_naturalness", "confidence", "abstain", "notes",
        ],
    },
}


MORPHOLOGY_JUDGE_SCHEMA = {
    "name": "feature_aware_morphology_judgment",
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
                        "morphology_ok": {"type": "boolean"},
                        "target_interpretation": {
                            "type": "string",
                            "enum": ["supports_query", "contradicts_query", "unrelated", "unclear"],
                        },
                        "error_dimensions": {
                            "type": "array",
                            "items": {"type": "string", "enum": MORPH_ERROR_DIMENSIONS},
                            "uniqueItems": True,
                        },
                    },
                    "required": [
                        "id", "morphology_ok", "target_interpretation", "error_dimensions",
                    ],
                },
            },
            "allomorph_treated_as_wrong": {"type": "boolean"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "abstain": {"type": "boolean"},
            "notes": {"type": "string"},
        },
        "required": [
            "answers_query", "candidate_assessments", "allomorph_treated_as_wrong",
            "confidence", "abstain", "notes",
        ],
    },
}


ADJUDICATOR_SCHEMA = {
    "name": "judge_disagreement_advisory",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "recommendation": {
                "type": "string", "enum": ["accept", "reject", "human_review"],
            },
            "answers_query": {"type": "array", "items": {"type": "string"}},
            "morphology_valid": {"type": "boolean"},
            "naturalness_valid": {"type": "boolean"},
            "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
            "reason_codes": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "semantic_disagreement", "order_instability", "morphology_disagreement",
                        "naturalness_disagreement", "low_confidence", "other",
                    ],
                },
                "uniqueItems": True,
            },
            "notes": {"type": "string"},
        },
        "required": [
            "recommendation", "answers_query", "morphology_valid", "naturalness_valid",
            "confidence", "reason_codes", "notes",
        ],
    },
}
