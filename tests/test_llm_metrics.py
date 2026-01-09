import pytest
from rag_tools.metrics.llm_metrics import (
    compute_llm_metrics,
    calculate_bleu_score,
    calculate_chrf_score,
    calculate_ter_score,
)

# Простые тестовые строки
REFERENCE = "The quick brown fox jumps over the lazy dog"
CANDIDATE_SAME = "The quick brown fox jumps over the lazy dog"
CANDIDATE_DIFFERENT = "Другая строка для теста"


def test_bleu_perfect_match():
    score = calculate_bleu_score(REFERENCE, CANDIDATE_SAME)
    assert score > 0, "BLEU должен быть положительным для идентичного текста"


def test_chrf_perfect_match():
    score = calculate_chrf_score(REFERENCE, CANDIDATE_SAME)
    assert score == 100, "CHRF должен быть 100 для идентичного текста"


def test_ter_perfect_match():
    score = calculate_ter_score(REFERENCE, CANDIDATE_SAME)
    assert score == 0, "TER должен быть 0 для идентичного текста"


def test_compute_llm_metrics_identical():
    results = compute_llm_metrics(REFERENCE, CANDIDATE_SAME)
    assert results["BLEU"] > 0
    assert results["CHRF"] == 100
    assert results["TER"] == 0
    assert "BLEU" in results and "CHRF" in results and "TER" in results


def test_compute_llm_metrics_different():
    results = compute_llm_metrics(REFERENCE, CANDIDATE_DIFFERENT)
    assert results["CHRF"] < 100
    assert results["TER"] > 0
