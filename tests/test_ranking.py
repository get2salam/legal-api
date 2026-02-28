"""
Tests for BM25 relevance scorer (services/ranking.py).

All tests are synchronous -- no DB or async fixtures needed.
"""

import pytest

from services.ranking import _STOP_WORDS, BM25Scorer, score_results

# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases_input(self):
        tokens = BM25Scorer._tokenize("Hello World COURT")
        assert tokens == ["hello", "world", "court"]

    def test_strips_punctuation(self):
        tokens = BM25Scorer._tokenize("court's ruling, 2024.")
        assert "court" in tokens
        assert "ruling" in tokens
        assert "2024" in tokens

    def test_removes_stop_words(self):
        tokens = BM25Scorer._tokenize("the court of appeal")
        assert "the" not in tokens
        assert "of" not in tokens
        assert "court" in tokens
        assert "appeal" in tokens

    def test_removes_single_character_tokens(self):
        tokens = BM25Scorer._tokenize("a b c defendant")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "c" not in tokens
        assert "defendant" in tokens

    def test_empty_string_returns_empty_list(self):
        assert BM25Scorer._tokenize("") == []

    def test_only_stop_words_returns_empty(self):
        assert BM25Scorer._tokenize("the and is or") == []

    def test_numeric_tokens_are_kept(self):
        tokens = BM25Scorer._tokenize("judgement 2024 order")
        assert "2024" in tokens

    def test_mixed_alphanumeric(self):
        tokens = BM25Scorer._tokenize("section 302 criminal code")
        assert "section" in tokens
        assert "302" in tokens

    def test_stop_words_set_non_empty(self):
        assert len(_STOP_WORDS) >= 20


# ---------------------------------------------------------------------------
# BM25Scorer construction
# ---------------------------------------------------------------------------


CORPUS = [
    "The court found the defendant guilty of fraud.",
    "Appeal dismissed. No new evidence presented.",
    "The defendant appealed against the conviction for fraud.",
    "Contract law: offer, acceptance, and consideration.",
    "Criminal liability requires intent and actus reus.",
]


class TestBM25Init:
    def test_corpus_length_stored(self):
        scorer = BM25Scorer(CORPUS)
        assert scorer._n == len(CORPUS)

    def test_avgdl_positive(self):
        scorer = BM25Scorer(CORPUS)
        assert scorer._avgdl > 0

    def test_empty_corpus_avgdl_zero(self):
        scorer = BM25Scorer([])
        assert scorer._n == 0
        assert scorer._avgdl == 0.0

    def test_invalid_b_raises(self):
        with pytest.raises(ValueError, match="b must be"):
            BM25Scorer(CORPUS, b=1.5)

    def test_negative_b_raises(self):
        with pytest.raises(ValueError, match="b must be"):
            BM25Scorer(CORPUS, b=-0.1)

    def test_negative_k1_raises(self):
        with pytest.raises(ValueError, match="k1 must be"):
            BM25Scorer(CORPUS, k1=-1.0)

    def test_k1_zero_accepted(self):
        scorer = BM25Scorer(CORPUS, k1=0.0)
        assert scorer.k1 == 0.0

    def test_b_boundary_values_accepted(self):
        BM25Scorer(CORPUS, b=0.0)
        BM25Scorer(CORPUS, b=1.0)


# ---------------------------------------------------------------------------
# BM25Scorer.score
# ---------------------------------------------------------------------------


class TestBM25Score:
    def setup_method(self):
        self.scorer = BM25Scorer(CORPUS)

    def test_relevant_doc_scores_higher_than_irrelevant(self):
        # Doc 0 mentions "fraud"; doc 1 does not
        score_relevant = self.scorer.score("fraud", 0)
        score_irrelevant = self.scorer.score("fraud", 1)
        assert score_relevant > score_irrelevant

    def test_completely_irrelevant_doc_scores_zero(self):
        score = self.scorer.score("quantum physics optics", 0)
        assert score == 0.0

    def test_empty_query_returns_zero(self):
        assert self.scorer.score("", 0) == 0.0
        assert self.scorer.score("   ", 0) == 0.0

    def test_stop_word_only_query_returns_zero(self):
        assert self.scorer.score("the and is", 0) == 0.0

    def test_out_of_range_raises_index_error(self):
        with pytest.raises(IndexError):
            self.scorer.score("fraud", 99)

    def test_negative_index_raises_index_error(self):
        with pytest.raises(IndexError):
            self.scorer.score("fraud", -1)

    def test_score_non_negative(self):
        for i in range(len(CORPUS)):
            assert self.scorer.score("defendant", i) >= 0.0

    def test_multi_term_query_higher_than_single_term(self):
        # Doc 2 mentions both "defendant" and "fraud"
        score_multi = self.scorer.score("defendant fraud", 2)
        score_single = self.scorer.score("defendant", 2)
        assert score_multi > score_single

    def test_custom_k1_changes_score(self):
        scorer_default = BM25Scorer(CORPUS)
        scorer_custom = BM25Scorer(CORPUS, k1=0.5)
        s_default = scorer_default.score("fraud", 0)
        s_custom = scorer_custom.score("fraud", 0)
        assert s_default != s_custom

    def test_b_zero_disables_length_normalisation(self):
        scorer = BM25Scorer(CORPUS, b=0.0)
        # Just verify it runs without error and returns a score
        assert scorer.score("fraud", 0) >= 0.0


# ---------------------------------------------------------------------------
# BM25Scorer.rank
# ---------------------------------------------------------------------------


class TestBM25Rank:
    def setup_method(self):
        self.scorer = BM25Scorer(CORPUS)

    def test_rank_returns_descending_order(self):
        ranked = self.scorer.rank("defendant")
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_excludes_zero_scores(self):
        ranked = self.scorer.rank("quantum mechanics")
        assert all(s > 0 for _, s in ranked)

    def test_rank_unknown_term_returns_empty(self):
        assert self.scorer.rank("xyzzy") == []

    def test_rank_indices_are_valid(self):
        ranked = self.scorer.rank("defendant")
        for idx, _ in ranked:
            assert 0 <= idx < len(CORPUS)

    def test_empty_query_returns_empty_rank(self):
        assert self.scorer.rank("") == []

    def test_top_result_is_most_relevant(self):
        ranked = self.scorer.rank("fraud")
        # Docs 0 and 2 both mention "fraud"; one of them must be top-ranked
        top_idx = ranked[0][0]
        assert top_idx in {0, 2}


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestBM25EdgeCases:
    def test_single_document_corpus(self):
        scorer = BM25Scorer(["contract breach damages liability"])
        assert scorer.score("breach", 0) > 0.0
        assert scorer.score("quantum", 0) == 0.0

    def test_all_docs_identical_text(self):
        docs = ["court ruling defendant conviction"] * 4
        scorer = BM25Scorer(docs)
        scores = [scorer.score("defendant", i) for i in range(4)]
        assert scores[0] == pytest.approx(scores[1])
        assert scores[1] == pytest.approx(scores[2])

    def test_length_normalisation_penalises_long_irrelevant_docs(self):
        short_relevant = "fraud conviction"
        long_padded = ("contract consideration damages breach " * 25) + "fraud"
        scorer = BM25Scorer([short_relevant, long_padded])
        score_short = scorer.score("fraud", 0)
        score_long = scorer.score("fraud", 1)
        # BM25 length penalty: short doc with one fraud-mention wins over
        # a very long doc with one fraud-mention buried in irrelevant tokens
        assert score_short > score_long

    def test_tf_saturation_limits_score_growth(self):
        high_tf = "fraud " * 10
        low_tf = "fraud breach contract liability"
        scorer = BM25Scorer([high_tf, low_tf])
        s_high = scorer.score("fraud", 0)
        s_low = scorer.score("fraud", 1)
        # High TF should score higher, but not 10x (saturation kicks in)
        assert s_high > s_low
        assert s_high / s_low < 10.0

    def test_idf_non_negative(self):
        scorer = BM25Scorer(CORPUS)
        for term in ["defendant", "fraud", "court", "appeal"]:
            assert scorer._idf(term) >= 0.0

    def test_rare_term_higher_idf_than_common_term(self):
        # "actus" appears in only 1 doc; "defendant" appears in 2
        scorer = BM25Scorer(CORPUS)
        idf_rare = scorer._idf("actus")
        idf_common = scorer._idf("defendant")
        assert idf_rare > idf_common


# ---------------------------------------------------------------------------
# score_results convenience function
# ---------------------------------------------------------------------------


class TestScoreResults:
    def test_returns_correct_length(self):
        texts = ["fraud conviction", "contract breach", "appeal dismissed"]
        scores = score_results("fraud", texts)
        assert len(scores) == len(texts)

    def test_empty_texts_returns_empty(self):
        assert score_results("fraud", []) == []

    def test_scores_non_negative(self):
        texts = ["fraud appeal", "contract law", "liability damages"]
        scores = score_results("fraud", texts)
        assert all(s >= 0.0 for s in scores)

    def test_matching_doc_scores_highest(self):
        texts = ["fraud conviction defendant", "contract offer acceptance", "tort negligence"]
        scores = score_results("fraud", texts)
        assert scores[0] == max(scores)
