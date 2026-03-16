"""
Tests for field-weighted BM25 scorer (services/field_ranking.py).

Covers:
- FieldWeights validation
- _FieldIndex scoring correctness
- FieldBM25Scorer score / rank
- Field-boost behaviour (title/citation outweigh text)
- score_cases convenience helper
- Edge cases: empty corpus, missing fields, empty query, OOB index
"""

import pytest

from services.field_ranking import FieldBM25Scorer, FieldWeights, _tokenize, score_cases

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Five realistic-ish legal case dicts spanning multiple fields
DOCS = [
    {
        "title": "Smith v Jones",
        "citation": "2021 SC 12",
        "headnote": "Contract breach of duty; damages awarded.",
        "text": "The plaintiff alleged that the defendant breached the contract.",
    },
    {
        "title": "Crown v Allen",
        "citation": "2019 CrLJ 88",
        "headnote": "Criminal liability; mens rea not established.",
        "text": "Prosecution failed to prove intent beyond reasonable doubt.",
    },
    {
        "title": "Baker Industries Ltd v Harper",
        "citation": "2022 SC 45",
        "headnote": "Contract formation; offer and acceptance analysis.",
        "text": "The court examined offer, acceptance, and consideration for the contract.",
    },
    {
        "title": "Reynolds Negligence Appeal",
        "citation": "2020 AC 7",
        "headnote": "Tort law; negligence and duty of care.",
        "text": "Duty of care was not established on the facts presented.",
    },
    {
        "title": "Public Interest Litigation",
        "citation": "2023 PIL 1",
        "headnote": "Constitutional rights; fundamental freedoms.",
        "text": "The petition invoked constitutional guarantees against arbitrary detention.",
    },
]


# ---------------------------------------------------------------------------
# _tokenize
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_lowercases(self):
        assert "smith" in _tokenize("Smith")

    def test_removes_stop_words(self):
        tokens = _tokenize("the court of appeal")
        assert "the" not in tokens
        assert "of" not in tokens
        assert "court" in tokens

    def test_removes_single_chars(self):
        assert "a" not in _tokenize("a b c contract")
        assert "contract" in _tokenize("a b c contract")

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_numeric_tokens_kept(self):
        assert "2021" in _tokenize("judgment 2021")

    def test_alphanumeric_mixed(self):
        tokens = _tokenize("section 302")
        assert "section" in tokens
        assert "302" in tokens


# ---------------------------------------------------------------------------
# FieldWeights
# ---------------------------------------------------------------------------


class TestFieldWeights:
    def test_defaults_are_positive(self):
        w = FieldWeights()
        assert w.title > 0
        assert w.citation > 0
        assert w.headnote > 0
        assert w.text > 0

    def test_citation_default_highest(self):
        w = FieldWeights()
        assert w.citation >= w.title >= w.headnote >= w.text

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="FieldWeights.title"):
            FieldWeights(title=-1.0)

    def test_negative_citation_raises(self):
        with pytest.raises(ValueError, match="FieldWeights.citation"):
            FieldWeights(citation=-0.5)

    def test_zero_weight_allowed(self):
        w = FieldWeights(text=0.0)
        assert w.text == 0.0

    def test_as_dict_has_all_fields(self):
        w = FieldWeights()
        d = w.as_dict()
        for field in ("title", "citation", "headnote", "text"):
            assert field in d

    def test_custom_weights(self):
        w = FieldWeights(title=10.0, citation=1.0, headnote=1.0, text=1.0)
        assert w.title == 10.0

    def test_all_zero_allowed(self):
        # Degenerate but should not error at construction
        w = FieldWeights(title=0.0, citation=0.0, headnote=0.0, text=0.0)
        assert w.as_dict() == {"title": 0.0, "citation": 0.0, "headnote": 0.0, "text": 0.0}


# ---------------------------------------------------------------------------
# FieldBM25Scorer — construction
# ---------------------------------------------------------------------------


class TestFieldBM25Init:
    def test_num_documents(self):
        scorer = FieldBM25Scorer(DOCS)
        assert scorer.num_documents == len(DOCS)

    def test_empty_corpus(self):
        scorer = FieldBM25Scorer([])
        assert scorer.num_documents == 0

    def test_invalid_b_raises(self):
        with pytest.raises(ValueError, match="b must be"):
            FieldBM25Scorer(DOCS, b=1.5)

    def test_negative_b_raises(self):
        with pytest.raises(ValueError, match="b must be"):
            FieldBM25Scorer(DOCS, b=-0.1)

    def test_negative_k1_raises(self):
        with pytest.raises(ValueError, match="k1 must be"):
            FieldBM25Scorer(DOCS, k1=-1.0)

    def test_b_boundary_0_accepted(self):
        FieldBM25Scorer(DOCS, b=0.0)

    def test_b_boundary_1_accepted(self):
        FieldBM25Scorer(DOCS, b=1.0)

    def test_k1_zero_accepted(self):
        scorer = FieldBM25Scorer(DOCS, k1=0.0)
        assert scorer.num_documents == len(DOCS)

    def test_custom_weights_stored(self):
        w = FieldWeights(title=5.0)
        scorer = FieldBM25Scorer(DOCS, weights=w)
        assert scorer._weights.title == 5.0

    def test_missing_fields_default_to_empty(self):
        docs = [{"title": "contract breach"}, {"title": "tort liability"}]
        scorer = FieldBM25Scorer(docs)
        # Should not raise even though citation/headnote/text are absent
        assert scorer.score("contract", 0) >= 0.0


# ---------------------------------------------------------------------------
# FieldBM25Scorer — score
# ---------------------------------------------------------------------------


class TestFieldBM25Score:
    def setup_method(self):
        self.scorer = FieldBM25Scorer(DOCS)

    def test_score_non_negative(self):
        for i in range(len(DOCS)):
            assert self.scorer.score("contract", i) >= 0.0

    def test_contract_query_matches_contract_docs(self):
        # Docs 0, 2 are contract-related; docs 1, 3 are not
        s_contract = self.scorer.score("contract", 0)
        s_criminal = self.scorer.score("contract", 1)
        assert s_contract > s_criminal

    def test_empty_query_returns_zero(self):
        assert self.scorer.score("", 0) == 0.0
        assert self.scorer.score("   ", 0) == 0.0

    def test_stop_word_only_query_returns_zero(self):
        assert self.scorer.score("the and is", 0) == 0.0

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError):
            self.scorer.score("contract", 99)

    def test_negative_index_raises(self):
        with pytest.raises(IndexError):
            self.scorer.score("contract", -1)

    def test_empty_corpus_score_always_zero(self):
        empty = FieldBM25Scorer([])
        # No IndexError expected since _n == 0 but range guard fires
        with pytest.raises(IndexError):
            empty.score("contract", 0)

    def test_unknown_term_scores_zero(self):
        assert self.scorer.score("xyzzy quantum optics", 0) == 0.0

    def test_multi_term_higher_than_single(self):
        s_multi = self.scorer.score("contract breach", 0)
        s_single = self.scorer.score("contract", 0)
        assert s_multi >= s_single


# ---------------------------------------------------------------------------
# FieldBM25Scorer — citation / title boost
# ---------------------------------------------------------------------------


class TestFieldBoosts:
    """Verify that high-boost fields (citation, title) produce higher scores
    than when those fields are zeroed out."""

    def test_citation_match_boosts_score(self):
        """Exact citation match should score higher than a text-only match."""
        # Doc 0 has citation "2021 SC 12"; doc 3 has citation "2020 AC 7"
        # Query for "2021" should benefit from citation boost on doc 0
        default_scorer = FieldBM25Scorer(DOCS)
        no_citation_scorer = FieldBM25Scorer(DOCS, weights=FieldWeights(citation=0.0))

        score_default = default_scorer.score("2021", 0)
        score_no_cit = no_citation_scorer.score("2021", 0)

        # With citation boost enabled, score for a citation-matching doc is higher
        assert score_default > score_no_cit

    def test_title_match_boosts_score(self):
        """Title match should score higher when title boost is active."""
        default_scorer = FieldBM25Scorer(DOCS)
        no_title_scorer = FieldBM25Scorer(DOCS, weights=FieldWeights(title=0.0))

        # "Smith" appears only in title of doc 0
        score_with_title = default_scorer.score("Smith", 0)
        score_no_title = no_title_scorer.score("Smith", 0)
        assert score_with_title > score_no_title

    def test_text_only_match_scores_lower_than_title_match(self):
        """A term appearing only in the long text field should score lower
        than the same term appearing in the short title field (all else equal)."""
        docs = [
            {"title": "negligence", "citation": "", "headnote": "", "text": "unrelated words here"},
            {
                "title": "unrelated case",
                "citation": "",
                "headnote": "",
                "text": "negligence discussed",
            },
        ]
        scorer = FieldBM25Scorer(docs)
        # Doc 0: "negligence" in title (high boost)
        # Doc 1: "negligence" in text (low boost)
        assert scorer.score("negligence", 0) > scorer.score("negligence", 1)

    def test_all_zero_weights_always_return_zero(self):
        w = FieldWeights(title=0.0, citation=0.0, headnote=0.0, text=0.0)
        scorer = FieldBM25Scorer(DOCS, weights=w)
        for i in range(len(DOCS)):
            assert scorer.score("contract", i) == 0.0


# ---------------------------------------------------------------------------
# FieldBM25Scorer — rank
# ---------------------------------------------------------------------------


class TestFieldBM25Rank:
    def setup_method(self):
        self.scorer = FieldBM25Scorer(DOCS)

    def test_rank_returns_descending_scores(self):
        ranked = self.scorer.rank("contract")
        scores = [s for _, s in ranked]
        assert scores == sorted(scores, reverse=True)

    def test_rank_excludes_zero_scores(self):
        ranked = self.scorer.rank("xyzzy")
        assert ranked == []

    def test_rank_indices_in_valid_range(self):
        ranked = self.scorer.rank("contract")
        for idx, _ in ranked:
            assert 0 <= idx < len(DOCS)

    def test_rank_contract_top_is_contract_doc(self):
        ranked = self.scorer.rank("contract")
        top_idx = ranked[0][0]
        # Docs 0, 2 are the contract-related ones
        assert top_idx in {0, 2}

    def test_empty_query_returns_empty_rank(self):
        assert self.scorer.rank("") == []

    def test_empty_corpus_rank_empty(self):
        scorer = FieldBM25Scorer([])
        assert scorer.rank("contract") == []

    def test_single_doc_corpus_rank(self):
        scorer = FieldBM25Scorer([DOCS[0]])
        ranked = scorer.rank("contract")
        assert len(ranked) == 1
        assert ranked[0][0] == 0

    def test_rank_no_duplicates(self):
        ranked = self.scorer.rank("liability")
        indices = [idx for idx, _ in ranked]
        assert len(indices) == len(set(indices))


# ---------------------------------------------------------------------------
# score_cases convenience helper
# ---------------------------------------------------------------------------


class TestScoreCases:
    """Tests using simple namespace objects to mimic ORM Case instances."""

    def _make_case(self, title="", citation="", headnote="", text=""):
        class _Case:
            pass

        c = _Case()
        c.title = title
        c.citation = citation
        c.headnote = headnote
        c.text = text
        return c

    def test_returns_correct_length(self):
        cases = [self._make_case(title=d["title"], text=d["text"]) for d in DOCS]
        scores = score_cases("contract", cases)
        assert len(scores) == len(cases)

    def test_empty_cases_returns_empty(self):
        assert score_cases("contract", []) == []

    def test_empty_query_returns_ones(self):
        cases = [self._make_case(title="test") for _ in range(3)]
        scores = score_cases("", cases)
        assert scores == [1.0, 1.0, 1.0]

    def test_scores_in_unit_interval(self):
        cases = [self._make_case(title=d["title"], text=d["text"]) for d in DOCS]
        scores = score_cases("contract", cases)
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_top_score_is_one(self):
        cases = [self._make_case(title=d["title"], text=d["text"]) for d in DOCS]
        scores = score_cases("contract", cases)
        assert max(scores) == pytest.approx(1.0)

    def test_matching_case_scores_highest(self):
        cases = [
            self._make_case(title="Contract Breach Smith", text="breach of contract"),
            self._make_case(title="Negligence Appeal", text="duty of care"),
            self._make_case(title="Criminal Intent Case", text="mens rea"),
        ]
        scores = score_cases("contract breach", cases)
        assert scores[0] == max(scores)

    def test_custom_weights_applied(self):
        cases = [
            self._make_case(citation="contract citation match"),
            self._make_case(text="contract appears in text only"),
        ]
        heavy_cit = FieldWeights(title=1.0, citation=100.0, headnote=1.0, text=1.0)
        scores = score_cases("contract", cases, weights=heavy_cit)
        # Citation-matching doc should dominate with heavy citation boost
        assert scores[0] > scores[1]

    def test_unknown_term_all_zeros(self):
        cases = [self._make_case(title="contract law"), self._make_case(title="tort negligence")]
        scores = score_cases("xyzzy_unknown_term", cases)
        assert all(s == 0.0 for s in scores)

    def test_missing_attributes_handled_gracefully(self):
        """Cases without all attributes should not raise."""

        class _BareCase:
            title = "contract dispute"
            # no citation, headnote, text attrs

        scores = score_cases("contract", [_BareCase()])
        assert len(scores) == 1
        assert scores[0] >= 0.0


# ---------------------------------------------------------------------------
# Edge / regression
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_single_document_all_fields(self):
        scorer = FieldBM25Scorer([DOCS[0]])
        assert scorer.score("Smith", 0) > 0.0

    def test_document_with_only_empty_strings(self):
        scorer = FieldBM25Scorer([{"title": "", "citation": "", "headnote": "", "text": ""}])
        assert scorer.score("contract", 0) == 0.0

    def test_very_long_text_does_not_bury_short_title(self):
        """BM25 length normalisation should prevent a 500-word text field
        from outscoring a 2-word title on a title-matching query."""
        padding = "irrelevant filler words " * 100
        docs = [
            {"title": "fraud conviction", "citation": "", "headnote": "", "text": padding},
            {"title": "unrelated case", "citation": "", "headnote": "", "text": padding + " fraud"},
        ]
        scorer = FieldBM25Scorer(docs)
        # Doc 0: "fraud" in title (high boost + short field)
        # Doc 1: "fraud" buried deep in text (low boost + long field)
        assert scorer.score("fraud", 0) > scorer.score("fraud", 1)

    def test_reproducible_scores(self):
        """Same scorer returns identical scores on repeated calls."""
        scorer = FieldBM25Scorer(DOCS)
        s1 = scorer.score("contract", 0)
        s2 = scorer.score("contract", 0)
        assert s1 == pytest.approx(s2)

    def test_large_corpus_does_not_raise(self):
        """Scorer should handle 1000-doc corpus without error."""
        docs = [
            {"title": f"Case {i}", "citation": f"2020 SC {i}", "headnote": "test", "text": "body"}
            for i in range(1000)
        ]
        scorer = FieldBM25Scorer(docs)
        ranked = scorer.rank("Case 500")
        assert len(ranked) > 0
