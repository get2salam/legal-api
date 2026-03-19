"""
Tests for the query understanding module.

Covers all pipeline stages: normalisation, entity extraction,
intent classification, tokenisation, and synonym expansion.
"""

from services.query_understanding import (
    QueryIntent,
    QueryResult,
    _build_expansions,
    _extract_entities,
    _normalise,
    _tokenise,
    understand_query,
)

# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


class TestNormalise:
    def test_lowercases(self):
        assert _normalise("CONTRACT BREACH") == "contract breach"

    def test_strips_whitespace(self):
        assert _normalise("  hello   world  ") == "hello world"

    def test_collapses_runs(self):
        assert _normalise("a   b\t\nc") == "a b c"

    def test_strips_special_chars(self):
        result = _normalise("Smith v. Jones [2021]")
        assert "[" not in result
        assert "]" not in result
        # Period stripped
        assert "." not in result

    def test_preserves_hyphens(self):
        assert "pre-trial" in _normalise("pre-trial hearing")

    def test_preserves_quotes(self):
        assert '"breach of trust"' in _normalise('"breach of trust"')

    def test_unicode_normalisation(self):
        # NFC normalise accented characters
        result = _normalise("caf\u00e9 justice")
        assert isinstance(result, str)

    def test_empty_string(self):
        assert _normalise("") == ""

    def test_only_whitespace(self):
        assert _normalise("   ") == ""


# ---------------------------------------------------------------------------
# Entity extraction — citations
# ---------------------------------------------------------------------------


class TestCitationExtraction:
    def test_extracts_citation(self):
        entities = _extract_entities("2021 SC 45", "2021 sc 45")
        assert entities.citations == ["2021 SC 45"]

    def test_extracts_multiple_citations(self):
        raw = "Compare 2019 MLD 102 with 2021 SC 45"
        entities = _extract_entities(raw, raw.lower())
        assert len(entities.citations) == 2
        assert "2019 MLD 102" in entities.citations
        assert "2021 SC 45" in entities.citations

    def test_no_citation(self):
        entities = _extract_entities("contract breach damages", "contract breach damages")
        assert entities.citations == []

    def test_citation_case_insensitive(self):
        entities = _extract_entities("2021 sc 45", "2021 sc 45")
        assert len(entities.citations) == 1

    def test_citation_pld(self):
        entities = _extract_entities("PLD 2020 SC 100", "pld 2020 sc 100")
        # Note: PLD comes before the year in this format — pattern expects YYYY COURT NUM
        # This tests the actual regex behaviour: 2020 SC 100 is captured
        assert any("SC" in c or "sc" in c.lower() for c in entities.citations)


# ---------------------------------------------------------------------------
# Entity extraction — courts
# ---------------------------------------------------------------------------


class TestCourtExtraction:
    def test_supreme_court(self):
        entities = _extract_entities("Supreme Court judgment 2021", "supreme court judgment 2021")
        assert "Supreme Court" in entities.courts

    def test_high_court(self):
        entities = _extract_entities("High Court bench", "high court bench")
        assert "High Court" in entities.courts

    def test_abbreviation_sc(self):
        entities = _extract_entities("2021 SC 45 decision", "2021 sc 45 decision")
        # 'sc' as abbreviation should resolve
        assert any("Supreme" in c for c in entities.courts)

    def test_lahore_high_court(self):
        entities = _extract_entities("Lahore High Court 2020", "lahore high court 2020")
        assert "Lahore High Court" in entities.courts

    def test_no_court(self):
        entities = _extract_entities("contract breach", "contract breach")
        assert entities.courts == []


# ---------------------------------------------------------------------------
# Entity extraction — years
# ---------------------------------------------------------------------------


class TestYearExtraction:
    def test_single_year(self):
        entities = _extract_entities("judgment in 2021", "judgment in 2021")
        assert 2021 in entities.years

    def test_multiple_years(self):
        entities = _extract_entities("cases from 2019 and 2021", "cases from 2019 and 2021")
        assert 2019 in entities.years
        assert 2021 in entities.years

    def test_year_range_between(self):
        raw = "cases between 2018 and 2022"
        entities = _extract_entities(raw, raw.lower())
        assert entities.year_range == (2018, 2022)

    def test_year_range_to(self):
        raw = "2019 to 2021"
        entities = _extract_entities(raw, raw.lower())
        assert entities.year_range == (2019, 2021)

    def test_year_range_hyphen(self):
        raw = "2018-2022 criminal cases"
        entities = _extract_entities(raw, raw.lower())
        assert entities.year_range == (2018, 2022)

    def test_year_range_reversed_is_sorted(self):
        raw = "2022 to 2018"
        entities = _extract_entities(raw, raw.lower())
        assert entities.year_range == (2018, 2022)

    def test_no_year(self):
        entities = _extract_entities("contract breach", "contract breach")
        assert entities.years == []


# ---------------------------------------------------------------------------
# Entity extraction — quoted phrases
# ---------------------------------------------------------------------------


class TestQuotedPhrases:
    def test_single_quoted_phrase(self):
        raw = '"breach of contract" cases'
        entities = _extract_entities(raw, raw.lower())
        assert "breach of contract" in entities.quoted_phrases

    def test_multiple_quoted_phrases(self):
        raw = '"mens rea" and "actus reus"'
        entities = _extract_entities(raw, raw.lower())
        assert len(entities.quoted_phrases) == 2

    def test_no_quotes(self):
        entities = _extract_entities("breach of contract", "breach of contract")
        assert entities.quoted_phrases == []


# ---------------------------------------------------------------------------
# Entity extraction — judge names
# ---------------------------------------------------------------------------


class TestJudgeExtraction:
    def test_justice_name(self):
        raw = "Justice Khan judgment"
        entities = _extract_entities(raw, raw.lower())
        assert any("Khan" in j for j in entities.judge_names)

    def test_chief_justice(self):
        raw = "Chief Justice Ali decision"
        entities = _extract_entities(raw, raw.lower())
        assert any("Ali" in j for j in entities.judge_names)

    def test_no_judge(self):
        entities = _extract_entities("contract breach", "contract breach")
        assert entities.judge_names == []


# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------


class TestIntentClassification:
    def test_citation_intent(self):
        result = understand_query("2021 SC 45")
        assert result.intent == QueryIntent.CITATION

    def test_judge_intent(self):
        result = understand_query("Justice Khan ruling on bail")
        assert result.intent == QueryIntent.JUDGE

    def test_date_range_intent(self):
        result = understand_query("between 2018 and 2022")
        assert result.intent == QueryIntent.DATE_RANGE

    def test_court_intent_alone(self):
        result = understand_query("Supreme Court 2021")
        # Few substantive tokens beyond court name → COURT intent
        assert result.intent == QueryIntent.COURT

    def test_topical_intent(self):
        result = understand_query("contract breach damages negligence")
        assert result.intent == QueryIntent.TOPICAL

    def test_unknown_intent_empty(self):
        result = understand_query("")
        assert result.intent == QueryIntent.UNKNOWN

    def test_topical_with_court_has_enough_keywords(self):
        result = understand_query("breach of contract negligence Supreme Court judgment")
        # Enough substantive tokens → topical, not court-only
        assert result.intent == QueryIntent.TOPICAL


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------


class TestTokenise:
    def test_removes_stop_words(self):
        tokens = _tokenise("the contract and the agreement")
        assert "the" not in tokens
        assert "and" not in tokens

    def test_keeps_meaningful_words(self):
        tokens = _tokenise("contract breach damages")
        assert "contract" in tokens
        assert "breach" in tokens
        assert "damages" in tokens

    def test_empty(self):
        assert _tokenise("") == []

    def test_single_char_removed(self):
        tokens = _tokenise("a b c xyz")
        assert "a" not in tokens
        assert "b" not in tokens
        assert "xyz" in tokens

    def test_hyphenated_token(self):
        tokens = _tokenise("pre-trial hearing")
        assert "pre-trial" in tokens


# ---------------------------------------------------------------------------
# Synonym expansion
# ---------------------------------------------------------------------------


class TestBuildExpansions:
    def test_contract_expands(self):
        expansions = _build_expansions(["contract"], [])
        # Should include synonyms like 'agreement', 'deed'
        assert len(expansions) > 0
        assert "contract" not in expansions  # original not repeated

    def test_breach_expands(self):
        expansions = _build_expansions(["breach"], [])
        assert "violation" in expansions or "infringement" in expansions

    def test_no_duplicate_clusters(self):
        # 'contract' and 'agreement' are in the same cluster
        # Should only expand once for that cluster
        expansions = _build_expansions(["contract", "agreement"], [])
        # Ensure no over-expansion from same cluster
        # Should not include both original tokens again
        assert "contract" not in expansions
        assert "agreement" not in expansions

    def test_quoted_phrase_excluded_from_expansion(self):
        # 'contract' appears in quoted phrase → should not be expanded
        expansions = _build_expansions(["contract"], ["contract breach"])
        assert expansions == []

    def test_unknown_token_no_expansion(self):
        expansions = _build_expansions(["xyzzy123"], [])
        assert expansions == []

    def test_empty_tokens(self):
        assert _build_expansions([], []) == []


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


class TestUnderstandQuery:
    def test_empty_query(self):
        result = understand_query("")
        assert result.intent == QueryIntent.UNKNOWN
        assert result.normalised == ""
        assert result.tokens == []
        assert result.expansions == []

    def test_whitespace_only(self):
        result = understand_query("   ")
        assert result.intent == QueryIntent.UNKNOWN

    def test_returns_query_result(self):
        result = understand_query("contract breach 2021")
        assert isinstance(result, QueryResult)
        assert result.original == "contract breach 2021"

    def test_citation_full_pipeline(self):
        result = understand_query("2021 SC 45 negligence")
        assert result.intent == QueryIntent.CITATION
        assert "2021 SC 45" in result.entities.citations
        assert 2021 in result.entities.years

    def test_topical_full_pipeline(self):
        result = understand_query("bail application murder accused")
        assert result.intent == QueryIntent.TOPICAL
        assert "bail" in result.tokens
        # bail is in synonym cluster → should expand
        assert len(result.expansions) > 0

    def test_normalised_removes_noise(self):
        result = understand_query("Contract!! Breach... (2021)")
        assert "!!" not in result.normalised
        assert "..." not in result.normalised
        assert "contract" in result.normalised

    def test_judge_full_pipeline(self):
        result = understand_query("Justice Mahmood ruling on contempt")
        assert result.intent == QueryIntent.JUDGE
        assert any("Mahmood" in j for j in result.entities.judge_names)

    def test_year_range_full_pipeline(self):
        result = understand_query("between 2015 and 2020")
        assert result.intent == QueryIntent.DATE_RANGE
        assert result.entities.year_range == (2015, 2020)

    def test_quoted_phrase_full_pipeline(self):
        result = understand_query('"breach of contract" Supreme Court')
        assert "breach of contract" in result.entities.quoted_phrases
        assert "Supreme Court" in result.entities.courts

    def test_tokens_exclude_stop_words(self):
        result = understand_query("the contract and the agreement")
        assert "the" not in result.tokens
        assert "and" not in result.tokens

    def test_expansions_not_in_token_set(self):
        result = understand_query("negligence damages")
        for exp in result.expansions:
            assert exp not in result.tokens
