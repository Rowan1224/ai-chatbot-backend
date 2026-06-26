"""
Integration tests for the two-stage duplicate detection pipeline.

Tests real pg_trgm fuzzy matching and pgvector HNSW cosine similarity
against a live pgvector/pgvector:pg16 container.  These are the queries
that cannot be meaningfully tested with mocks.

Stages under test:
  Stage 1 — find_fuzzy_candidates  (pg_trgm trigram similarity)
  Stage 2 — find_similar_requests  (pgvector cosine distance)
  Full pipeline — both stages chained
"""

import uuid
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from src.core.database import (
    find_fuzzy_candidates,
    find_similar_requests,
    save_request,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Unit vectors chosen for predictable cosine similarity:
#   EMBED_A · EMBED_A  = 1.0   (identical — should always pass threshold)
#   EMBED_B · EMBED_B  = 1.0
#   EMBED_A · EMBED_B  ≈ 0.0   (orthogonal — should never pass threshold)
# We use a simple construction: fill all dims with equal values, normalised.

import math  # noqa: E402 (must follow pytestmark assignment)

_DIM = 1536
_SCALE = 1.0 / math.sqrt(_DIM)

EMBED_A = [_SCALE] * _DIM                     # unit vector in +1 direction
EMBED_B = [-_SCALE if i < _DIM // 2 else _SCALE for i in range(_DIM)]  # partially flipped
EMBED_ORTHOGONAL = [-_SCALE] * _DIM           # opposite direction — cosine similarity ≈ -1 (very different)


async def _insert_row(
    pg_client,
    request_type: str,
    embedding: list,
    created_at: datetime | None = None,
) -> str:
    """Insert a row directly, optionally backdating created_at."""
    data = {
        "request_type": request_type,
        "target_environment": "production",
        "business_justification": "Test",
        "name": "Test User",
        "employee_id": "EMP-TEST",
    }
    request_id = await save_request(pg_client, data, embedding)

    if created_at is not None:
        async with pg_client.pool.acquire() as conn:
            await conn.execute(
                "UPDATE requests SET created_at = $1 WHERE id = $2",
                created_at,
                uuid.UUID(request_id),
            )

    return request_id


# ---------------------------------------------------------------------------
# Stage 1 — find_fuzzy_candidates (pg_trgm)
# ---------------------------------------------------------------------------


class TestFindFuzzyCandidates:
    """Stage 1: trigram pre-filter returns candidate UUIDs."""

    @pytest.mark.asyncio
    async def test_exact_match_is_found(self, pg_client, clean_db):
        """A row whose request_type exactly matches the query is returned."""
        request_id = await _insert_row(
            pg_client, "infrastructure-provisioning", EMBED_A
        )

        candidates = await find_fuzzy_candidates(
            pg_client=pg_client,
            request_type="infrastructure-provisioning",
            lookback_days=90,
            fuzzy_threshold=0.3,
        )

        assert request_id in candidates

    @pytest.mark.asyncio
    async def test_near_match_with_typo_is_found(self, pg_client, clean_db):
        """A minor typo in the query still finds the candidate."""
        request_id = await _insert_row(
            pg_client, "infrastructure-provisioning", EMBED_A
        )

        # One character different — trigram similarity should still exceed 0.3
        candidates = await find_fuzzy_candidates(
            pg_client=pg_client,
            request_type="infrastructure-provisionng",   # typo: missing 'i'
            lookback_days=90,
            fuzzy_threshold=0.3,
        )

        assert request_id in candidates

    @pytest.mark.asyncio
    async def test_unrelated_type_is_not_found(self, pg_client, clean_db):
        """A completely different request_type must not appear in candidates."""
        await _insert_row(pg_client, "infrastructure-provisioning", EMBED_A)

        candidates = await find_fuzzy_candidates(
            pg_client=pg_client,
            request_type="access-grant",
            lookback_days=90,
            fuzzy_threshold=0.3,
        )

        # "infrastructure-provisioning" and "access-grant" share very few
        # trigrams — similarity should be well below 0.3
        assert len(candidates) == 0

    @pytest.mark.asyncio
    async def test_old_row_excluded_by_lookback(self, pg_client, clean_db):
        """A row older than lookback_days must NOT appear in candidates."""
        old_date = datetime.utcnow() - timedelta(days=200)
        request_id = await _insert_row(
            pg_client,
            "infrastructure-provisioning",
            EMBED_A,
            created_at=old_date,
        )

        candidates = await find_fuzzy_candidates(
            pg_client=pg_client,
            request_type="infrastructure-provisioning",
            lookback_days=90,   # 90-day window — 200-day-old row must be excluded
            fuzzy_threshold=0.3,
        )

        assert request_id not in candidates

    @pytest.mark.asyncio
    async def test_recent_row_included_within_lookback(self, pg_client, clean_db):
        """A row within the lookback window must be included."""
        recent_date = datetime.utcnow() - timedelta(days=30)
        request_id = await _insert_row(
            pg_client,
            "infrastructure-provisioning",
            EMBED_A,
            created_at=recent_date,
        )

        candidates = await find_fuzzy_candidates(
            pg_client=pg_client,
            request_type="infrastructure-provisioning",
            lookback_days=90,
            fuzzy_threshold=0.3,
        )

        assert request_id in candidates

    @pytest.mark.asyncio
    async def test_multiple_candidates_returned(self, pg_client, clean_db):
        """Multiple matching rows are all returned as candidate IDs."""
        ids = []
        for _ in range(3):
            rid = await _insert_row(
                pg_client, "infrastructure-provisioning", EMBED_A
            )
            ids.append(rid)

        candidates = await find_fuzzy_candidates(
            pg_client=pg_client,
            request_type="infrastructure-provisioning",
            lookback_days=90,
            fuzzy_threshold=0.3,
        )

        for rid in ids:
            assert rid in candidates

    @pytest.mark.asyncio
    async def test_empty_table_returns_empty(self, pg_client, clean_db):
        """No rows → no candidates."""
        candidates = await find_fuzzy_candidates(
            pg_client=pg_client,
            request_type="infrastructure-provisioning",
            lookback_days=90,
        )
        assert candidates == []


# ---------------------------------------------------------------------------
# Stage 2 — find_similar_requests (pgvector cosine similarity)
# ---------------------------------------------------------------------------


class TestFindSimilarRequests:
    """Stage 2: vector search scoped to candidate IDs."""

    @pytest.mark.asyncio
    async def test_identical_embedding_is_found(self, pg_client, clean_db):
        """A row with the same embedding must exceed any reasonable threshold."""
        request_id = await _insert_row(
            pg_client, "infrastructure-provisioning", EMBED_A
        )

        with patch("src.core.database.settings") as mock_settings:
            mock_settings.vector_search_provider = "pgvector"
            mock_settings.duplicate_detection_enabled = True

            results = await find_similar_requests(
                pg_client=pg_client,
                embedding=EMBED_A,
                candidate_ids=[request_id],
                threshold=0.85,
            )

        assert len(results) == 1
        assert results[0]["similarity_score"] >= 0.85

    @pytest.mark.asyncio
    async def test_orthogonal_embedding_is_excluded(self, pg_client, clean_db):
        """A row with a near-zero cosine similarity must NOT be returned."""
        # EMBED_ORTHOGONAL has cosine similarity close to 0 with EMBED_A
        request_id = await _insert_row(
            pg_client, "infrastructure-provisioning", EMBED_ORTHOGONAL
        )

        with patch("src.core.database.settings") as mock_settings:
            mock_settings.vector_search_provider = "pgvector"
            mock_settings.duplicate_detection_enabled = True

            results = await find_similar_requests(
                pg_client=pg_client,
                embedding=EMBED_A,
                candidate_ids=[request_id],
                threshold=0.85,
            )

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_scoped_to_candidate_ids_only(self, pg_client, clean_db):
        """A row NOT in candidate_ids must never be returned."""
        # Insert two rows — only one is in the candidate list
        in_scope_id = await _insert_row(
            pg_client, "infrastructure-provisioning", EMBED_A
        )
        out_of_scope_id = await _insert_row(
            pg_client, "infrastructure-provisioning", EMBED_A
        )

        with patch("src.core.database.settings") as mock_settings:
            mock_settings.vector_search_provider = "pgvector"
            mock_settings.duplicate_detection_enabled = True

            results = await find_similar_requests(
                pg_client=pg_client,
                embedding=EMBED_A,
                candidate_ids=[in_scope_id],   # out_of_scope_id deliberately omitted
                threshold=0.85,
            )

        returned_ids = [r["_id"] for r in results]
        assert out_of_scope_id not in returned_ids

    @pytest.mark.asyncio
    async def test_empty_candidate_ids_returns_empty(self, pg_client, clean_db):
        """Passing an empty candidate list short-circuits without a DB query."""
        await _insert_row(pg_client, "infrastructure-provisioning", EMBED_A)

        with patch("src.core.database.settings") as mock_settings:
            mock_settings.vector_search_provider = "pgvector"
            mock_settings.duplicate_detection_enabled = True

            results = await find_similar_requests(
                pg_client=pg_client,
                embedding=EMBED_A,
                candidate_ids=[],
                threshold=0.85,
            )

        assert results == []

    @pytest.mark.asyncio
    async def test_result_contains_expected_fields(self, pg_client, clean_db):
        """Each result dict must include _id, data, similarity_score, created_at."""
        request_id = await _insert_row(
            pg_client, "service-deployment", EMBED_A
        )

        with patch("src.core.database.settings") as mock_settings:
            mock_settings.vector_search_provider = "pgvector"
            mock_settings.duplicate_detection_enabled = True

            results = await find_similar_requests(
                pg_client=pg_client,
                embedding=EMBED_A,
                candidate_ids=[request_id],
                threshold=0.5,
            )

        assert len(results) == 1
        row = results[0]
        assert "_id" in row
        assert "data" in row
        assert "similarity_score" in row
        assert "created_at" in row
        assert isinstance(row["data"], dict)


# ---------------------------------------------------------------------------
# Full pipeline — Stage 1 then Stage 2 chained
# ---------------------------------------------------------------------------


class TestFullDuplicatePipeline:
    """End-to-end: fuzzy candidates → vector similarity."""

    @pytest.mark.asyncio
    async def test_matching_request_found_end_to_end(self, pg_client, clean_db):
        """
        Seed one matching and one unrelated row.
        Stage 1 must narrow to matching candidates;
        Stage 2 must return the similar row above threshold.
        """
        # Seed: one row we expect to find
        match_id = await _insert_row(
            pg_client, "infrastructure-provisioning", EMBED_A
        )
        # Seed: one unrelated row that should not appear
        await _insert_row(pg_client, "access-grant", EMBED_A)

        # Stage 1 — fuzzy candidates for "infrastructure-provisioning"
        candidates = await find_fuzzy_candidates(
            pg_client=pg_client,
            request_type="infrastructure-provisioning",
            lookback_days=90,
            fuzzy_threshold=0.3,
        )
        assert match_id in candidates

        # Stage 2 — vector search scoped to those candidates
        with patch("src.core.database.settings") as mock_settings:
            mock_settings.vector_search_provider = "pgvector"
            mock_settings.duplicate_detection_enabled = True

            results = await find_similar_requests(
                pg_client=pg_client,
                embedding=EMBED_A,
                candidate_ids=candidates,
                threshold=0.85,
            )

        returned_ids = [r["_id"] for r in results]
        assert match_id in returned_ids

    @pytest.mark.asyncio
    async def test_no_duplicate_when_embeddings_differ(self, pg_client, clean_db):
        """
        If the stored embedding is orthogonal to the query, Stage 2
        must return nothing even when Stage 1 finds a fuzzy candidate.
        """
        # Seed a row with a very different embedding
        await _insert_row(
            pg_client, "infrastructure-provisioning", EMBED_ORTHOGONAL
        )

        candidates = await find_fuzzy_candidates(
            pg_client=pg_client,
            request_type="infrastructure-provisioning",
            lookback_days=90,
            fuzzy_threshold=0.3,
        )
        # Stage 1 finds the row by request_type
        assert len(candidates) >= 1

        with patch("src.core.database.settings") as mock_settings:
            mock_settings.vector_search_provider = "pgvector"
            mock_settings.duplicate_detection_enabled = True

            results = await find_similar_requests(
                pg_client=pg_client,
                embedding=EMBED_A,   # query embedding differs significantly
                candidate_ids=candidates,
                threshold=0.85,
            )

        # Stage 2 rejects it — not a duplicate
        assert results == []


# Made with Bob
