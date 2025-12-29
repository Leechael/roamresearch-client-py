"""
Tests for search-related functionality in client.py.
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from roamresearch_client_py.client import (
    normalize_tag,
    escape_for_query,
    build_tag_condition,
    build_todo_pattern,
    RoamClient,
)


# ============================================================================
# Pure function tests (no mocking required)
# ============================================================================

class TestNormalizeTag:
    """Tests for normalize_tag() function."""

    def test_hash_tag(self):
        assert normalize_tag("#TODO") == "TODO"

    def test_double_bracket_tag(self):
        assert normalize_tag("[[Project]]") == "Project"

    def test_hash_double_bracket_tag(self):
        assert normalize_tag("#[[My Tag]]") == "My Tag"

    def test_plain_tag(self):
        assert normalize_tag("plain") == "plain"

    def test_tag_with_spaces(self):
        assert normalize_tag("  #TODO  ") == "TODO"

    def test_nested_brackets(self):
        # Edge case: nested brackets
        assert normalize_tag("[[outer [[inner]]]]") == "outer inner"

    def test_multiple_hashes(self):
        assert normalize_tag("##double") == "double"


class TestEscapeForQuery:
    """Tests for escape_for_query() function."""

    def test_no_escape_needed(self):
        assert escape_for_query("hello") == "hello"

    def test_escape_double_quotes(self):
        assert escape_for_query('say "hi"') == 'say \\"hi\\"'

    def test_multiple_quotes(self):
        assert escape_for_query('"a" and "b"') == '\\"a\\" and \\"b\\"'

    def test_empty_string(self):
        assert escape_for_query("") == ""


class TestBuildTagCondition:
    """Tests for build_tag_condition() function."""

    def test_simple_tag(self):
        condition = build_tag_condition("TODO")
        assert '[[TODO]]' in condition
        assert '#TODO ' in condition
        assert '#TODO\\n' in condition
        assert '#[[TODO]]' in condition

    def test_tag_at_end_of_string(self):
        """Tags at end of string should match via ends-with."""
        condition = build_tag_condition("TODO")
        assert 'ends-with?' in condition
        assert 'ends-with? ?s "#TODO"' in condition

    def test_tag_with_quotes_escaped(self):
        condition = build_tag_condition('say "hi"')
        assert '\\"' in condition

    def test_condition_is_or_clause(self):
        condition = build_tag_condition("test")
        assert condition.startswith("(or ")


class TestBuildTodoPattern:
    """Tests for build_todo_pattern() function."""

    def test_todo_pattern(self):
        assert build_todo_pattern("TODO") == "{{[[TODO]]}}"

    def test_done_pattern(self):
        assert build_todo_pattern("DONE") == "{{[[DONE]]}}"

    def test_lowercase_todo(self):
        # Should normalize to uppercase
        assert build_todo_pattern("todo") == "{{[[TODO]]}}"

    def test_mixed_case(self):
        assert build_todo_pattern("ToDo") == "{{[[TODO]]}}"

    def test_invalid_status_raises(self):
        with pytest.raises(ValueError, match="status must be 'TODO' or 'DONE'"):
            build_todo_pattern("PENDING")

    def test_empty_status_raises(self):
        with pytest.raises(ValueError):
            build_todo_pattern("")


# ============================================================================
# Mock tests for RoamClient search methods
# ============================================================================

class TestSearchByTag:
    """Tests for RoamClient.search_by_tag() method."""

    @pytest.fixture
    def mock_client(self):
        """Create a RoamClient with mocked internals."""
        with patch.object(RoamClient, '__init__', lambda self, **kwargs: None):
            client = RoamClient()
            client._client = MagicMock()
            client.api_token = "test"
            client.graph = "test"
            return client

    @pytest.mark.anyio
    async def test_search_by_tag_returns_results(self, mock_client):
        mock_client.q = AsyncMock(return_value=[
            ["uid1", "Block with #TODO tag", "Page1"],
            ["uid2", "Another [[TODO]] block", "Page2"],
        ])

        results = await mock_client.search_by_tag("TODO", limit=10)

        assert len(results) == 2
        assert results[0] == ["uid1", "Block with #TODO tag", "Page1"]

    @pytest.mark.anyio
    async def test_search_by_tag_respects_limit(self, mock_client):
        mock_client.q = AsyncMock(return_value=[
            ["uid1", "text1", "Page1"],
            ["uid2", "text2", "Page1"],
            ["uid3", "text3", "Page1"],
        ])

        results = await mock_client.search_by_tag("tag", limit=2)

        assert len(results) == 2

    @pytest.mark.anyio
    async def test_search_by_tag_empty_results(self, mock_client):
        mock_client.q = AsyncMock(return_value=[])

        results = await mock_client.search_by_tag("nonexistent")

        assert results == []

    @pytest.mark.anyio
    async def test_search_by_tag_normalizes_input(self, mock_client):
        mock_client.q = AsyncMock(return_value=[])

        # All these should normalize to "TODO"
        await mock_client.search_by_tag("#TODO")
        await mock_client.search_by_tag("[[TODO]]")
        await mock_client.search_by_tag("#[[TODO]]")

        # Each call should have been made (we don't check query content here)
        assert mock_client.q.call_count == 3


class TestSearchTodos:
    """Tests for RoamClient.search_todos() method."""

    @pytest.fixture
    def mock_client(self):
        with patch.object(RoamClient, '__init__', lambda self, **kwargs: None):
            client = RoamClient()
            client._client = MagicMock()
            client.api_token = "test"
            client.graph = "test"
            return client

    @pytest.mark.anyio
    async def test_search_todos_returns_results(self, mock_client):
        mock_client.q = AsyncMock(return_value=[
            ["uid1", "{{[[TODO]]}} Buy milk", "Daily"],
            ["uid2", "{{[[TODO]]}} Call mom", "Daily"],
        ])

        results = await mock_client.search_todos(status="TODO")

        assert len(results) == 2
        assert "{{[[TODO]]}}" in results[0][1]

    @pytest.mark.anyio
    async def test_search_todos_done(self, mock_client):
        mock_client.q = AsyncMock(return_value=[
            ["uid1", "{{[[DONE]]}} Completed task", "Daily"],
        ])

        results = await mock_client.search_todos(status="DONE")

        assert len(results) == 1

    @pytest.mark.anyio
    async def test_search_todos_invalid_status(self, mock_client):
        with pytest.raises(ValueError, match="status must be 'TODO' or 'DONE'"):
            await mock_client.search_todos(status="PENDING")

    @pytest.mark.anyio
    async def test_search_todos_case_insensitive_status(self, mock_client):
        mock_client.q = AsyncMock(return_value=[])

        # Should not raise
        await mock_client.search_todos(status="todo")
        await mock_client.search_todos(status="Todo")


class TestFindReferences:
    """Tests for RoamClient.find_references() method."""

    @pytest.fixture
    def mock_client(self):
        with patch.object(RoamClient, '__init__', lambda self, **kwargs: None):
            client = RoamClient()
            client._client = MagicMock()
            client.api_token = "test"
            client.graph = "test"
            return client

    @pytest.mark.anyio
    async def test_find_references_returns_results(self, mock_client):
        mock_client.q = AsyncMock(return_value=[
            ["ref1", "See ((abc123)) for details", "Notes"],
            ["ref2", "Related: ((abc123))", "Projects"],
        ])

        results = await mock_client.find_references("abc123")

        assert len(results) == 2

    @pytest.mark.anyio
    async def test_find_references_respects_limit(self, mock_client):
        mock_client.q = AsyncMock(return_value=[
            ["ref1", "text1", "Page1"],
            ["ref2", "text2", "Page2"],
            ["ref3", "text3", "Page3"],
        ])

        results = await mock_client.find_references("uid", limit=2)

        assert len(results) == 2


class TestFindPageReferences:
    """Tests for RoamClient.find_page_references() method."""

    @pytest.fixture
    def mock_client(self):
        with patch.object(RoamClient, '__init__', lambda self, **kwargs: None):
            client = RoamClient()
            client._client = MagicMock()
            client.api_token = "test"
            client.graph = "test"
            return client

    @pytest.mark.anyio
    async def test_find_page_references_returns_results(self, mock_client):
        mock_client.q = AsyncMock(return_value=[
            ["ref1", "Link to [[Project Notes]]", "Daily"],
            ["ref2", "#[[Project Notes]] is important", "Tasks"],
        ])

        results = await mock_client.find_page_references("Project Notes")

        assert len(results) == 2
