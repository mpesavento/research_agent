import pytest
from unittest.mock import Mock, patch
from pathlib import Path
from research_agent.workflow import create_market_research_orchestrator
from research_agent.storage import LocalStorageBackend
from langchain_core.messages import AIMessage

# Mock responses for different agents
MOCK_MARKET_TRENDS_RESPONSE = "Market is trending towards digital transformation."
MOCK_COMPETITOR_RESPONSE = "Major competitors include Company A and Company B."
MOCK_CONSUMER_RESPONSE = "Consumers show strong preference for sustainable products."
MOCK_REPORT_RESPONSE = "Final synthesized market research report."

@pytest.fixture
def mock_llm_responses():
    """Mock LLM responses for testing"""
    with patch('research_agent.agents.model') as mock_model:
        # Mock structured output for queries
        mock_model.with_structured_output.return_value.invoke.return_value.queries = [
            "test query 1",
            "test query 2"
        ]

        # Mock regular responses
        mock_model.invoke.return_value = AIMessage(content="Mock response")
        yield mock_model

@pytest.fixture
def mock_search_tool():
    """Mock search tool responses"""
    with patch('research_agent.agents.search_tool') as mock_tool:
        mock_tool.invoke.return_value = [
            {"title": "Test Result", "content": "Test content"}
        ]
        yield mock_tool

@pytest.fixture
def test_storage_dir(tmp_path):
    """Create a temporary directory for test storage"""
    storage_dir = tmp_path / "test_reports"
    storage_dir.mkdir(exist_ok=True)
    return storage_dir

# Unit Tests
@pytest.mark.unit
class TestMarketResearchUnit:
    def test_orchestrator_initialization(self, test_storage_dir):
        """Test basic orchestrator initialization"""
        orchestrator = create_market_research_orchestrator(
            storage_type="local",
            storage_config={"base_dir": str(test_storage_dir)}
        )
        assert orchestrator is not None
        assert isinstance(orchestrator.storage, LocalStorageBackend)
        assert orchestrator.storage.base_dir == Path(test_storage_dir)
        assert orchestrator.storage.base_dir.exists()

    def test_default_orchestrator_initialization(self):
        """Test orchestrator initialization with default settings"""
        orchestrator = create_market_research_orchestrator()
        assert orchestrator is not None
        assert isinstance(orchestrator.storage, LocalStorageBackend)
        assert orchestrator.storage.base_dir.name == "reports"

    def test_empty_query_validation(self):
        """Test that empty queries are rejected"""
        orchestrator = create_market_research_orchestrator()
        with pytest.raises(ValueError, match="Query cannot be empty"):
            orchestrator.run_research("")

# Integration Tests
@pytest.mark.integration
class TestMarketResearchIntegration:
    def test_full_research_workflow(
        self,
        mock_llm_responses,
        mock_search_tool,
        test_storage_dir
    ):
        """Test the complete research workflow with mocked external dependencies"""
        # Setup orchestrator with local storage
        orchestrator = create_market_research_orchestrator(
            storage_type="local",
            storage_config={"base_dir": str(test_storage_dir)}
        )

        # Mock specific responses for different stages
        def mock_invoke_side_effect(messages):
            if "market trends" in str(messages).lower():
                return AIMessage(content=MOCK_MARKET_TRENDS_RESPONSE)
            elif "competitor" in str(messages).lower():
                return AIMessage(content=MOCK_COMPETITOR_RESPONSE)
            elif "consumer" in str(messages).lower():
                return AIMessage(content=MOCK_CONSUMER_RESPONSE)
            else:
                return AIMessage(content=MOCK_REPORT_RESPONSE)

        mock_llm_responses.invoke.side_effect = mock_invoke_side_effect

        # Run research
        query = "Analyze the market for eco-friendly products"
        result = orchestrator.run_research(query)

        # Assertions
        assert result["final_report"] is not None
        assert isinstance(result["final_report"], str)
        assert "report_info" in result
        assert "path" in result["report_info"]
        assert "access_path" in result["report_info"]
        assert Path(result["report_info"]["path"]).exists()
        assert "findings_info" in result
        assert Path(result["findings_info"]["path"]).exists()
        assert isinstance(result["agent_outputs"], dict)

        # Verify files were created in the test directory
        assert test_storage_dir.exists()
        assert len(list(test_storage_dir.glob("*.txt"))) == 2  # Should have report and findings files

    def test_status_callback_integration(self, test_storage_dir):
        """Test that status callbacks are properly called"""
        status_updates = []
        def status_callback(message):
            status_updates.append(message)

        orchestrator = create_market_research_orchestrator(
            storage_type="local",
            storage_config={"base_dir": str(test_storage_dir)},
            status_callback=status_callback
        )

        with patch('research_agent.agents.model') as mock_model, \
             patch('research_agent.agents.search_tool') as mock_search:
            # Setup basic mocks
            mock_model.with_structured_output.return_value.invoke.return_value.queries = ["test"]
            mock_model.invoke.return_value = AIMessage(content="Test response")
            mock_search.invoke.return_value = [{"title": "Test", "content": "Test"}]

            # Run research
            orchestrator.run_research("Test query")

        # Verify status updates were received
        assert len(status_updates) > 0

        # Check for key workflow stages
        expected_prefixes = [
            "🔄 Starting market research workflow",
            "🔄 Executing research workflow",
            "🔍 Starting Market Trends Analysis",
            "✅ Market Trends Analysis complete",
            "🏢 Starting Competitor Analysis",
            "✅ Competitor Analysis complete",
            "👥 Starting Consumer Behavior Analysis",
            "✅ Consumer Analysis complete",
            "📝 Starting Final Report Generation",
            "✅ Final Report Generation complete",
            "💾 Saving research outputs",
            "✅ Research workflow complete"
        ]

        # Check that each expected prefix appears in the updates
        for prefix in expected_prefixes:
            matching_updates = [update for update in status_updates if update.startswith(prefix)]
            assert matching_updates, f"Missing status update starting with: {prefix}"

        # Print status updates for debugging if test fails
        if not all(any(update.startswith(prefix) for update in status_updates)
                  for prefix in expected_prefixes):
            print("\nActual status updates received:")
            for update in status_updates:
                print(f"  {update}")