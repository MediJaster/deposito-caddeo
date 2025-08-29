import os
import pytest

from rag_or_web_flow.tools.calculator_tool import CalculatorTool
from rag_or_web_flow.tools.web_search_tool import WebSearchTool
from rag_or_web_flow.tools.rag_tool import RetrievalTool


class TestCalculatorTool:
    def test_addition(self):
        tool = CalculatorTool()
        assert tool._run("addition", 2.5, 3.5) == 6.0

    def test_subtraction(self):
        tool = CalculatorTool()
        assert tool._run("subtraction", 10.0, 4.0) == 6.0

    def test_multiplication(self):
        tool = CalculatorTool()
        assert tool._run("multiplication", 3.0, 2.0) == 6.0

    def test_division(self):
        tool = CalculatorTool()
        assert tool._run("division", 12.0, 2.0) == 6.0

    def test_division_by_zero(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError):
            tool._run("division", 1.0, 0.0)

    def test_unsupported_operation(self):
        tool = CalculatorTool()
        with pytest.raises(ValueError):
            tool._run("power", 2.0, 3.0)


class TestWebSearchTool:
    def test_search_query_validation(self):
        tool = WebSearchTool()
        with pytest.raises(ValueError):
            tool._run("")

    def test_search_returns_string(self):
        tool = WebSearchTool()
        out = tool._run("example domain")
        assert isinstance(out, str)
        assert out.count("\n") <= 2


class TestRetrievalTool:
    def test_query_validation(self):
        tool = RetrievalTool()
        with pytest.raises(ValueError):
            tool._run("")

    @pytest.mark.skip(reason="Requires running Qdrant and Azure OpenAI env vars set")
    def test_returns_string_when_backend_available(self):
        tool = RetrievalTool()
        assert isinstance(tool._run("test"), str)


