#!/usr/bin/env python
import json

from pydantic import BaseModel

from crewai.flow import Flow, listen, start, router

from multi_tool_flow.crews.greeting_crew.greeting_crew import GreetingCrew

from multi_tool_flow.crews.addition_crew.addition_crew import AdditionCrew
from multi_tool_flow.crews.web_search_crew.web_search_crew import WebSearchCrew

from multi_tool_flow.tools.tool_explanation_output import ToolExplanationOutput


class MultiToolState(BaseModel):
    tool_info: ToolExplanationOutput = None


class MultiToolFlow(Flow[MultiToolState]):
    @start()
    @listen("reselect_tool")
    def select_tool(self):
        user_input = input("Chat with LLM: ").strip()
        result = GreetingCrew().crew().kickoff(inputs={"user_input": user_input})

        json_result = json.loads(result.raw)
        self.state.tool_info = ToolExplanationOutput(**json_result)

        print("Tool explanation:\n", self.state.tool_info.explanation)

        return self.state.tool_info.tool_name

    @router(select_tool)
    def route_tool(self, tool_name: str) -> str:
        return tool_name if tool_name != "" else "reselect_tool"

    @listen("Addition Tool")
    def handle_addition(self):
        addition_to_calculate = input(
            "Enter two numbers to add (e.g., '5 and 3' or 'add 5 and 3'): "
        )

        result = (
            AdditionCrew().crew().kickoff(inputs={"user_input": addition_to_calculate})
        )

        print("Addition:", addition_to_calculate, "\nResult:\n", result.raw)

    @listen("Web Search Tool")
    def handle_web_search(self):
        search_query = input("Enter your search query: ")

        result = WebSearchCrew().crew().kickoff(inputs={"search_query": search_query})

        print("Web Search:", search_query, "\nSummary:\n", result.raw)


def kickoff():
    multi_tool_flow = MultiToolFlow()
    multi_tool_flow.kickoff()


def plot():
    multi_tool_flow = MultiToolFlow()
    multi_tool_flow.plot()


if __name__ == "__main__":
    kickoff()
