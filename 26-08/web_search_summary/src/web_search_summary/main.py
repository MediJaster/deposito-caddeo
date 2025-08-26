#!/usr/bin/env python

from pydantic import BaseModel

from crewai.flow import Flow, listen, start

from web_search_summary.crews.web_search_crew.web_search_crew import WebSearchCrew


class WebSearchState(BaseModel):
    search_query: str = ""


class WebSearchFlow(Flow[WebSearchState]):
    @start()
    def get_search_query(self):
        search_query = input("Enter your search query: ")
        self.state.search_query = search_query

        return search_query

    @listen(get_search_query)
    def generate_web_search_summary(self, search_query: str):
        print("Generating web search summary")
        result = WebSearchCrew().crew().kickoff(inputs={"search_query": search_query})

        print("Web Search:", search_query, "\nSummary:\n", result.raw)


def kickoff():
    web_search_flow = WebSearchFlow()
    web_search_flow.kickoff()


def plot():
    web_search_flow = WebSearchFlow()
    web_search_flow.plot()


if __name__ == "__main__":
    kickoff()
