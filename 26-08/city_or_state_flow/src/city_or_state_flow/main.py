#!/usr/bin/env python
import json
from typing import Literal

from pydantic import BaseModel, Field

from crewai import LLM
from crewai.flow import Flow, start, router, listen


class LLMChoice(BaseModel):
    choice: str = Field(description="The randomly chosen city or country")
    choice_type: Literal["city", "country"] = Field(
        description="Whether the choice is a city or a country"
    )


class CityOrCountryState(BaseModel):
    choice: LLMChoice = None


class CityOrCountryFlow(Flow[CityOrCountryState]):
    @start()
    def generate_random_choice(self):
        llm = LLM(model="azure/gpt-4.1", temperature=1, response_format=LLMChoice)

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that generates a random choice between a city and a country.",
            },
            {
                "role": "user",
                "content": (
                    "Randomly choose either a city or a country. "
                    "Respond with a JSON object containing the fields 'choice' and 'choice_type'."
                ),
            },
        ]

        # Make the LLM call with JSON response format
        response = llm.call(messages=messages)

        # Parse the JSON response
        random_choice_dict = json.loads(response)
        self.state.choice = LLMChoice(**random_choice_dict)

        return self.state

    @router(generate_random_choice)
    def switch_flow(self, state):
        return state.choice.choice_type

    @listen("city")
    def handle_city(self):
        llm = LLM(model="azure/gpt-4.1")

        # generate a fun fact about the city
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that provides fun facts about cities.",
            },
            {
                "role": "user",
                "content": f"Tell me a fun fact about the city {self.state.choice.choice}.",
            },
        ]

        response = llm.call(messages=messages)
        print(f"Fun fact about {self.state.choice.choice}:\n\n{response}")

    @listen("country")
    def handle_country(self):
        llm = LLM(model="azure/gpt-4.1")

        # return one of the country's neighboring countries
        messages = [
            {
                "role": "system",
                "content": "You are a geography expert that returns a country's neighboring countries.",
            },
            {
                "role": "user",
                "content": f"Tell me a neighboring country of {self.state.choice.choice}.",
            },
        ]

        response = llm.call(messages=messages)
        print(f"Country: {self.state.choice.choice}\nNeighbor: {response}")


def kickoff():
    city_or_country_flow = CityOrCountryFlow()
    city_or_country_flow.kickoff()


def plot():
    city_or_country_flow = CityOrCountryFlow()
    city_or_country_flow.plot()


if __name__ == "__main__":
    kickoff()
