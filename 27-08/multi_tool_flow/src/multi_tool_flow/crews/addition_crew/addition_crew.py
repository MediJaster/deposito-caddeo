from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from ...tools.addition_tool import AdditionTool


@CrewBase
class AdditionCrew:
    """AdditionCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def addition_calculator(self) -> Agent:
        return Agent(
            config=self.agents_config["addition_calculator"],  # type: ignore[index]
            tools=[AdditionTool()],  # type: ignore[name-defined]
        )

    @task
    def perform_addition(self) -> Task:
        return Task(
            config=self.tasks_config["perform_addition"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AdditionCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
