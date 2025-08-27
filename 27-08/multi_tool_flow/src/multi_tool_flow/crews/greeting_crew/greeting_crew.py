from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent

from ...tools.tool_explanation_output import ToolExplanationOutput

from ...tools.addition_tool import AdditionTool
from ...tools.web_search_tool import WebSearchTool


@CrewBase
class GreetingCrew:
    """GreetingCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def tool_explainer(self) -> Agent:
        return Agent(
            config=self.agents_config["tool_explainer"],  # type: ignore[index]
            output_model=ToolExplanationOutput,
            tools=[AdditionTool(), WebSearchTool()],
        )

    @task
    def tool_explanation_task(self) -> Task:
        return Task(
            config=self.tasks_config["tool_explanation_task"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the GreetingCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
