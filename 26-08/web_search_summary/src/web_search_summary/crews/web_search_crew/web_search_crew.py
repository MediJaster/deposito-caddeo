from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List

from crewai_tools import SerperDevTool

# Import your custom tool (cleaner import)

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class WebSearchCrew:
    """Web Search Crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    # If you would lik to add tools to your crew, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def result_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config["result_summarizer"],  # type: ignore[index]
            tools=[
                SerperDevTool(
                    api_key=None,  # Uses the SERPER_API_KEY environment variable
                    n_results=3,  # Number of results to return
                    save_file=False,  # Save results to file
                    search_type="search",  # "search" or "news"
                    max_usage_count=1,
                )
            ],  # Add your custom tool here
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def write_summary(self) -> Task:
        return Task(
            config=self.tasks_config["write_summary"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Summarizer Crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
