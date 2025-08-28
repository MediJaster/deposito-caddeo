from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List


from ...tools.rag_tool import RetrievalTool


@CrewBase
class RagCrew:
    """RagCrew crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config["researcher"],  # type: ignore[index]
            verbose=True,
            tools=[RetrievalTool()],
        )

    @task
    def rewrite_query_and_perform_search(self) -> Task:
        return Task(
            config=self.tasks_config["rewrite_query_and_perform_search"],  # type: ignore[index]
        )

    @task
    def respond_to_query(self) -> Task:
        return Task(
            config=self.tasks_config["respond_to_query"],  # type: ignore[index]
        )

    @crew
    def crew(self) -> Crew:
        """Creates the RagCrew crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
