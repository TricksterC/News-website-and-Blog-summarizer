from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Crew
from langchain.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

# Team Member 1 and their task (modified)
Srujan1 = Agent(
    role="Curator",
    goal='Take input from user as a website link, and save the title and text from the article',
    backstory="""You are good at surfing the web, and can easily go through any given website.""",
    verbose=False,
    allow_delegation=True,
    tools=[search_tool]
)

# 1. Prompt for user input
user_link = input("Enter a website link (news article or blog): ")

# 2. Update task description, data, and expected output
task1 = Task(
    description=f"Process the link provided by user: {user_link}",
    agent=Srujan1,
    data={"link": user_link},
    expected_output="A summary of the article, including title (if available), key points, and a bias check."  # Describe expected output
)

# Team Member 2 and their task (corrected)
Srujan2 = Agent(
    role='Summary creator',
    goal='Craft a summary on the text provided.',
    backstory="""You are good with grammar and vocabulary and are very articulate. """,
    verbose=False,
    allow_delegation=True,
)

task2 = Task(
    description="""Use the text provided by curator and create a short summary of maximum 8 lines. 
                 The summary must include the title (if available) and all key points mentioned in the text.""",
    agent=Srujan2,
    expected_output="A concise summary of the extracted text, including the title (if available) and key points."  # Describe expected output
)

# Team Member 3 and their task
Srujan3 = Agent(
    role='Bias verifier',
    goal='To make sure the given summary isnt biased and misleading.',
    backstory="""You are good at verifying and fact checking. """,
    verbose=False,
    allow_delegation=False,
)

task3 = Task(
    description="Make sure summary provided is not biased, and misleading. The content must go through one final check in terms of grammar, factual accuracy etc.",
    agent=Srujan3,
    expected_output="A verified summary, free from bias and factual errors."  # Optional but descriptive
)


# Company
crew = Crew(
    agents=[Srujan1, Srujan2, Srujan3],
    tasks=[task1, task2, task3],
    verbose=2  # Consider adjusting verbosity (0, 1, or 2)
)

# 3. Start CrewAI processing
result = crew.kickoff()
print(result)

