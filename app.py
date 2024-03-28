from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool, WebsiteSearchTool, YoutubeVideoSearchTool, GithubSearchTool
from langchain_community.chat_models import ChatOllama
import os

os.environ['OPENAI_API_KEY'] = 'sk-111111111111111111111111111111111111111111111111'
os.environ['OPENAI_API_BASE'] = 'http://localhost:11434/v1'
os.environ['OPENAI_MODEL_NAME'] = 'llama2'

# Initialize the Ollama language model
# ollama_mistral = ChatOllama(model="dolphin2.2-mistral:7b-q6_K")
ollama_llama2 = ChatOllama(model="llama2")

# Load environment variables from .env file
# load_dotenv()

# Initialize the tools
# internet_search_tool = SerperDevTool()
website_search_tool = WebsiteSearchTool(
    website='https://github.com/karpathy/nanoGPT',
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
            ),
        ),
    )
)
github_search_tool = GithubSearchTool(
    github_repo="https://github.com/karpathy/nanoGPT",
    content_types=["code", "repo"],
    config=dict(
        llm=dict(
            provider="ollama", # or google, openai, anthropic, llama2, ...
            config=dict(
                model="llama2",
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
            ),
        ),
    )
)
youtube_search_tool = YoutubeVideoSearchTool(
    youtube_video_url="https://www.youtube.com/watch?v=kCc8FmEb1nY",
    config=dict(
        llm=dict(
            provider="ollama",
            config=dict(
                model="llama2",
            ),
        ),
        embedder=dict(
            provider="google",
            config=dict(
                model="models/embedding-001",
                task_type="retrieval_document",
            ),
        ),
    )
)

# Define the agents
editor = Agent(
    role="Generative AI Thought Leader and Inventor",
    goal="Drive thought leadership and education in the field of Generative AI focusing clon the work of Andrej Karpathy",
    backstory="In the style of Andrej Karpathy, a thought leader in the field of Generative AI",
    llm=ollama_llama2,
    max_rpm=3,
    max_iter=5,
    allow_delegation=False
)

you_tube_researcher = Agent(
    role="YouTube Researcher",
    goal="Search for and analyze YouTube videos from Andrej Karpathy on nanoGPT, neural networks, and Generative AI",
    backstory="A YouTube and AI enthusiast with a passion for learning and sharing knowledge",
    llm=ollama_llama2,
    tools=[youtube_search_tool],
    max_rpm=3,
    max_iter=5,
    allow_delegation=False
)

github_code_researcher = Agent(
    role="Github Code Researcher",
    goal="Search for and analyze code repositories on Generative AI",
    backstory="A GitHub and AI enthusiast with a passion for exploring code and projects",
    llm=ollama_llama2,
    tools=[github_search_tool, website_search_tool],
    max_rpm=3,
    max_iter=5,
    allow_delegation=False
)

teacher_assistant = Agent(
    role="Teacher Assistant",
    goal="Make Generative AI and Neural Network concepts more accessible to students",
    backstory="An avid writer and educator with a passion for AI education",
    llm=ollama_llama2,
    max_rpm=3,
    max_iter=5,
    allow_delegation=False
)

# Define the tasks
youtube_research_task = Task(
  description="Research and summarize YouTube videos on Generative AI",
  expected_output="structured markdown summarizing the video with the following sections: Title, Video Id, Description, Key Points, and References",
  async_execution=True,
  agent=you_tube_researcher,
  tools=[youtube_search_tool]
)

github_research_task = Task(
  description="Research and summarize code repositories on Generative AI",
  expected_output="structured markdown summarizing the repository that includes following sections: Repository Name, Description, Key Concepts & Features, and References.  leverages a mix of text, code blocks, and mermaid.js models to visualize the architecture of the codebase.",
  async_execution=True,
  agent=github_code_researcher,
  tools=[github_search_tool, website_search_tool]
)

write_research_paper = Task(
  description="Write a research paper on Generative AI concepts",
  expected_output="a well-structured research paper that explains the concepts of Generative AI, Neural Networks, and their applications in a clear and concise manner",
  agent=teacher_assistant,
  context=[youtube_research_task, github_research_task],
  output_file="research_paper.md"
)

# Define the crew
crew = Crew(
    agents=[editor, you_tube_researcher, github_code_researcher, teacher_assistant],
    tasks=[youtube_research_task, github_research_task, write_research_paper],
    process=Process.hierarchical,
    max_iter=15,
    manager_llm=ollama_llama2,
    verbose=2,
    output_format="markdown"
)

# Kick off the crew's work
result = crew.kickoff()

# Print the final result
print(result)