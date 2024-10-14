from langchain_openai import OpenAI, ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import BaseTool
from langchain.schema import SystemMessage
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class ContentIdeaInput(BaseModel):
    topic: str = Field(description="The main topic or theme for the social media post")

class ContentIdeaTool(BaseTool):
    name = "content_idea_generator"
    description = "Generate content ideas for a social media post"
    args_schema = ContentIdeaInput

    def _run(self, topic: str):
        llm = OpenAI(temperature=0.7)
        prompt = f"Generate 3 creative content ideas for a social media post about {topic}:"
        return llm(prompt)

class CaptionWriterInput(BaseModel):
    idea: str = Field(description="The content idea to expand into a caption")

class CaptionWriterTool(BaseTool):
    name = "caption_writer"
    description = "Write a caption based on a content idea"
    args_schema = CaptionWriterInput

    def _run(self, idea: str):
        llm = OpenAI(temperature=0.7)
        prompt = f"Write an engaging caption for a social media post based on this idea: {idea}"
        return llm(prompt)

class HashtagGeneratorInput(BaseModel):
    caption: str = Field(description="The caption to generate hashtags for")

class HashtagGeneratorTool(BaseTool):
    name = "hashtag_generator"
    description = "Generate relevant hashtags for a social media post"
    args_schema = HashtagGeneratorInput

    def _run(self, caption: str):
        llm = OpenAI(temperature=0.7)
        prompt = f"Generate 5-7 relevant hashtags for this caption: {caption}"
        return llm(prompt)

class ContentOptimizerInput(BaseModel):
    post: str = Field(description="The full post to optimize, including caption and hashtags")

class ContentOptimizerTool(BaseTool):
    name = "content_optimizer"
    description = "Optimize and format the final social media post"
    args_schema = ContentOptimizerInput

    def _run(self, post: str):
        llm = OpenAI(temperature=0.7)
        prompt = f"Optimize and format this post:\n\n{post}"
        return llm(prompt)

content_idea_tool = ContentIdeaTool()
caption_writer_tool = CaptionWriterTool()
hashtag_generator_tool = HashtagGeneratorTool()
content_optimizer_tool = ContentOptimizerTool()

llm = ChatOpenAI(temperature=0)

def create_agent(tools, system_message):
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_message),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    return create_openai_functions_agent(llm, tools, prompt)

content_idea_agent = AgentExecutor(
    agent=create_agent(
        tools=[content_idea_tool],
        system_message="You are a creative content idea generator."
    ),
    tools=[content_idea_tool],
    verbose=True
)

caption_writer_agent = AgentExecutor(
    agent=create_agent(
        tools=[caption_writer_tool],
        system_message="You are a skilled caption writer."
    ),
    tools=[caption_writer_tool],
    verbose=True
)

hashtag_generator_agent = AgentExecutor(
    agent=create_agent(
        tools=[hashtag_generator_tool],
        system_message="You are an expert hashtag generator."
    ),
    tools=[hashtag_generator_tool],
    verbose=True
)

content_optimizer_agent = AgentExecutor(
    agent=create_agent(
        tools=[content_optimizer_tool],
        system_message="You are a content optimizer."
    ),
    tools=[content_optimizer_tool],
    verbose=True
)

def generate_social_media_post(topic):
    content_ideas = content_idea_agent.invoke({"input": f"Generate content ideas for {topic}"})
    print("Content Ideas:", content_ideas['output'])

    first_idea = content_ideas['output'].split("\n")[0]
    caption = caption_writer_agent.invoke({"input": f"Write a caption for this idea: {first_idea}"})
    print("Caption:", caption['output'])

    hashtags = hashtag_generator_agent.invoke({"input": f"Generate hashtags for this caption: {caption['output']}"})
    print("Hashtags:", hashtags['output'])

    full_post = f"Caption: {caption['output']}\nHashtags: {hashtags['output']}"
    final_post = content_optimizer_agent.invoke({"input": f"Optimize this post:\n{full_post}"})
    print("Final Post:", final_post['output'])

    return final_post['output']

if __name__ == "__main__":
    topic = "sustainable living tips"
    result = generate_social_media_post(topic)
    print("\nFinal Result:")
    print(result)