# set environment
import os
from dotenv import load_dotenv
load_dotenv()

# create tools

#websearch tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.pydantic_v1 import BaseModel, Field

internet_search = TavilySearchResults()
internet_search.name = "internet_search"
internet_search.description = "Returns a list of relevant document snippets for a textual query retrieved from the internet."


class TavilySearchInput(BaseModel):
    query: str = Field(description="Query to search the internet with")


internet_search.args_schema = TavilySearchInput

# python repl
from langchain.agents import Tool
from langchain_experimental.utilities import PythonREPL

python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="Executes python code and returns the result. The code runs in a static sandbox without interactive mode, so print output or save output to a file.",
    func=python_repl.run,
)
repl_tool.name = "python_interpreter"

# from langchain_core.pydantic_v1 import BaseModel, Field
class ToolInput(BaseModel):
    code: str = Field(description="Python code to execute.")
repl_tool.args_schema = ToolInput


# RAG

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
#from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS

# Set embeddings
embd = CohereEmbeddings()



# Load PDFs from directory
loader = PyPDFDirectoryLoader("financials/")
docs = loader.load()



# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs)

# Add to vectorstore
vectorstore = FAISS.from_documents(
    documents=doc_splits,
    embedding=embd,
)

vectorstore_retriever = vectorstore.as_retriever()

from langchain.tools.retriever import create_retriever_tool

vectorstore_search = create_retriever_tool(
    retriever=vectorstore_retriever,
    name="vectorstore_search",
    description="Retrieve relevant info from a vectorstore that contains documents related to finanacial data, budget for State of California.",
)



# create agents
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate

# LLM
from langchain_cohere.chat_models import ChatCohere

chat = ChatCohere(model="command-r-plus", temperature=0.6)

# Preamble
preamble = """
Use all tools that are available to answear the question. 
If the query covers the topics of State of California Financial Data, Budget Information, use the vectorstore search first.
You are equipped with an internet search tool, and python interpreter, and a special vectorstore of information about State of California Financial Budget Data Information.

"""

# Prompt
prompt = ChatPromptTemplate.from_template("{input}")

# Create the ReAct agent
agent = create_cohere_react_agent(
    llm=chat,
    tools=[vectorstore_search, internet_search, repl_tool],
    prompt=prompt,
)


agent_executor = AgentExecutor(
    agent=agent, tools=[vectorstore_search, internet_search, repl_tool], verbose=True
)



# test agent
'''
result = agent_executor.invoke(
    {
        "input": "Can you compare department of transporation budget between  22-23 and 23-24 fiscal year in the Budget data from State of California?",
        "preamble": preamble,
    }
)
'''

'''
result = agent_executor.invoke(
    {
        "input": "Can you write an executive summary to compare department of transporation budget between  22-23 and 23-24 fiscal year in the Budget data from State of California?",
        "preamble": preamble,
    }
)
'''

result = agent_executor.invoke(
    {
        "input": "Can you create a bar chart plot and write an executive summary to compare department of transporation budget between  22-23 and 23-24 fiscal year in the Budget data from State of California?",
        "preamble": preamble,
    }
)
