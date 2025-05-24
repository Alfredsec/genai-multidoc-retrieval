#!/usr/bin/env python
# coding: utf-8

# In[1]:


from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()


# In[2]:


import nest_asyncio
nest_asyncio.apply()


# In[3]:


urls = [
    "https://openreview.net/pdf?id=VtmBAGCN7o",
    "https://openreview.net/pdf?id=6PmJoRfdaK",
    "https://openreview.net/pdf?id=hSyW5go0v8",
]

papers = [
    "metagpt.pdf",
    "longlora.pdf",
    "selfrag.pdf",
]


# In[4]:


from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]


# In[5]:


initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]


# In[6]:


from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")


# In[7]:


len(initial_tools)


# In[8]:


from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)


# In[9]:


response = agent.query(
    "Tell me about the evaluation dataset used in LongLoRA, "
    "and then tell me about the evaluation results"
)


# In[10]:


response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
print(str(response))


# In[ ]:




