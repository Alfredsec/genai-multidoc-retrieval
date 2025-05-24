## Design and Implementation of a Multidocument Retrieval Agent Using LlamaIndex

### AIM:
To design and implement a multidocument retrieval agent using LlamaIndex to extract and synthesize information from multiple research articles, and to evaluate its performance by testing it with diverse queries, analyzing its ability to deliver concise, relevant, and accurate responses.

### PROBLEM STATEMENT:
With the exponential growth of scientific literature, manually extracting and comparing insights from multiple research papers is time-consuming and inefficient. There is a need for an intelligent agent that can automatically retrieve, summarize, and synthesize information from multiple documents. This project aims to address this challenge by implementing a multidocument retrieval agent using LlamaIndex, capable of handling natural language queries and providing accurate and concise responses from a collection of research papers.

### DESIGN STEPS:

#### STEP 1:
Collect and prepare research papers in PDF format. Download relevant papers such as MetaGPT, LongLoRA, and Self-RAG, and save them locally.

#### STEP 2:
Use get_doc_tools from a utility script to create vector-based retrieval and summarization tools for each paper using LlamaIndex. These tools enable semantic search and summarization functionalities.

#### STEP 3:
Initialize an agent using the FunctionCallingAgentWorker and AgentRunner classes from LlamaIndex and print the response for your query.

### PROGRAM:
```python
from helper import get_openai_api_key
OPENAI_API_KEY = get_openai_api_key()
import nest_asyncio
nest_asyncio.apply()
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
from utils import get_doc_tools
from pathlib import Path

paper_to_tools_dict = {}
for paper in papers:
    print(f"Getting tools for paper: {paper}")
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]
initial_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
from llama_index.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
len(initial_tools)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.agent import AgentRunner

agent_worker = FunctionCallingAgentWorker.from_tools(
    initial_tools, 
    llm=llm, 
    verbose=True
)
agent = AgentRunner(agent_worker)
response = agent.query("Give me a summary of both Self-RAG and LongLoRA")
print(str(response))
```
### OUTPUT:

![image](https://github.com/user-attachments/assets/e056b306-6820-42e4-af4f-882c5990ccf2)


### RESULT:
A multidocument retrieval agent was successfully designed and implemented using LlamaIndex. The system was able to extract and synthesize information from multiple research articles and responded effectively to diverse queries. It demonstrated the ability to deliver concise, relevant, and accurate responses, validating its performance and usefulness for multi-document analysis tasks.
