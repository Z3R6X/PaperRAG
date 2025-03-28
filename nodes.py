from typing import Annotated, Sequence, List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import time
import json
import re

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.retrievers import BM25Retriever


### State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_question: str
    retrieved_paper: Document
    paper_filter: List[str]


### Input
class RetrieverInput(BaseModel):
        """Input to the retriever."""
        query: str = Field(description="query to look up in retriever")


### Nodes
# Agent with tool calling 
def agent(state, llm):
    print("---CALL AGENT---")

    messages = state["messages"]
    user_question = messages[-1].content
    response = llm.invoke(messages)

    return {"messages": [response], "user_question": user_question}


# Simple agent without tool calling to handle questions unrelated to data
def simple_agent(state, llm):
    print("---CALL SIMPLE AGENT---")
    
    messages = state["messages"]
    response = llm.invoke(messages)
    
    return {"messages": [response]}


# Create a retriever filtered by paper titles determined by LLM
def create_filter_retriever(dense_vs, sparse_chunks, filter, verbose=None):

    start_time = time.time()
    # Crease filtered dense retriever
    dense_retriever_papers = dense_vs.as_retriever(
        search_kwargs={
            "k": 40, # Number of chunks retrieved by dense retriever
            "filter": {"title": {"$in": filter}}
        }
    )

    # Filter chunks and create filtered sparse retriever
    filtered_chunks = [c for c in sparse_chunks if c.metadata["title"] in filter]
    sparse_retriever_papers = BM25Retriever.from_documents(
        documents=filtered_chunks,
    )
    sparse_retriever_papers.k = 40 # Number of chunks retrieved by sparse retriever

    ## Create ensemble retriever with dense and sparse weighting
    retriever_papers = EnsembleRetriever(
        retrievers = [sparse_retriever_papers, dense_retriever_papers],
        #weighs = [0.25, 0.75],
        weighs = [0.4, 0.6],
        id_key = "chunk_id"
    )
    end_time = time.time()
    print(f"\nTime to create filtered retriever: {end_time-start_time}s")

    return retriever_papers


# Filter papers with LLM
def filter_papers(state, llm, metadata_list, verbose=None):
    print("--- Filter Paper ---")

    user_question = state["user_question"]

    # List paper titles
    title_string = "\n".join([f"{metadata['id']}: {metadata['title']} - Authors: {metadata['authors']}" for metadata in metadata_list])

    # Insert titles in prompt with filter instruction
    title_message = f"""Return the indices of scientific publications that might be relevant to answer the user question.
Return the indices as a python-style list (e.g. [0, 1, 2, 3]).
Question: {user_question}
Papers:
{title_string}
Indices (as python-style list): """
    
    filter_messages = [
        SystemMessage(
            content="Your task is the determine the relevance of scientific publications given a user question."
        ),
        HumanMessage(
            content=title_message
        )
    ]

    print(f"\nInstruction: {title_message}") if verbose else None

    # Generate reponse with LLM
    response = llm.invoke(filter_messages)

    print(f"\nResponse: {response}") #if verbose else None

    # Extract indices from reponse
    try: 
        indices = json.loads(response.content)

    except: 
        match = re.search(r"\[.*?\]", response.content)
        indices = json.loads(match.group(0))

    if bool(indices) is False:
        print("\nNo filter set by model")
        filtered_titles = [d["id"] for d in metadata_list]
        return {"paper_filter": filtered_titles}
    
    indices = [int(idx) for idx in indices]
    
    print(f"\nIndices: {indices}") #if verbose else None

    # Create filter with indices
    filtered_titles = [d["title"] for d in metadata_list if d["id"] in indices]
    print(f"\nFiltered Titles:\n{'\n'.join([f'{idx}. {title}' for idx, title in zip(indices, filtered_titles)])}") #if verbose else None

    return {"paper_filter": filtered_titles}


# Retrieve relevant chunks using Hypothetical Document Embedding (HyDE)
def hyde(state, llm, compressor, dense_vs, sparse_chunks, verbose=None):
    print("--- HYDE ---")

    # Insert user question in HyDE template
    user_question = state["user_question"]

    hyde_messages = [
        SystemMessage(
            content="Your task is to write a short passage in the style of an academic paper published by a group focusing on machine learning, neural networks, robotics and computer vision."
        ),
        HumanMessage(
            content=f"""Please write a passage to answer the question.
                    Do not mention the title or the names of the authors in the passage.
                    Question: {user_question}
                    Passage: """
        )
    ]

    print(f"\nHyDE Message: {hyde_messages}") if verbose else None

    # Generate response
    response = llm.invoke(hyde_messages)
    print(f"\nHyDE Response: {response}") if verbose else None

    # Create filtered retriever
    retriever = create_filter_retriever(dense_vs, sparse_chunks, filter=state["paper_filter"])

    # Create reranker
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    ) 

    # Retrieve relevant chunks
    print(f"\nFilter: {state['paper_filter']}") if verbose else None
    documents = compression_retriever.get_relevant_documents(
        response.content,
    )

    # Format chunks with metadata
    template = "Title: {}\nAuthors: {}\nDate: {}\nExtracted Content:\n{}"

    retrievals = []
    for doc in documents:
        formated_retrieval = template.format(
            doc.metadata["title"],
            doc.metadata["authors"],
            doc.metadata["date"],
            doc.page_content
        )

        retrievals.append(formated_retrieval)

    combined_retrievals_hyde = "\n\n".join(retrievals)

    print(f"\n\nCombined Retrievals (hyde):\n{combined_retrievals_hyde}") if verbose else None

    # Create tool message with formated chunks
    tool_message = ToolMessage(
        content=combined_retrievals_hyde,
        tool_call_id = "test"
    )

    return {"messages": [tool_message]} 


# Generate reponse based on retrieved chunks
def generate(state, llm):
    print("---GENERATE---")

    messages = state["messages"]

    instruction = HumanMessage(content="Answer the user question given the recieved context.")
    #instruction = HumanMessage(content="If the recieved context is not related to the question answer the question without considering the retrieved context.")
    
    messages.append(instruction)
    response = llm.invoke(messages)

    return {"messages": [response]}


# Pass function
def pass_response(state):
    print("\n--- Pass ---")
    return state


# Retrieve function for retriever tool
def do_retrieve(query:str) -> str:
    return "RAG used to improve response"
