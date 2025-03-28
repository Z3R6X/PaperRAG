from nodes import AgentState, RetrieverInput
from nodes import agent, simple_agent, filter_papers, hyde, do_retrieve, generate
from util import GTEEmbedding
from util import load_pdfs
from util import print_event, display_graph
from util import save_docs_to_jsonl, load_docs_from_jsonl

import argparse
import os
import time
import json
import uuid

from openai import OpenAI
from sentence_transformers import SentenceTransformer

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.tools.simple import Tool
from langchain.retrievers.document_compressors import FlashrankRerank

from langgraph.prebuilt import tools_condition
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_base", default="http://134.2.17.206:8000/v1")
    parser.add_argument("--localhost", action="store_true")
    parser.add_argument("--cache_dir", default="cache")
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--pdf_dir", default="data/pdfs")
    parser.add_argument("--force_rag", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose

    # Setup OpenAI LLM with VLLM
    openai_api_key = "EMPTY"
    if args.localhost:
        api_base = "http://0.0.0.0:8000/v1"
    else:
        api_base = args.api_base
    #openai_api_base = "http://0.0.0.0:8000/v1"
    #openai_api_base = "http://134.2.17.206:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=api_base,
    )
    model_ids = client.models.list()
    model_id = model_ids.data[0].id   
    print(f"Backbone LLM: {model_id}")
    llm = ChatOpenAI(
        model=model_id,
        openai_api_key=openai_api_key,
        openai_api_base=api_base,
        temperature=0,
    )

    # Load embedding model
    emb_model_gte = SentenceTransformer(
        #"Alibaba-NLP/gte-Qwen2-7B-Instruct",
        "Alibaba-NLP/gte-Qwen2-1.5B-instruct",
        trust_remote_code=True,
    )
    embeddings_vectorstore = GTEEmbedding(emb_model_gte)

    # Create data dir paths
    dense_path = os.path.join(args.data_dir, "chroma_db_basic_chunking")
    chunks_path = os.path.join(args.data_dir, "basic_chunks.jsonl")
    metadata_path = os.path.join(args.data_dir, 'paper_metadata.json')

    # Check if paths already exist
    if os.path.exists(dense_path) and os.path.exists(chunks_path) and os.path.exists(metadata_path):
        
        # Load dense vectorstore for dense retrieval
        dense_vectorstore = Chroma(
            embedding_function=embeddings_vectorstore,
            persist_directory=dense_path
        )

        # Load chunks for sparse retrieval
        chunks = load_docs_from_jsonl(chunks_path)

        # Load metadata
        metadata_list = json.load(open(metadata_path))
    
    else:
        # Create raw and full paper paths
        raw_paper_path = os.path.join(args.data_dir, "papers_raw.jsonl")
        full_paper_path = os.path.join(args.data_dir, "papers_full.jsonl")

        # Check if paths already exist (intermediate step)
        if os.path.exists(raw_paper_path) and os.path.exists(full_paper_path):
            
            # Load documents
            print("\nLoad documents from memory")
            documents = load_docs_from_jsonl(raw_paper_path)
            full_paper_docs = load_docs_from_jsonl(full_paper_path)
 
        else:
            # Create documents from PDFs and save in data dir
            documents, documents_paper = load_pdfs(args.pdf_dir, cache_dir=args.cache_dir)
            save_docs_to_jsonl(documents, raw_paper_path)

            # Create documents for each paper and save in data dir
            full_paper_docs = []
            for i, doc_paper in enumerate(documents_paper):

                print(f"\nTitle: {doc_paper[0].metadata["title"]}")
                metadata=doc_paper[0].metadata
                paper_text = " ".join([doc.page_content for doc in doc_paper])
                full_paper_docs.append(Document(page_content=paper_text, metadata=metadata))

            save_docs_to_jsonl(full_paper_docs, full_paper_path)

        # Create and save metadata in data dir
        metadata_list = [{"id": id, "title": doc.metadata["title"], "authors": doc.metadata["authors"]} for id, doc in enumerate(full_paper_docs)]
        with open(metadata_path, 'w') as f:
            json.dump(metadata_list, f, indent=1)

        if True:
            # Create fixed length chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)

            # Add metadata to each chunk
            ref_title = chunks[0].metadata["title"] 
            chunk_id = 0
            for chunk in chunks:
                if chunk.metadata["title"] != ref_title:
                    ref_title = chunk.metadata["title"]
                    chunk_id = 0
                chunk.metadata["chunk_id"] = chunk_id
                chunk_id += 1

            len_chunks = len(chunks)
            print(f"\nNumber of chunks to embed: {len_chunks}")

            ## Create dense vectorstore and save to data dir
            start_time = time.time()
            for chunk in chunks:
                chunk.metadata["retriever_type"] = "dense"
                for key in chunk.metadata.keys():
                    # Replace None values with placeholder string
                    if chunk.metadata[key] == None:
                        print(f"\nChunk with None value: {chunk.metadata}")
                        chunk.metadata[key] = "None value replaced"

            print("\nCreate dense vectorstore")
            dense_vectorstore = Chroma.from_documents(
                documents=chunks,
                collection_name="rag-chroma",
                embedding=embeddings_vectorstore,
                persist_directory=dense_path
            )
            end_time = time.time()
            print(f"\nDense vectorstore created in {end_time-start_time} seconds")

            ## Create chunks for sparse retriever and save to data dir
            for chunk in chunks:
                chunk.metadata["retriever_type"] = "sparse"

            save_docs_to_jsonl(chunks, chunks_path)

    # Create retriever tool
    retriever_papers_tool =  Tool(
        name="search_publication",
        description="Search and return information about a reasearch group focusing on cognitive systems.",
        func=do_retrieve,
        args_schema=RetrieverInput
    )
    tools = [retriever_papers_tool]

    llm_no_tools = llm

    # Bind tool to LLM
    llm_with_tools = llm.bind_tools(tools)

    # Create compressor for reranking
    compressor = FlashrankRerank(
        model="ms-marco-MultiBERT-L-12",
        #model="rank_zephyr_7b_v1_full",
        top_n=25, # Maximum number of chunks after reranking
        )

    # Define functions for graph nodes
    def agent_graph(state):
        return agent(state, llm=llm_with_tools)
    
    def generate_graph(state):
        return generate(state, llm=llm_no_tools)

    def filter_papers_graph(state):
        return filter_papers(state, llm=llm_no_tools, metadata_list=metadata_list)
    
    def hyde_graph(state):
        return hyde(
                state,
                llm=llm_no_tools,
                dense_vs=dense_vectorstore,
                sparse_chunks=chunks,
                compressor=compressor
            )
    
    # Create langgraph workflow
    workflow = StateGraph(AgentState)

    # Add nodes to workflow
    workflow.add_node("agent", agent_graph)  # agent
    retrieve = ToolNode(tools)
    workflow.add_node("retrieve", retrieve) 
    workflow.add_node("generate", generate_graph)
    workflow.add_node("hyde", hyde_graph)
    workflow.add_node("filter_papers", filter_papers_graph)

    # Add edges to workflow
    workflow.add_edge(START, "agent")
    
    if args.force_rag:
        workflow.add_edge("agent", "retrieve")
    else:
        def simple_agent_graph(state):
            return simple_agent(state, llm=llm_no_tools)
        
        workflow.add_node("simple_agent", simple_agent_graph)
    
        workflow.add_conditional_edges(
            "agent",
            #"hyde",
            tools_condition,
            {
                "tools": "retrieve",
                #"tools": "hyde",
                END: "simple_agent",
            },
        )

        workflow.add_edge("simple_agent", END)
        
    workflow.add_edge("retrieve", "filter_papers")
    workflow.add_edge("filter_papers", "hyde")
    workflow.add_edge("hyde", "generate")
    workflow.add_edge("generate", END)

    # Add memory saver
    #memory = MemorySaver()
    
    # Compile and disply workflow graph
    graph = workflow.compile()
    display_graph(graph)

    config = {"configurable": {"thread_id": "1"}}

    _printed = set()

    thread_id = str(uuid.uuid4())

    config = {
        "configurable": {
            # Checkpoints are accessed by thread_id
            "thread_id": thread_id,
        }
    }

    # Start chat loop
    while True:

        user_input = input("\nUser: ")

        start_time = time.time()
        
        events = graph.stream(
            {"messages": ("user", user_input)}, config, stream_mode="values"
        )

        for event in events:
            print_event(event, _printed) if verbose else None

        end_time = time.time()

        print(f"\nAssistant ({end_time-start_time}s): {event.get("messages")[-1].content}") if not verbose else None


if __name__ == "__main__":
    main()
