from typing import List, Iterable
import time
import os
from tqdm import tqdm
import json

import torch
from transformers import pipeline

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader


class GTEEmbedding:
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        print("\nStart inserting entities")
        
        start_time = time.time()
        text_embeddings = self.model.encode(texts)
        text_embeddings = [emb.tolist() for emb in text_embeddings]
        end_time = time.time()
        
        print(f"\nVectorstore insert time for {len(text_embeddings)} vectors: ", end="")
        print(f"{round(end_time - start_time, 2)} seconds")

        return text_embeddings
            
    def embed_query(self, query: str) -> List[float]:

        q_embedding = self.model.encode(query, prompt_name="query").tolist()
        
        return q_embedding
    

def load_pdfs(
        directory_path=None,
        cache_dir=None,
        model="meta-llama/Llama-3.2-3B-Instruct"
        ):

    os.environ["HF_HUB_CACHE"] = cache_dir

    pipe = pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    def list_pdf_files(directory):
        return [f for f in os.listdir(directory) if f.lower().endswith('.pdf')]

    pdf_files = list_pdf_files(directory_path)
    print(pdf_files)

    all_document_pages = []
    all_document_papers = []
    for file_path in tqdm(pdf_files):
        document_pages = []
        loader = PyPDFLoader(os.path.join(directory_path, file_path))

        document_pages.extend(loader.load())

        #print(f"\nNumber of Pages: {len(document_pages)}")

        metadata = get_metadata_with_llm(document_pages[0], pipe)

        for doc in document_pages:
            for key in metadata.keys():
                doc.metadata[key] = metadata[key]

        all_document_pages.extend(document_pages)
        all_document_papers.append(document_pages)

    return all_document_pages, all_document_papers


def get_metadata_with_llm(doc, pipe):

    import re

    def extract_quoted_substring(text):
        match = re.search(r'["\'](.*?)["\']', text)
        return match.group(1) if match else None

    metadata_tags = ["title", "date", "authors", "important keywords"]
    print("\n")

    metadata = {}

    for tag in metadata_tags:
        prompt = f"""
        This is the first page of a scientific publication:
        {doc.page_content}
        Extract the {tag} from the publication.
        Return the answer in a simple string with quotation marks (e.g. \"answer\").
        If you are unsure chose the most likely possibility.
        If you cant find the required information, return "unknown".
        """

        messages = [
            {"role": "system", "content": "Your task is to extract information from a scientific publication"},
            {"role": "user", "content": prompt},
        ]
        outputs = pipe(
            messages,
            max_new_tokens=256,
        )
        #print(outputs[0]["generated_text"][-1])
        print(f"Extracted info: {outputs[0]["generated_text"][-1]["content"]}")
        tag_value = extract_quoted_substring(outputs[0]["generated_text"][-1]["content"])

        metadata[tag] = tag_value

    return metadata


def display_graph(graph):
    img_data = graph.get_graph().draw_mermaid_png()
    with open("workflow_graph.png", "wb") as f:
        f.write(img_data)


def print_event(event: dict, _printed: set, max_length=2000):
    current_state = event.get("dialog_state")
    if current_state:
        print("Currently in: ", current_state[-1])
    message = event.get("messages")
    if message:
        if isinstance(message, list):
            message = message[-1]
        if message.id not in _printed:
            msg_repr = message.pretty_repr(html=True)
            if len(msg_repr) > max_length:
                msg_repr = msg_repr[:max_length] + " ... (truncated)"
            print(msg_repr)
            _printed.add(message.id)


def save_docs_to_jsonl(array:Iterable[Document], file_path:str)->None:
    with open(file_path, 'w') as jsonl_file:
        for doc in array:
            jsonl_file.write(doc.json() + '\n')


def load_docs_from_jsonl(file_path)->Iterable[Document]:
    array = []
    with open(file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array

