import os
import tqdm
from dotenv import load_dotenv,find_dotenv
from pathlib import Path

from flask import Flask, jsonify, request
from flask_restful import Api

# import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, JSONLoader
from langchain.llms import LlamaCpp, GPT4All
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate 
from langchain.prompts import SystemMessagePromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from qdrant_client.http import models
from qdrant_client import QdrantClient
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.document_loaders.csv_loader import CSVLoader
#from sentence_transformers import CrossEncoder

import openai
from langchain.llms import OpenAI
import shutil


def create_one_collection(name_of_the_collection, dimension, qdrant_client):
    # print(name_of_the_collection, dimension, qdrant_client)
    try:
        qdrant_client.create_collection(
            collection_name=name_of_the_collection,
            vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
        )
        return True
    except Exception as e:
        # print(e)
        return False
    

def delete_file_qdrant(filename_to_del, collection, qdrant_client):
    try:
        source_folder = "source_documents"
        collection_folder_path = os.path.join(source_folder, collection)
        # print(collection_folder_path)
        file_name = os.path.join(collection_folder_path, filename_to_del)
        qdrant_client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchAny(any=[file_name]),
                        ),
                    ],
                )
            ),
        )
        return True

    except Exception as e:
        print(e)
        return False
    
    
# def delete_file_qdrant(filename_to_del, collection, qdrant_client):
#     a = qdrant_client.scroll(
#     collection_name=collection,
#     scroll_filter=models.Filter(),
#     limit=100000000000
#     )
    
#     unique_filenames = {}
#     for entry in a[0]:
#         filename = os.path.basename(entry.payload["metadata"]['source'])
#         entry_id = entry.id

#         if filename in unique_filenames:
#             unique_filenames[filename].append(entry_id)
#         else:
#             unique_filenames[filename] = [entry_id]
            
#     # print(unique_filenames)
    
#     try:
#         resp =  qdrant_client.delete(
#             collection_name=collection,
#             points_selector=models.PointIdsList(
#                 points=unique_filenames[filename_to_del],
#             ),
#         )
#         return True
    
#     except Exception as e:
#         print(e)
#         return False



def delete_file_from_source_collection_folder(filename_to_del, collection, source_docs):
    # Specify the file path (relative or absolute)
    source_folder = source_docs
    folder_path = os.path.join(source_folder,collection)
    file_path = os.path.join(folder_path, filename_to_del)
    
    # Check if the file exists before attempting to delete
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File '{file_path}' has been deleted.")
        return True
    else:
        print(f"File '{file_path}' not found.")
        return False


def getting_all_the_file_names_from_one_collection_folder(collection_name, folder_name_to_del):
    source_folder = "source_documents"
    collection_folder_path = os.path.join(source_folder, collection_name)
    # print(collection_folder_path)
    folder_path = os.path.join(collection_folder_path, folder_name_to_del)
    # print(folder_path)

    files = [os.path.join(folder_path,item) for item in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, item))]
    return files


def delete_folder_qdrant(folder_name_to_del, collection, qdrant_client):
    try:
        file_list = getting_all_the_file_names_from_one_collection_folder(collection, folder_name_to_del)
        qdrant_client.delete(
            collection_name=collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchAny(any=file_list),
                        ),
                    ],
                )
            ),
        )
        return True

    except Exception as e:
        print(e)
        return False

# def delete_folder_qdrant(folder_name_to_del, collection, qdrant_client):
#     a = qdrant_client.scroll(
#     collection_name=collection,
#     scroll_filter=models.Filter(),
#     limit=100000000000
#     )
    
#     unique_filenames = {}
#     unique_filenames[folder_name_to_del] = []
#     for entry in a[0]:
#         filename = entry.payload["metadata"]['source']
#         entry_id = entry.id
        
#         if folder_name_to_del in filename:
#             unique_filenames[folder_name_to_del].append(entry_id)
            
#     # print( unique_filenames )
#     try:
#         resp =  qdrant_client.delete(
#             collection_name=collection,
#             points_selector=models.PointIdsList(
#                 points=unique_filenames[folder_name_to_del],
#             ),
#         )
#         return True
    
#     except Exception as e:
#         print(e)
#         return False
    
    
def delete_folder_from_source_collection_folder(folder_name_to_del, collection, source_docs):
    # Specify the file path (relative or absolute)
    source_folder = source_docs
    collection_folder_path = os.path.join(source_folder, collection)
    folder_path = os.path.join(collection_folder_path, folder_name_to_del)
    
    # Check if the file exists before attempting to delete
    if os.path.exists(folder_path):
        # If it exists, delete the folder
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' has been deleted.")
        return True
    else:
        print(f"Folder '{folder_path}' does not exist.")
        return False

                
def check_qdrant_ingested_files(collection_name, qdrant_client): #to  check the files already ingested or not in the  given qdrant collection
    try:
        # qdrant_client = QdrantClient(url=url, api_key=api_key)
        a = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(),
            limit=100000000000
        )
        unique_filenames = {os.path.basename(entry.payload["metadata"]['source']) for entry in a[0]}
        already_ingested_files = sorted(list(unique_filenames))
        return already_ingested_files
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        raise RuntimeError(error_message)
        
        
        
def ingesting_file_qdrant_db(collection_name, texts, model_type, url, qdrant_api_key,  embedding_model, cache_folder, openai_api_key):
    if model_type == "openai":
        embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    elif model_type == "mistral":
        embedding_function = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=cache_folder)
    
    # Create a vectorstore from documents
    qdrant = Qdrant.from_documents(
                        texts,
                        embedding_function,
                        url=url,
                        prefer_grpc=True,
                        api_key=qdrant_api_key,
                        collection_name=collection_name,
                    )
    return True


def texts_splitter(documents):
    try:
        # Split documents into chunks
        # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=8000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        return texts
    except Exception as e:
        error_message = f"An error occurred during text splitting: {str(e)}"
        raise RuntimeError(error_message)


def check_file_present_qdrant_collection(collection_name, qdrant_client, file_path):
    try:
        # qdrant_client = QdrantClient(url=url, api_key=api_key)
        a = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(must=[
            models.FieldCondition(
                key="metadata.source",
                    match=models.MatchValue(value=file_path),
            ),
        ])
        )
        
        if len(a[0]) > 0:
            return True
        else:
            return False
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        raise RuntimeError(error_message)
        
        
def check_qdrant_ingested_files(collection_name, qdrant_client): #to  check the files already ingested or not in the  given qdrant collection
    try:
        # qdrant_client = QdrantClient(url=url, api_key=api_key)
        a = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=models.Filter(),
            limit=100000000000
        )
        unique_filenames = {os.path.basename(entry.payload["metadata"]['source']) for entry in a[0]}
        already_ingested_files = sorted(list(unique_filenames))
        return already_ingested_files
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        raise RuntimeError(error_message)
        
        
        
def ingesting_file_qdrant_db(collection_name, texts, model_type, url, qdrant_api_key,  embedding_model, cache_folder, openai_api_key):
    if model_type == "openai":
        embedding_function = OpenAIEmbeddings(openai_api_key=openai_api_key)
    elif model_type == "mistral":
        embedding_function = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=cache_folder)
    
    # Create a vectorstore from documents
    qdrant = Qdrant.from_documents(
                        texts,
                        embedding_function,
                        url=url,
                        prefer_grpc=True,
                        api_key=qdrant_api_key,
                        collection_name=collection_name,
                    )
    return True
