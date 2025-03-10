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
from sentence_transformers import CrossEncoder
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, JSONLoader, PyPDFLoader
import json
import openai
from langchain_openai import ChatOpenAI, OpenAI
import shutil
from langchain_community.document_loaders import UnstructuredFileLoader, TextLoader, JSONLoader, BSHTMLLoader, \
    UnstructuredExcelLoader, WebBaseLoader, PyPDFLoader, Docx2txtLoader
from concurrent.futures import ProcessPoolExecutor
from zipfile import ZipFile
from sqs_queue_url import *
from utils import *

import warnings
warnings.filterwarnings('ignore')

config_file_path = 'params.json'

with open(config_file_path, 'r') as config_file:
    config_data = json.load(config_file)
    config = config_data.get('config', {})

qdrant_client = QdrantClient(
    url=config['url'],
    api_key=config['qdrant_api_key'],
)


def file_upload_qdrant_sqs(sqs_message):
    filename = sqs_message['filename']
    file_path = sqs_message['file_path']
    collection_name = sqs_message['collection_name']
    model_type = sqs_message['model_type']
    folder_path = sqs_message['folder_path']

    # try:

    if filename.lower().endswith((".json")):
        loader = JSONLoader(file_path, jq_schema='.content')
        documents = loader.load()

    elif filename.lower().endswith((".csv")):
        loader = CSVLoader(file_path)
        documents = loader.load()

    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
        documents = loader.load()

    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents = loader.load()

    else:
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()

    # Split documents into chunks
    texts = texts_splitter(documents)
    
    # Ingestion
    response = ingesting_file_qdrant_db(collection_name, texts, model_type, config["url"], config["qdrant_api_key"],
					config['embedding_model'], config['cache_folder'], config['open_api_key'])
    # print("response:", response)
    
    if response:
        return "Ingesting Done"
    else:
        return "Ingestion Failed"

    # except Exception as e:
    # # If the file was saved, attempt to remove it
    #     if 'file_path' in locals():
    #         try:
    #             os.remove(file_path)
    #         except:
    #             pass  # If file removal fails, we don't want to mask the original error

    #     # Log the error (you might want to use a proper logging system)
    #     print(f"An error occurred: {str(e)}")
    #     # Return a user-friendly error message
    #     return f"An error occurred during file processing: {str(e)}"

def folder_upload_qdrant_sqs(sqs_message):
    zip_file_path = sqs_message['zip_file_path']
    destination_folder = sqs_message['destination_folder']
    collection_name = sqs_message['collection_name']
    model_type = sqs_message['model_type']

    try:
        with ZipFile(zip_file_path, 'r') as zObject:
        	zObject.extractall(path=destination_folder)

        if os.path.exists(zip_file_path):
        	os.remove(zip_file_path)
        else:
        	print("The file does not exist")

        file_list = []
        full_folder_path = zip_file_path.split('.')[0]
        for root, dirs, files in os.walk(full_folder_path):
        	for file_name in files:
        		file_path = os.path.join(root, file_name)
        		if file_path.endswith((".txt", ".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".html", ".json", ".wav", ".mp3", ".eml")):
        			file_list.append(file_path)

        print("to_be_ingested:", file_list)

        if len(file_list) > 0:
        	errors = []
        	try:
        		with ProcessPoolExecutor(max_workers=2) as executor:
        			futures = []
        			for i, file_path in tqdm.tqdm(enumerate(file_list)):
        				try:
        					print("\n new file getting ingested....")
        					future = executor.submit(ingest_file_process, destination_folder, os.path.normpath(file_path),collection_name, model_type)
        					futures.append(future)
        				except Exception as e:
        					error_message = f"Error submitting task for file {file_path}: {str(e)}"
        					print(error_message)  # Print the error
        					errors.append(error_message)  # Add the error to the list

        			# Wait for all tasks to complete
        			results = []
        			for future in futures:
        				try:
        					result = future.result()
        					results.append(result)
        				except Exception as e:
        					print(f"Error in task execution: {str(e)}")

        	except Exception as e:
        		print(f"An error occurred in the process pool execution: {str(e)}")
        	finally:
        		# Ensure the executor is shut down properly
        		executor.shutdown(wait=True)

        # After all operations, save the errors to a file
        with open('error_log.json', 'w') as f:
        	json.dump(errors, f, indent=4)

        # Open the file in write mode
        return "Ingestion Complete"

    except Exception as e:
    	return f"Ingestion Incomplete with error: {str(e)}"


def ingest_file_process(folder_path, file_path, collection_name, model_type):
    print("ingest_file_process: file_path", file_path)
    # file_path = os.path.join(folder_path, filename)

    if file_path.lower().endswith((".json")):
        loader = JSONLoader(file_path, jq_schema='.content')
        documents = loader.load()

    elif file_path.lower().endswith((".csv")):
        loader = CSVLoader(file_path)
        documents = loader.load()

    elif file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()

    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
        documents = loader.load()

    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
        documents = loader.load()

    else:
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()

    # Split documents into chunks
    texts = texts_splitter(documents)
    print("ingest_file_process, texts lenght: ", len(texts))

    # Ingestion
    response = ingesting_file_qdrant_db(collection_name, texts, model_type, config["url"],
                                        config["qdrant_api_key"], config['embedding_model'], config['cache_folder'],
                                        config['open_api_key'])
    print("ingest_file_process:", response)

    return

if __name__ == "__main__":

    while (True):
        
        response = sqs.receive_message(QueueUrl=queue_url,
                                       MaxNumberOfMessages=1,  # Adjust as needed
                                       MessageAttributeNames=['All'])

        # Print the received message(s)
        for message in response.get('Messages', []):
            print(f'Received message: {message["Body"]}')
            sqs_message = message["Body"]
            sqs_message = json.loads(sqs_message)

            if sqs_message["operation_type"] == "file_upload_qdrant_sqs":
                resp = file_upload_qdrant_sqs(sqs_message)
            elif sqs_message["operation_type"] == "folder_upload_qdrant_sqs":
                resp = folder_upload_qdrant_sqs(sqs_message)
            
            print(resp)

            sqs.delete_message(QueueUrl=queue_url, ReceiptHandle=message['ReceiptHandle'])
