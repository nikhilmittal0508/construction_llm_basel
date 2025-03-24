from flask import Flask, jsonify, request
from flask_restful import Api
from zipfile import ZipFile
# import transformers
# from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
# from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
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
# from sentence_transformers import CrossEncoder
# from transformers import AutoModelForSequenceClassification
from langchain.document_loaders import UnstructuredFileLoader, TextLoader, JSONLoader, PyPDFLoader
import json
import openai
#from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI, OpenAI
import shutil
from concurrent.futures import ProcessPoolExecutor
import uuid
from sqs_queue_url import *
from utils import *

import warnings
warnings.filterwarnings('ignore')

config_file_path = 'params.json'

with open(config_file_path, 'r') as config_file:
    config_data = json.load(config_file)
    config = config_data.get('config', {})
    
    
# Set the cache directory
cache_dir = "./hal_model"

# Create the cache directory if it doesn't exist
os.makedirs(cache_dir, exist_ok=True)

# hal_model = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model',
#                                                                trust_remote_code=True,
#                                                                cache_dir=cache_dir,
#                                                                # token="",
#                                                                revision="main")


# Initiate the memory
memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )

def get_memory():
    return memory

def reset_memory_rag():
    global memory
    memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
            )
    return memory

qdrant_client = QdrantClient(
    url=config['url'], 
    api_key=config['qdrant_api_key'],
)

app = Flask(__name__)
api = Api(app)



@app.route('/api/create_collection', methods=['POST'])
def create_new_collection():
    inputs = request.json
    collection_name = inputs['collection_name']
    model_type = inputs['model_type']
    
    collection_name = collection_name + "_" + model_type
    if model_type == "openai":
        vector_size =  1536
    else:
        vector_size =  384
    
    response = create_one_collection(collection_name, vector_size, qdrant_client)
    if response:
        source_folder = config['source_docs']
        folder_path = os.path.join(source_folder, collection_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        return jsonify(response)    
    
    return jsonify(response)


@app.route('/api/get_all_collections', methods=['POST'])
def get_all_collection():
    inputs = request.json
    model_type = inputs['model_type']
    
    collection_list = []
    all_collections =  qdrant_client.get_collections().collections
    for i in all_collections:
        if model_type in i.name:
            collection_list.append(i.name)
    return jsonify(collection_list)


@app.route('/api/get_all_folders', methods=['POST'])
def getting_all_the_folder_names_from_collection_folder():
    inputs = request.json
    collection_name = inputs['collection_name']
    
    source_folder = config['source_docs']
    collection_folder_path = os.path.join(source_folder, collection_name)
    folders = [folder for folder in os.listdir(collection_folder_path) if os.path.isdir(os.path.join(collection_folder_path, folder))]
    return jsonify(folders)


@app.route('/api/get_all_filenames', methods=['POST'])
def getting_all_the_file_names_from_collection_folder():
    inputs = request.json
    collection_name = inputs['collection_name']
    
    source_folder = config['source_docs']
    collection_folder_path = os.path.join(source_folder, collection_name)
    files = [item for item in os.listdir(collection_folder_path) if os.path.isfile(os.path.join(collection_folder_path, item))]
    return jsonify(files)


@app.route('/api/delete_collection', methods=['POST'])
def delete_collection():
    inputs = request.json
    collection_name = inputs['collection_name']
   
    #deleting the collection in qdrant
    deletion_status = qdrant_client.delete_collection(collection_name=collection_name)
    
    #deleting the collection folder from source folder
    # Check if the file exists before attempting to delete
    source_folder = config['source_docs']
    collection_folder_path = os.path.join(source_folder, collection_name)
    
    if os.path.exists(collection_folder_path):
        # If it exists, delete the folder
        shutil.rmtree(collection_folder_path)
        print(f"Folder '{collection_folder_path}' has been deleted.")
    else:
        print(f"Folder '{collection_folder_path}' does not exist.")
        
    return jsonify(deletion_status)



@app.route('/api/delete_file', methods=['POST'])
def delete_single_file():
    inputs = request.json
    collection_name = inputs['collection_name']
    filename = inputs['filename']
    
    resp_qdrant = delete_file_qdrant(filename, collection_name, qdrant_client)
    if resp_qdrant:
        resp_source_file = delete_file_from_source_collection_folder(filename, collection_name, config['source_docs'])
        return jsonify(resp_source_file)
    
    return jsonify(resp_qdrant) # boolean return


@app.route('/api/delete_folder', methods=['POST'])
def delete_folder():
    inputs = request.json
    collection_name = inputs['collection_name']
    folder_name = inputs['folder_name']
    
    resp_qdrant = delete_folder_qdrant(folder_name, collection_name, qdrant_client)
    if resp_qdrant:
        resp_source_folder = delete_folder_from_source_collection_folder(folder_name, collection_name, config['source_docs'])
        return jsonify(resp_source_folder)
    
    return jsonify(resp_qdrant) # boolean return


@app.route('/api/file_upload_qdrant', methods=['POST'])
def file_ingestion_qdrant():
    
    collection_name = request.form["collection_name"]
    model_type = request.form["model_type"]
    
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return "No file part"

    uploaded_file = request.files['file']
    #print("type: ",type(uploaded_file))

    # Check if a file is selected
    if uploaded_file.filename == '':
        return "No selected file"
    
    # Check if the file type is allowed
    if uploaded_file.filename.lower().endswith((".txt", ".pdf", ".docx", ".xlsx", ".pptx", ".csv", ".html", ".json")):
        
        # already_ingested_files = check_qdrant_ingested_files(collection_name, qdrant_client)
        
        # Check if the file is already ingested
        # if uploaded_file.filename in already_ingested_files:
            # return f"File '{uploaded_file.filename}' is already ingested."
        
        source_folder = config['source_docs']
        folder_path = os.path.join(source_folder, collection_name)
        file_path = os.path.join(folder_path,  uploaded_file.filename)

        if check_file_present_qdrant_collection(collection_name, qdrant_client, file_path):
            return f"File '{uploaded_file.filename}' is already ingested."
        
        # Save the file in the source folder
        uploaded_file.save(file_path)

        sqs_messages = {
                    "operation_type": "file_upload_qdrant_sqs",
                    "file_path": file_path,
                    "filename": uploaded_file.filename,
                    "collection_name": collection_name,
                    "model_type": model_type,
                    "folder_path": folder_path
                }

        sqs_messages = json.dumps(sqs_messages)

        #sqs_queue.put(sqs_messages)

        unique_id = uuid.uuid4()

        sqs.send_message(QueueUrl=queue_url,MessageBody=sqs_messages,MessageGroupId="Group-"+str(unique_id),MessageDeduplicationId="Dup-"+str(unique_id))

        
        return "request for file ingestion is added into the sqs queue, pls come back after 30min"
    else:
        return "Error: Unsupported file type"
    
    
    
@app.route('/api/folder_upload_qdrant', methods=['POST'])
def ingest_folder():
    try:
        collection_name = request.form["collection_name"]
        model_type = request.form["model_type"]

        # Check if the POST request has the file part
        if 'file' not in request.files:
            return "No file part"

        uploaded_file = request.files['file']
       
        # Check if a file is selected
        if uploaded_file.filename == '':
            return "No selected file"

        # Specify the destination folder path
        source_folder = config['source_docs']
        destination_folder = os.path.join(source_folder, collection_name)
        print("source_folder: ",source_folder) 
        print("destination_folder: ",destination_folder)
        

        print(uploaded_file.filename)
        zip_file_path = os.path.join(destination_folder , uploaded_file.filename)
        print("zip_file_path: ",zip_file_path)

        uploaded_file.save(zip_file_path)

        print("Print after upload zip file")

        sqs_messages = {
                    "operation_type": "folder_upload_qdrant_sqs",
                    "zip_file_path": zip_file_path,
                    "destination_folder": destination_folder,
                    "collection_name": collection_name,
                    "model_type": model_type,
                }

        sqs_messages = json.dumps(sqs_messages)

        #sqs_queue.put(sqs_messages)

        unique_id = uuid.uuid4()

        sqs.send_message(QueueUrl=queue_url,MessageBody=sqs_messages,MessageGroupId="Group-"+str(unique_id),MessageDeduplicationId="Dup-"+str(unique_id))
        #sqs.send_message(QueueUrl=queue_url,MessageBody=sqs_messages,MessageGroupId="12345",MessageDeduplicationId="56789")

        return "request for folder ingestion is added into the sqs queue, pls come back after 30min"
        
    except Exception as e:  # Catch any general exception
        return jsonify({"output": "Ingestion Incomplete", "error": str(e)})
    
    
def ingest_file_process(folder_path,filename,collection_name,model_type):
    print(filename)
    file_path = os.path.join(folder_path, filename)
        
    if file_path.endswith((".json")):
        loader = JSONLoader(file_path, jq_schema='.content')
        documents = loader.load()

    elif file_path.endswith((".csv")):
        loader = CSVLoader(file_path)
        documents = loader.load()
        
    elif file_path.endswith((".pdf")):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
                    
    elif file_path.endswith((".xlsx")):
        loader = UnstructuredExcelLoader(file_path)
        documents = loader.load()

    else:
        loader = UnstructuredFileLoader(file_path)
        documents = loader.load()
    
    #Split documents into chunks
    texts =  texts_splitter(documents)
    
    # Ingestion
    response = ingesting_file_qdrant_db(collection_name, texts, model_type, config["url"], config["qdrant_api_key"], config['embedding_model'], config['cache_folder'], config['open_api_key'])

    return 

    
@app.route('/api/get_answer', methods=['POST'])
def get_answer():
    inputs = request.json
    query = inputs['query']
    no_of_source = inputs['no_of_source']
    qa_prompt = inputs['user_prompt']
    model_type = inputs["model_type"]
    collection_name = inputs["collection_name"]
    doc_search_criteria = inputs["search_type"]
    reset_memory = inputs["reset_memory"]
    
    print(inputs)
    
    if not qa_prompt:
        # Use a generic QA prompt if user prompt is empty
        qa_prompt = """Answer the question based on the context below. If the
                       question cannot be answered using the information provided answer
                       with 'I don't know' """
        
    general_system_template = r"""
        ----
        {context}
        ----
        """
    general_system_template = qa_prompt + "\n" + general_system_template
    general_user_template = "Question:```{question}```"
    messages = [
                SystemMessagePromptTemplate.from_template(general_system_template),
                HumanMessagePromptTemplate.from_template(general_user_template)
    ]
    prompt = ChatPromptTemplate.from_messages(messages)

    # print(prompt)
    
    if model_type == "openai":
        embedding_function = OpenAIEmbeddings(openai_api_key=config["open_api_key"])
    elif model_type == "mistral":
        embedding_function = HuggingFaceEmbeddings(model_name=config['embedding_model'], cache_folder=config['cache_folder'])
    
    # qdrant_client = QdrantClient(url=url, api_key=api_key)
    qdrant_db = Qdrant(qdrant_client, embeddings=embedding_function, collection_name=collection_name)
    
    if doc_search_criteria=="similarity_score_threshold":
        retriever = qdrant_db.as_retriever(search_type=doc_search_criteria, search_kwargs={"k": no_of_source, "score_threshold":0.80})
    else:
        retriever = qdrant_db.as_retriever(search_type=doc_search_criteria, search_kwargs={"k": no_of_source})
    
    if model_type == "openai":
        # Create QA chain
        #llm = OpenAI(openai_api_key=config["open_api_key"], model="gpt-3.5-turbo-instruct", max_tokens=2000)
        # llm = ChatOpenAI(openai_api_key=config["open_api_key"], model="gpt-4-0125-preview", max_tokens=2000)
        llm = ChatOpenAI(openai_api_key=config["open_api_key"], model="gpt-4o", max_tokens=2000)

    elif model_type == "mistral":
        
        llm = LlamaCpp(
            streaming = True,
            model_path=config['mistral_model_path'],
            temperature=0.75,
            top_p=1,
            verbose=True,
            n_ctx=4096,
            n_gpu_layers=-1,
            n_batch=1000,
            max_tokens = 2000,
            callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
        )
        
    # qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='refine', retriever=retriever, return_source_documents=True)
    get_new_memory = None
    if reset_memory:
        get_new_memory = reset_memory_rag()
    get_new_memory = get_memory()

    qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=get_new_memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt},
        )

    # Run QA
    # result = qa_chain({"query": query})
    result = qa_chain({"question": query})


    # hal_score = hal_model.predict([result['question'],result['answer']])


    new_result = {}
    new_result["question"] = result["question"]
    new_result["answer"] = result["answer"]

    result_list = []
    for document in result["source_documents"]:
        result_dict = {
            'page_content': document.page_content,
            'metadata': document.metadata
        }
        result_list.append(result_dict)

    # print(result_list)
    new_result["source_documents_list"] = result_list
    
    # if hal_score > 0.3:
    #     new_result["is_hallucination"] = False
    # else:
    #     new_result["is_hallucination"] = True
        

    
    # result = qa.run(query)
    return jsonify(new_result)


@app.route('/api/get_relevant_docs', methods=['POST'])
def get_relevant_documents():
    inputs = request.json
    query = inputs['query']
    no_of_source = inputs['no_of_source']
    model_type = inputs["model_type"]
    collection_name = inputs["collection_name"]
    doc_search_criteria = inputs["search_type"]
    
    # print(inputs)
    
    if model_type == "openai":
        embedding_function = OpenAIEmbeddings(openai_api_key=config["open_api_key"])
    elif model_type == "mistral":
        embedding_function = HuggingFaceEmbeddings(model_name=config['embedding_model'], cache_folder=config['cache_folder'])
    
    # qdrant_client = QdrantClient(url=url, api_key=api_key)
    qdrant_db = Qdrant(qdrant_client, embeddings=embedding_function, collection_name=collection_name)
    
    if doc_search_criteria=="similarity_score_threshold":
        retriever = qdrant_db.as_retriever(search_type=doc_search_criteria, search_kwargs={"k": no_of_source, "score_threshold":0.80})
    else:
        retriever = qdrant_db.as_retriever(search_type=doc_search_criteria, search_kwargs={"k": no_of_source})

    docs = retriever.get_relevant_documents(query)
    # Sources of the documents
    source = {}
    for i in range(len(docs)):
        source[str(i+1)+"_Page Content"] = docs[i].page_content
        source[str(i+1)+"_Doc Name"] = docs[i].metadata
        
    return jsonify(source)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)
