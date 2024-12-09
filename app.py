import os
import random
import string
import time
from concurrent.futures import ThreadPoolExecutor

import streamlit as st
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError
#from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain.embeddings import BedrockEmbeddings
#from langchain_community.embeddings import BedrockEmbeddings
from langchain_aws import BedrockEmbeddings
#from langchain.vectorstores import Chroma
from langchain_community.vectorstores import Chroma
#from langchain.chat_models import BedrockChat
from langchain_community.chat_models import BedrockChat
from langchain_aws import ChatBedrock
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import re

def download_file_from_s3(bucket_name, key, local_path):
    """Download a file from S3 to a local path."""
    try:
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, key, local_path)
        st.write(f"Downloaded {key} from S3 to {local_path}")
    except NoCredentialsError:
        st.write("Error: No AWS credentials found.")
    except PartialCredentialsError:
        st.write("Error: Incomplete AWS credentials found.")
    except ClientError as e:
        st.write(f"ClientError: {e.response['Error']['Message']}")
    except Exception as e:
        st.write(f"An unexpected error occurred: {e}")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using PyPDFLoader."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()
    
    # Create a single document with page information
    combined_text = ""
    page_map = {}
    for i, page in enumerate(pages):
        page_content = page.page_content
        page_start = len(combined_text)
        combined_text += page_content + "\n\n"
        page_map[page_start] = i + 1
    
    return combined_text, page_map


def load_data_from_s3(bucket_name, prefix):
    """Load and concatenate text from all PDF files in an S3 bucket."""
    try:
        s3 = boto3.client('s3')
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        pdf_files = [content['Key'] for content in response.get('Contents', []) if content['Key'].endswith('.pdf')]
        
        all_documents = []
        with ThreadPoolExecutor(12) as executor:
            local_paths = []
            for pdf_file in pdf_files:
                local_path = os.path.join("temp", os.path.basename(pdf_file))
                download_file_from_s3(bucket_name, pdf_file, local_path)
                local_paths.append((local_path, pdf_file))  # Store both path and original filename
            
            for local_path, original_filename in local_paths:
                text, page_map = extract_text_from_pdf(local_path)
                doc = Document(
                    page_content=text,
                    metadata={
                        "source": original_filename,
                        "file_path": local_path
                    }
                )
                all_documents.append(doc)
                os.remove(local_path)
        
        st.write("Processing of all PDFs done")
        return all_documents
    
    except Exception as e:
        st.write(f"An error occurred while loading data from S3: {e}")

def split_pdf_text(docs):
    """Split large document into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        separators=['\n\n', '\n'],
        chunk_overlap=150
    )
    
    all_splits = []
    for doc in docs:
        # Split each document
        splits = text_splitter.split_documents([doc])
        
        # Update metadata for each split to include chunk information
        for i, split in enumerate(splits):
            split.metadata.update({
                "chunk": i,
                "source": split.metadata.get("source", "Unknown source"),
                "page": split.metadata.get("page", f"Chunk {i+1}")
            })
        
        all_splits.extend(splits)
    
    return all_splits



def generate_embeddings_and_vector_db(splits):
    """Generate embeddings and create vector database."""
    try:
        if not splits:
            raise ValueError("No document splits to process")

        bedrock_client = boto3.client(service_name="bedrock-runtime")
        embedding = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1", client=bedrock_client)
        
        persist_directory = os.path.join('docs/chroma', ''.join(random.choices(string.ascii_letters + string.digits, k=20)))
        
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory
        )
        
        return vectordb
    
    except Exception as e:
        st.write(f"An error occurred while generating embeddings: {e}")

def generate_prompt_template():
    """Generate prompt template for QA chain."""
    template = """## Policy Assistant

**Instructions:** Please provide guidance on the user's question based on the context provided.

**Context:**

{context}

**User Query:**

{question}
"""
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["question"], template=template)
    return QA_CHAIN_PROMPT

def query_retrieval_pipeline(vectordb, QA_CHAIN_PROMPT, question):
    """Run query retrieval pipeline."""
    try:
        llm = ChatBedrock(model_id="amazon.titan-text-express-v1")
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"  # Add this line to specify which key to store
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vectordb.as_retriever(),
            combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT},
            memory=memory,
            return_source_documents=True,
            return_generated_question=True
        )
        
        #result = qa_chain({"question": question})
        result = qa_chain.invoke({"question": question})
        return result["answer"], result["source_documents"]
    
    except Exception as e:
        st.write(f"An error occurred during query retrieval: {e}")
        return "I encountered an error while processing your question.", []



def chat_actions():
    """Perform actions based on user input."""
    user_question = st.session_state.get("user_question", "")
    if not user_question.strip():
        st.session_state["chat_history"].append({
            "role": "user", 
            "content": user_question, 
            "is_user": True
        })
        result = "Please provide a valid question."
        st.session_state["chat_history"].append({
            "role": "assistant", 
            "content": result, 
            "is_user": False,
            "references": []
        })
    else:
        st.session_state["chat_history"].append({
            "role": "user", 
            "content": user_question, 
            "is_user": True
        })
        result, source_docs = query_retrieval_pipeline(
            st.session_state["vectordb"], 
            st.session_state["QA_CHAIN_PROMPT"], 
            user_question
        )
        
        # Extract references from source documents
        references = []
        for doc in source_docs:
            # You might want to customize this based on your document structure
            page_number = doc.metadata.get('page', 'Unknown page')
            source_file = doc.metadata.get('source', 'Unknown source')
            references.append(f"Page {page_number} from {source_file}")
        
        st.session_state["chat_history"].append({
            "role": "assistant", 
            "content": result, 
            "is_user": False,
            "references": references
        })




def refresh_chat():
    """Refresh chat history."""
    st.session_state["chat_history"] = []
    st.rerun()  # Use st.rerun() instead


def main():
    """Main function."""
    st.set_page_config(page_title="Policy Chatbot", page_icon=":robot_face:")

    if "vectordb" not in st.session_state:
        bucket_name = "zen-policy"
        prefix = "documents/"
        
        docs = load_data_from_s3(bucket_name, prefix)
        if not docs:
            st.error("Failed to load documents from S3.")
            return
        
        splits = split_pdf_text(docs)
        vectordb = generate_embeddings_and_vector_db(splits)
        QA_CHAIN_PROMPT = generate_prompt_template()

        st.session_state["vectordb"] = vectordb
        st.session_state["QA_CHAIN_PROMPT"] = QA_CHAIN_PROMPT

    st.title("HR Policy Chatbot :robot_face:")

    input_placeholder = st.empty()

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    if st.button("Refresh Chat"):
        refresh_chat()

    with input_placeholder.container():
        user_question = st.chat_input("Your Question:", on_submit=chat_actions, key="user_question")

    if user_question:
        with st.spinner("Processing your question..."):
            time.sleep(2)  # Simulate processing time
            for message in reversed(st.session_state["chat_history"]):
                with st.chat_message(name=message["role"]):
                    st.write(message["content"])
                    # Display references if they exist and it's an assistant message
                    if message["role"] == "assistant" and "references" in message and message["references"]:
                        st.markdown("---")
                        st.markdown("**References:**")
                        for ref in message["references"]:
                            st.markdown(f"- {ref}")


    else:
        st.warning("Please enter a question.")

    footer = st.container()
    with footer:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("<h3>Powered by <span style='color:aqua;font-size:bold;'>Zensar<span><h3>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
