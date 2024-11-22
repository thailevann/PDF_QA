from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import os
import torch

from llama_index.core import Document as Document_llama
from docx import Document as Documentdocx #pip install python-docx

import fitz
import easyocr

from chainlit.types import AskFileResponse
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI

from llama_index.core.node_parser import (
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate,HumanMessagePromptTemplate

import chainlit as cl

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import  TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from llama_index.readers.file import PyMuPDFReader





def splitter_document(doc):
    device_type = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device=device_type) # must be the same as the previous stage
    splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )
    nodes = splitter.get_nodes_from_documents(doc)
    return nodes

def is_scanned_pdf(doc):
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Kiểm tra xem trang có văn bản hay không
        text = page.get_text()
        if text.strip():  # Nếu có văn bản, không phải PDF quét
            return False
        
    return True 

def read_scanned(doc):
    reader = easyocr.Reader(['en', 'vi'])
    text_list = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()  
        img_bytes = pix.tobytes("png") 
        
        # Hiển thị hình ảnh
        # Nhận diện văn bản từ hình ảnh
        result = reader.readtext(img_bytes)
        
        # Kiểm tra xem có nhận diện được văn bản không
        text = ""
        for detection in result:
            detected_text = detection[1]
            
            # Chuyển đổi văn bản thành UTF-8 nếu cần thiết
            #detected_text = detected_text.encode('utf-8').decode('utf-8') 
            
            text += detected_text + " "  # Dán các đoạn văn bản vào
        
        # Tạo Document từ văn bản nhận diện được và thêm vào text_list
        document = Document_llama(text=text, metadata={"source": f"page_{page_num+1}"})
        text_list.append(document)  # Thêm đối tượng Document vào danh sách

    return text_list

def extract_images_and_extract_text(pdf_path):
    doc = fitz.open(pdf_path) 
    reader = easyocr.Reader(['en', 'vi']) 
    text_list = []  
    
    # Duyệt qua tất cả các trang trong file PDF
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list_on_page = page.get_images(full=True)
        for img_index, img in enumerate(image_list_on_page):
            xref = img[0]  # Tham chiếu đến hình ảnh
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]  

            result = reader.readtext(image_bytes)
            
            # Kiểm tra xem có nhận diện được văn bản không
            if result:
                text = ""
                for detection in result:
                    detected_text = detection[1]
                    text += detected_text + " "
                #text_list.append(text)
                document = Document_llama(text=text, metadata={"source": f"image_{img_index+1}"})
                text_list.append(document)

    return text_list

def read_docx(file_path):
    # Đọc file docx
    doc = Documentdocx(file_path)
    
    # Khởi tạo một danh sách để lưu các đối tượng Document
    documents = []
    
    # Duyệt qua tất cả các đoạn văn trong file .docx và lấy nội dung
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    
    # Tạo đối tượng Document với nội dung đã trích xuất
    document =Document_llama(text=text, metadata={"source": file_path})
    documents.append(document)
    
    return documents

def process_file(file: AskFileResponse):
    file_path = file.path
    if file_path.endswith(".txt"):
        loader = TextLoader(file_path,  encoding='utf-8')
        documents = loader.load()     
        documents = [
            Document_llama(text=doc.page_content, metadata={"source": file_path, "id_": str(i)})
            for i, doc in enumerate(documents)
        ]
    elif file_path.endswith(".pdf"):
        doc = fitz.open(file_path)
        is_scanned =  is_scanned_pdf(doc)
        if is_scanned:
            documents = read_scanned(doc)
        else:
            reader = PyMuPDFReader()
            documents = reader.load(file_path=file_path)        
            documents_from_image = extract_images_and_extract_text(file_path)
            documents.extend(documents_from_image)
            
    elif file_path.endswith(".docx"):
        #doc = fitz.open(file_path)
        documents = read_docx(file_path)
        documents_from_image = extract_images_and_extract_text(file_path)
        documents.extend(documents_from_image)
        
    else:
        raise ValueError("Unsupported file type")
    

    docs = splitter_document(documents)
    return docs


def get_vector_db(file_path):
    docs = process_file(file_path)  # Process the file to get documents
    documents = [Document(page_content=doc.text, metadata=doc.metadata) for doc in docs]
    embedding = HuggingFaceEmbeddings()
    vector_db = Chroma.from_documents(documents=documents, embedding=embedding)
    return vector_db

GOOGLE_API_KEY = 'AIzaSyDOX0ygiFy7AoUURHcesc5fRBPJQ8trRZ0'

LLM = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", api_key=GOOGLE_API_KEY)



welcome_message = """Welcome to the PDF QA! To get started:
1. Upload a PDF, txt or docx file
2. Ask a question about the file
"""
@cl.on_chat_start
async def on_chat_start():
    files = None
    while files is None:
        files = await cl.AskFileMessage(
            content=welcome_message,
            accept=["text/plain", "application/pdf",  "application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            #accept={"text/plain": [".docx"]},
            max_size_mb=20,
            timeout=180,
        ).send()
    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...",
                     disable_feedback=True)
    await msg.send()

    vector_db = await cl.make_async(get_vector_db)(file)

    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )
    retriever = vector_db.as_retriever(search_type="mmr",
                                       search_kwargs={'k': 3})
        
   
    template = """
    Bạn sẽ trả lời câu hỏi của người dùng dựa trên nội dung và lịch sử trò chuyện sau: 
    Thông tin:
    {context}.

    Nếu không tìm thấy thông tin phù hợp hãy:
    - Thông báo cho người dùng biết bạn không tìm thấy thông tin
    - Đưa ra câu trả lời dựa trên kiến thức nền tảng của bạn.
    Hãy giữ câu trả lời ngắn gọn và trả lời bằng tiếng Việt.
    Câu hỏi: {question}

    Câu trả lời:
    """
    messages = [
        SystemMessagePromptTemplate.from_template(template),
        HumanMessagePromptTemplate.from_template("{question}")
        ]
    custom_rag_prompt = ChatPromptTemplate.from_messages(messages)



    chain = ConversationalRetrievalChain.from_llm(
        llm=LLM,
        
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": custom_rag_prompt },
        get_chat_history=lambda h : h
    )

    msg.content = f"`{file.name}` processed. You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]

    source_documents = res["source_documents"]
    text_elements = []
    '''
    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content,
                        name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]
        
        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    '''
    await cl.Message(content=answer, elements=text_elements).send()