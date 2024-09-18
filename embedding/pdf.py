import os
import time
import fitz  # PyMuPDF
from tqdm import tqdm
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json

def main():
    start_time1 = time.time()
    
    # PDF 파일 경로 설정
    path = "C:/AAA_Analyze/pdf/clothes.pdf"

    # 텍스트를 로드 및 분할
    files_text = get_text_update(path)
    text_chunks = get_text_chunks(files_text)

    # 벡터스토어 생성 및 병합
    vectorstore = get_vectorstore(text_chunks)
    index_path = "C:/AAA_Analyze/vector/faiss_index"
    config_path = "C:/AAA_Analyze/vector/embeddings_config.json"

    # 인덱스가 이미 존재하면 로드 후 병합, 그렇지 않으면 새로 저장
    if not os.path.exists(index_path) or not os.path.exists(config_path):
        save_faiss_index(vectorstore, index_path, config_path)
    else:
        vectorstore2 = load_faiss_index(index_path, config_path)
        vectorstore.merge_from(vectorstore2)
        save_faiss_index(vectorstore, index_path, config_path)

    total_time = time.time() - start_time1
    print(f"총 소요시간 : {total_time:.2f}sec")
    print("임베딩 완료!")

def load_faiss_index(index_path, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    embeddings = HuggingFaceEmbeddings(
        model_name=config['model_name'],
        model_kwargs=config['model_kwargs'],
        encode_kwargs=config['encode_kwargs']
    )
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def get_text_update(path):
    start_time = time.time()
    doc_list = []

    # 경로가 디렉터리인지 파일인지 확인
    if os.path.isdir(path):
        docs = os.listdir(path)
        for doc in tqdm(docs, desc="문서 로드 진행중"):
            try:
                doc_list.extend(crop_and_load_pdf(os.path.join(path, doc)))
            except Exception as e:
                print(f"문서 로드 에러 {doc}: {e}")
                continue
    else:
        # 단일 파일 처리
        try:
            doc_list.extend(crop_and_load_pdf(path))
        except Exception as e:
            print(f"문서 로드 에러 {path}: {e}")

    total_time = time.time() - start_time
    print(f"문서 로드 : {total_time:.2f}sec")
    return doc_list

def crop_and_load_pdf(file_path):
    doc = fitz.open(file_path)
    cropped_docs = []
    for page_number in range(doc.page_count):
        try:
            page = doc.load_page(page_number)
            rect = page.rect
            crop_rect = fitz.Rect(rect.x0, rect.height * 0.1, rect.x1, rect.height * 0.9)
            page.set_cropbox(crop_rect)
            text = page.get_text("text")
            cropped_docs.append(Document(page_content=text))
        except Exception as e:
            print(f"Error processing page {page_number} of {file_path}: {e}")
            continue
    doc.close()
    return cropped_docs

def get_text_chunks(text_list):
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )
    chunks = text_splitter.split_documents(text_list)
    total_time = time.time() - start_time
    print(f"Chunk 분할 : {total_time:.2f}sec")
    return chunks

def get_vectorstore(text_chunks):
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vectordb = FAISS.from_documents(text_chunks, embeddings)
    total_time = time.time() - start_time
    print(f"Embedding : {total_time:.2f}sec")
    return vectordb

def save_faiss_index(vectordb, index_path, config_path):
    vectordb.save_local(index_path)
    config = {
        'model_name': vectordb.embeddings.model_name,
        'model_kwargs': vectordb.embeddings.model_kwargs,
        'encode_kwargs': vectordb.embeddings.encode_kwargs,
    }
    with open(config_path, 'w') as f:
        json.dump(config, f)

if __name__ == '__main__':
    main()
