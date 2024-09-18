import tiktoken
import pandas as pd
import os
import json
import time

from sqlalchemy import create_engine, text
import mysql.connector as mysql

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

def main():
    start_time = time.time()
    schema_name = 'clothes'
    products_table_name = 'products'
    product_details_table_name = 'product_details'
    
    # DB 연결 정보 입력
    db_info = {
        "user": "root",
        "password": "4487",
        "host": "localhost",
        "port": "3306",
        "dbname": "clothes"
    }
    
    # 메타데이터 로드
    sql = f'''
    SELECT *
      FROM `{schema_name}`.`{products_table_name}` AS A
      JOIN `{schema_name}`.`{product_details_table_name}` AS B ON A.`product_id` = B.`product_id`;
    '''
    df_data = load_from_mysql(sql, db_info)

    # 텍스트와 시퀀스 가져오기
    text_chunks = get_text_chunks(df_data)
    vectorstore = get_vectorstore(text_chunks)

    # 벡터 저장 
    index_path = "C:/AAA_Analyze/vector/faiss_index"
    config_path = "C:/AAA_Analyze/vector/embeddings_config.json"
    save_faiss_index(vectorstore, index_path, config_path)

    print(f"총 소요시간 : {time.time() - start_time:.2f}sec")
    print("Embedding Completed!")

def load_faiss_index(index_path, config_path):
    # Embedding Config 로드
    with open(config_path, 'r') as f:
        config = json.load(f)
    # Embedding 재입력
    embeddings = HuggingFaceEmbeddings(
        model_name=config['model_name'],
        model_kwargs=config['model_kwargs'],
        encode_kwargs=config['encode_kwargs']
    )
    # Faiss 객체 생성
    vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectordb

def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding("cl100k_base")
    return len(tokenizer.encode(text))

def get_text_chunks(df_data):
    start_time = time.time()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=tiktoken_len
    )
    
    documents = []
    
    for _, row in df_data.iterrows():
        try:
            page_content = f"Style: {row['style_descriptor']}, Price: {row['price']}, Color: {row['color']}, Size: {row['size']}, Stock: {row['stock_quantity']}"
            metadata = {
                'product_id': row['product_id'],
                'product_name': row['product_name'],
                'category': row['category']
            }
            
            documents.append(Document(page_content=page_content, metadata=metadata))
        except KeyError as e:
            print(f"Missing key: {e}")
    
    chunks = text_splitter.split_documents(documents)
    print(f"Chunk 분할 : {time.time() - start_time:.2f}sec")
    return chunks

def get_vectorstore(text_chunks):
    start_time = time.time()
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        model_kwargs={'device': 'cuda'},
        encode_kwargs={'normalize_embeddings': True}
    )
    try:
        vectordb = FAISS.from_documents(text_chunks, embeddings)
    except IndexError as e:
        print("임베딩 생성 중 오류 발생:", e)
        raise
    print(f"Embedding : {time.time() - start_time:.2f}sec")
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

def load_from_mysql(sql_query, db_info):
    connection = mysql.connect(
        user=db_info['user'], 
        password=db_info['password'], 
        host=db_info['host'], 
        database=db_info['dbname'], 
        port=db_info['port']
    )
    df = pd.read_sql(sql_query, connection)
    connection.close()
    return df

if __name__ == '__main__':
    main()
