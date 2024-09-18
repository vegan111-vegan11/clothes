import os
import json
import pandas as pd
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from flask_cors import CORS
import openai 
from flask import Flask, request, jsonify

app = Flask(__name__)
CORS(app)

# OpenAI API 키 설정
openai.api_key = "OpenAI API 키를 입력해주세요"

def load_faiss_index(index_path, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    embeddings = HuggingFaceEmbeddings(
        model_name=config['model_name'],
        model_kwargs=config['model_kwargs'],
        encode_kwargs=config['encode_kwargs']
    )
    vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vectordb

# 답변 생성
def generate_response_with_gpt4(query, documents):
    document_contents = []
    for doc in documents:
        product_name = doc.metadata.get('product_name', 'Unknown Product')
        document_contents.append(f"제품명: {product_name}\n설명: {doc.page_content}")
    
    combined_text = "\n\n".join(document_contents)
    prompt = f"사용자의 요청: {query}\n\n제품 목록:\n{combined_text}\n\n추천 제품:"

    # GPT-4를 위한 ChatCompletion 사용
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )

    chatbot_response = response['choices'][0]['message']['content'].strip()
    response = f"{chatbot_response}"
    
    return response

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    user_query = data.get('query')

    # FAISS 인덱스 로드
    try:
        embedding_path = 'C:/AAA_Analyze/vector'
        vectorstore = load_faiss_index(f"{embedding_path}/faiss_index", f"{embedding_path}/embeddings_config.json")
    except Exception as e:
        return jsonify({"error": f"Failed to load FAISS index: {str(e)}"}), 500

    try:
        retriever = vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={
                "k": 50,
                "embedding_fn_kwargs": {
                    "normalize_embeddings": True,  
                    "max_length": 512  
                }
            },
            verbose=True
        )
        search_results = retriever.get_relevant_documents(user_query)

        if not search_results:
            return jsonify({"response": "No relevant products found."}), 200
        else:
            response = generate_response_with_gpt4(user_query, search_results)
            return jsonify({"response": response}), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process query: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8503)
