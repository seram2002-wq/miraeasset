import streamlit as st
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer

# 웹 페이지 기본 설정
st.set_page_config(page_title="나만의 공시 검색 엔진", page_icon="🔍")
st.title("🤖 기업 공시 AI 검색 엔진")

# 캐싱: 페이지를 새로고침할 때마다 무거운 AI 모델을 다시 읽지 않도록 방어
@st.cache_resource
def load_engine():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    index = faiss.read_index('faiss_index/index.faiss')
    with open('faiss_index/metadata_final.pkl', 'rb') as f:
        metadata = pickle.load(f)
    return model, index, metadata

model, index, metadata = load_engine()
st.success(f"✅ 검색 준비 완료! (총 {index.ntotal}개의 데이터 탑재)")

# 검색창 만들기
query = st.text_input("궁금한 내용을 입력하세요 (예: 작년도 R&D 투자 비용은?)")

if query:
    st.markdown("---")
    query_vector = model.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, 3)
    
    # 결과 출력
    for i, idx in enumerate(indices[0]):
        score = distances[0][i]
        text = metadata[idx].get('text', '내용 없음')
        
        st.subheader(f"🏆 {i+1}위 검색 결과 (유사도: {score:.4f})")
        st.info(text)