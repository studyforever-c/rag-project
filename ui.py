import streamlit as st
import requests
from config import ASK

st.title('基于RAG的本地知识库问答系统')
if 'history' not in st.session_state:
    st.session_state.history = []
for msg in st.session_state.history:
    with st.chat_message(msg['role']) : st.write(msg['content'])

question = st.chat_input('请输入你的问题')

if question:
    with st.chat_message('user') : st.write(question)
    with st.chat_message('assistant'):
        with st.spinner('正在思考中...'):
            response = requests.post(ASK, json = {'question': question}).json()
        st.write(response['history'][-1]['content'])
    st.session_state.history = response['history']