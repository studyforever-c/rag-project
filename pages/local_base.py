import streamlit as st
import requests

Local_Base = 'http://localhost:8000/local_base/local_base'
Renew = 'http://localhost:8000/local_base/renew'

st.title('基于RAG的本地知识库问答系统')
st.text('(进入该页面可访问本地的知识库，将要处理的文件放在指定目录可更改知识库)')

if 'local_base' not in st.session_state:
    st.session_state.local_base = {}
local_base = st.session_state.local_base

with st.sidebar:
    if st.button('+ 开启新对话', width = 500):
        local_base.clear()
        requests.post(Renew)
if 'history' not in local_base:
    local_base['history'] = []
for msg in local_base['history']:
    with st.chat_message(msg['role']) : st.write(msg['content'])

question = st.chat_input('请输入你的问题')

if question:
    with st.chat_message('user') : st.write(question)
    with st.chat_message('assistant'):
        with st.spinner('正在思考中...'):
            response = requests.post(Local_Base, json = {'question': question}).json()
        st.write(response['history'][-1]['content'])
    local_base['history'] = response['history']