import streamlit as st
import requests

Input_Base = 'http://localhost:8000/input_base/input_base'
Upload = 'http://localhost:8000/input_base/upload'
Renew = 'http://localhost:8000/input_base/renew'


st.title('基于RAG的用户指定文档问答系统')
st.text('(进入该页面可以选择文件上传作为知识库)')

if 'input_base' not in st.session_state:
    st.session_state.input_base = {}
input_base = st.session_state.input_base

with st.sidebar:
    if st.button('+ 开启新对话', width = 500):
        input_base.clear()
        requests.post(Renew)
    st.divider()
    st.subheader('已上传文件列表：')

    if 'files' not in input_base:
        input_base['files'] = []
    for file in input_base['files']:
        st.write(file[1][0])

with st.bottom:
    if 'upload' not in input_base:
        input_base['upload'] = False
    if input_base['upload']:
        with st.spinner(f'知识库构建中...'):
            resp = requests.post(Upload, files = input_base['files'])
        st.success('知识库构建完成！')
        input_base['upload'] = False

    if 'no_file' not in input_base:
        input_base['no_file'] = False
    if input_base['no_file']:
        st.error('请至少选择一个文件进行上传！')

    l, r = st.columns([4, 1], vertical_alignment="bottom")
    with l:
        if 'key' not in input_base:
            input_base['key'] = 0
        files = st.file_uploader('请上传文件',
                                 ['md', 'pdf', 'docx'],
                                 True,
                                 key = input_base['key'])
    with r:
        if st.button('上传', type = 'primary'):

            if not files:
                input_base['no_file'] = True
            else:
                input_base['no_file'] = False
                input_base['files'].extend(
                    [("files", (file.name, file.read(), file.type)) for file in files]
                )
                input_base['upload'] = True
            input_base['key'] += 1
            st.rerun()



if 'history' not in input_base:
    input_base['history'] = []

for msg in input_base['history']:
    with st.chat_message(msg['role']) : st.write(msg['content'])
question = st.chat_input('请输入问题')
if question:
    with st.chat_message('user') : st.write(question)
    with st.chat_message('assistant'):
        with st.spinner('正在思考中...'):
            response = requests.post(Input_Base, json = {'question': question}).json()
        st.write(response['history'][-1]['content'])
    input_base['history'] = response['history']

