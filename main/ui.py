import streamlit as st

pages = st.navigation([
    st.Page('../pages/local_base.py', title ='民法典相关问题咨询助手'),
    st.Page('../pages/input_base.py', title ='基于用户输入文档的聊天')
])

pages.run()