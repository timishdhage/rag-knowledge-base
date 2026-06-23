import streamlit as st
import requests

st.set_page_config(page_title='RAG Knowledge Base', layout='wide')
st.title('RAG Knowledge Base')

api = st.text_input('API URL', 'http://localhost:8000')
question = st.text_area('Ask a question')

if st.button('Ask') and question:
    r = requests.post(f'{api}/v1/ask', json={'question': question}, timeout=60)
    data = r.json()
    st.subheader('Answer')
    st.write(data.get('answer'))
    st.subheader('Retrieved Chunks')
    st.json(data.get('retrieved'))
