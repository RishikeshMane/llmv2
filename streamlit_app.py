import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

from transformers import pipeline
from fastapi import FastAPI

checkpoint = "LaMini-T5-61M"

model = pipeline('text2text-generation', model = checkpoint)

result = []

def generate_response(txt):
    input_prompt = txt
    generated_text = model(input_prompt, max_length=512, do_sample=True)[0]['generated_text']
    #chain = load_summarize_chain(llm, chain_type='map_reduce')
    return generated_text

st.set_page_config(page_title='Group1 Summarization App')
st.title('Youtube data Summarization with LamiNi')

txt_input = st.text_area('Enter your text', '', height=200)

result1=[]
with st.form('summarize_form', clear_on_submit=True):
    submitted = st.form_submit_button('Submit')
    if submitted:
        with st.spinner('Calculating...'):
            response = generate_response(txt_input)
            result1.append(response)
            print(result1)
if len(result1):
    st.info(response)
