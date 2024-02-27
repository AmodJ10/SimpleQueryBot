## Integrate our code with api

import os
from constant import HUGGINGFACE_API_TOKEN
from langchain_community.llms import HuggingFaceHub
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

import streamlit as st

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_TOKEN
repo_id = "google/gemma-7b"

# Streamlit framework

st.title("Physics Search results")

input_text = st.text_input("Search the topic you want")
button = st.button("Generate Prompt")

# Prompt Template

first_input_prompt = PromptTemplate(
    input_variables = ['topic'],
    template = "Tell me about this {topic} concept in Physics in detail"
)

llm = HuggingFaceHub(model_kwargs = {"temperature" :  0.6}  , repo_id = repo_id)

chain1 = LLMChain(llm  = llm , prompt=first_input_prompt , verbose=True , output_key='concept')

second_input_prompt = PromptTemplate(
    input_variables = ['topic'],
    template = "Tell me some weird facts about this {topic}"
)

chain2 = LLMChain(llm = llm , prompt=second_input_prompt , verbose=True , output_key='facts')


parentChain = SequentialChain(chains=[chain1 , chain2] , input_variables=['topic'] , output_variables=['concept' , 'facts'] , verbose=True)
if input_text:
    if button:
        answer = parentChain({"topic" : input_text})
        print(answer)
        st.write(answer)