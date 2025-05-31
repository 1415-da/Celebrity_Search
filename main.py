import os
from dotenv import load_dotenv
import streamlit as st

from langchain import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

st.title('Celebrity Search Results')

input_text = st.text_input("Search for a celebrity...")

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("Groq API key not found. Please add it to your .env file as GROQ_API_KEY.")
else:
    # Initialize LLaMA3 LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

    # Prompt Templates
    first_input_prompt = PromptTemplate(
        input_variables=['name'],
        template="Tell me about celebrity {name}"
    )
    second_input_prompt = PromptTemplate(
        input_variables=['person'],
        template="When was {person} born?"
    )
    third_input_prompt = PromptTemplate(
        input_variables=['dob'],
        template="Mention 5 major events that happened around {dob} in the world"
    )
    awards_prompt = PromptTemplate(
        input_variables=['person'],
        template="List the major awards and recognitions received by {person}."
    )
    networth_prompt = PromptTemplate(
        input_variables=['person'],
        template="What is the estimated net worth of {person} in USD?"
    )
    social_prompt = PromptTemplate(
        input_variables=['person'],
        template="List the official or popular social media accounts and platforms used by {person}."
    )

    # Memory Buffers (optional, can help with context)
    person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
    dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
    descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')

    # Chains
    chain1 = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='person', memory=person_memory)
    chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob', memory=dob_memory)
    chain3 = LLMChain(llm=llm, prompt=third_input_prompt, verbose=True, output_key='description', memory=descr_memory)
    awards_chain = LLMChain(llm=llm, prompt=awards_prompt, verbose=True, output_key='awards')
    networth_chain = LLMChain(llm=llm, prompt=networth_prompt, verbose=True, output_key='networth')
    social_chain = LLMChain(llm=llm, prompt=social_prompt, verbose=True, output_key='social')

    # Sequential Chain including all steps
    parent_chain = SequentialChain(
        chains=[chain1, chain2, chain3, awards_chain, networth_chain, social_chain],
        input_variables=['name'],
        output_variables=['person', 'dob', 'description', 'awards', 'networth', 'social'],
        verbose=True
    )

    # Run chain and display results
    if input_text:
        result = parent_chain({'name': input_text})

        st.subheader("Generated Information:")
        st.write(result)

        with st.expander('Person Info'):
            st.info(result['person'])

        with st.expander('Birth Info'):
            st.info(result['dob'])

        with st.expander('Historical Events'):
            st.info(result['description'])

        with st.expander('Awards & Recognitions'):
            st.info(result['awards'])

        with st.expander('Net Worth'):
            st.info(result['networth'])

        with st.expander('Social Media'):
            st.info(result['social'])
