#Streamlit import
import streamlit as st
from streamlit_chat import message

#Importing chain and memory
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# All utility functions
import utils

from PIL import Image

#Intializing default session state
def initialize_session_state():

    st.session_state.setdefault('history', []) #for chat history
    st.session_state.setdefault('generated', ["Hello! I am here to provide answers to questions extracted from uploaded PDF files."]) #for ouput
    st.session_state.setdefault('past', ["Hello Buddy!"]) #for input

def create_conversational_chain(llm, vector_store):

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def display_chat(conversation_chain):
    
    reply_container = st.container() #To group the generated chat response
    container = st.container() #To group our chat input form


    with container:
        with st.form(key='chat_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask me questions from uploaded PDF", key='input')
            submit_button = st.form_submit_button(label='Send ⬆️')
        
        if submit_button and user_input:
            generate_response(user_input, conversation_chain)
    
    display_generated_responses(reply_container)


def generate_response(user_input, conversation_chain):

    with st.spinner('Spinning a snazzy reply...'):
        output = conversation_chat(user_input, conversation_chain, st.session_state['history'])

    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)

def conversation_chat(user_input, conversation_chain, history):

    result = conversation_chain.invoke({"question": user_input, "chat_history": history})
    history.append((user_input, result["answer"]))
    return result["answer"]


def display_generated_responses(reply_container):
   
    if st.session_state['generated']:
        with reply_container:
            #iterating over the indices of the 'generated' list stored in the session state
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=f"{i}_user", avatar_style="adventurer")
                message(st.session_state["generated"][i], key=str(i), avatar_style="bottts")

def main():

    # Step 1: Initialize session state
    initialize_session_state()
    
    st.title("Chat Bot")

    image = Image.open('chatbot.jpg')
    st.image(image, width=150)
    
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>

            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

    # Step 2: Initialize Streamlit
    st.sidebar.title("Upload Pdf")
    pdf_files = st.sidebar.file_uploader("", accept_multiple_files=True)
    
    # Step 3: Creating instance of Mistral 7B GGUF file format using llama.cpp    
    llm = utils.create_llm()

    #Step 4: Creating Vector Store and store uploaded Pdf file to Vector Database FAISS and return instance of vector store
    vector_store = utils.create_vector_store(pdf_files)

    if vector_store:
        #Step 5: If Vetor Store is created successfully then create the chain object
        chain = create_conversational_chain(llm, vector_store)

        #Step 6 - Display Chat to Web UI
        display_chat(chain)
    else:
        print('Initialzed App.')

if __name__ == "__main__":
    main()
