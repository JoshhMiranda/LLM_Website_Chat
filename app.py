import streamlit as st
import os 

from langchain_core.messages import AIMessage, HumanMessage

# used to import third party modules, in this case beautiful soup
from langchain_community.document_loaders import WebBaseLoader

# split text into chunks
from langchain.text_splitter import RecursiveCharacterTextSplitter

# chroma vectore store to store embeddings
from langchain_community.vectorstores import Chroma

# use open ai embeddings to embed text chunks
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

# load open ai keys
from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain




load_dotenv()

def get_vectorstore_from_url(url):
    # get text in document form
    loader = WebBaseLoader(url)
    document = loader.load()

    # split document into text chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    # store chunks in vectore store after embedding them using OpenAI embeddings
    # os.environ["OPENAI_API_KEY"] = ""
    vector_store = Chroma.from_documents(document_chunks, embedding = OpenAIEmbeddings())

    return vector_store



# embed entire chat history along with current user question
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    # create a reteriver using the vector we created
    retriever = vector_store.as_retriever()

    # create a prompt with all the chat history, current input of user and "given above conversation.....   "
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
        
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain



def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI()

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)



def get_response(user_query):
    # main page, create conversation chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversational_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversational_rag_chain.invoke({
            "chat_history" : st.session_state.chat_history,
            "input" : user_query
        })
    return response['answer']



# app config
st.set_page_config(page_title="Chat with website", page_icon=":compass:")
st.title("Chat with any website :compass:")


# app sidebar
with st.sidebar:
    st.header("Who do you want to talk to?")
    website_url = st.text_input("Enter website URL")

if website_url is None or website_url == "":
    st.info("Please enter a valid website URL")

else:
    # schema for langchain framework, keep track of chat history
    # session_state doesn't reload entire application when entereing a new query
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [ 
            AIMessage(content= "Hello, I am a bot. How can I help you?"),
        ]

    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)  




    # user input
    user_query = st.chat_input("Type your prompt...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)

        # append user input query to chat history, also append LLM response
        st.session_state.chat_history.append(HumanMessage(content= user_query))
        st.session_state.chat_history.append(AIMessage(content= response))



    # display the conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)




