import os
import psycopg
from datetime import datetime
import streamlit as st
import rag as l 
import concurrent.futures
from dotenv import load_dotenv
load_dotenv()

DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")

def save_question(question):
    conn = psycopg.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    cursor = conn.cursor()
    cursor.execute("INSERT INTO Rag_Questions (question, timestamp) VALUES (%s, %s)", 
                   (question, datetime.now()))
    conn.commit()
    conn.close()

def format_response(response):
    response = response.replace("\n•", "\n\n•") 
    response = response.replace("\n", "\n\n") 
    return response

def fetch_response(prompt, l):
    raw_response = l.answer_query(prompt)
    processed_response = l.ans_rep(raw_response)
    return format_response(processed_response)

def main():
    st.title("Diabetes Chatbot by mPower")
    st.write("Ask me question about Diabetes")
    
    if '_initialized' not in st.session_state:
        st.session_state['_initialized'] = True
        # initialize_db()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    prompt = st.chat_input("Pass your prompt here")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': "user", 'content': prompt})
        save_question(prompt)  # Save the user question in the database
        
        with st.spinner("Generating Response..."):
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(fetch_response, prompt, l)
                formatted_response = future.result()
        
        st.chat_message('assistant').markdown(formatted_response)
        st.session_state.messages.append({'role': "assistant", 'content': formatted_response})

if __name__ == '__main__':
    main()