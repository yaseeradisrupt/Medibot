import os
import streamlit as st
import requests
from load_dotenv import load_dotenv 
load_dotenv()


def main():
    st.title("MediBot")
    st.write("Welcome to MediBot! Please enter your symptoms below.")
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        st.chat_message(message['author']).markdown(message['text'])

    prompt = st.chat_input("Enter your symptoms here")
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'author': 'user', 'text': prompt})
        with st.spinner("typing..."):  # Display a spinner
            try:
                FAST_API_ENDPOINT = os.getenv("FASt_API_ENDPOINT", "http://localhost:8005")
                query_endpoint=FAST_API_ENDPOINT + '/query/'
                response = requests.post(query_endpoint, json={"query": prompt})
                if response.status_code == 200:
                    data = response.json()
                    if(data['result'] == 'error'):
                        st.error(data['message'])
                    result=data['data']['response']  
                    st.chat_message('assistant').markdown(result)
                    st.session_state.messages.append({'author': 'assistant', 'text':result})
                else:
                    st.error("Failed to get response from the server.")
            except Exception as e:
                st.error(str(e))

if __name__ == "__main__":
    main()