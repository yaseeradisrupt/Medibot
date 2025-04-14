import streamlit as st
import requests
import os
import re

def main():
    st.title("MediBot")
    st.write("Welcome to MediBot! Please enter your symptoms below.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(fix_formatting(message["content"]))

    prompt = st.chat_input("Enter your symptoms here")

    if prompt:
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            response_text = st.empty()

            with st.spinner("typing..."):
                try:
                    FAST_API_ENDPOINT = os.getenv("FAST_API_ENDPOINT", "http://localhost:8005")
                    query_endpoint = FAST_API_ENDPOINT + "/query/"

                    with requests.post(query_endpoint, json={"query": prompt}, stream=True) as response:
                        streamed_text = ""
                        for chunk in response.iter_lines():
                            if chunk:
                                decoded_chunk = chunk.decode("utf-8")
                                # .strip()
                                  # Ensure words are properly spaced
                                if streamed_text and not streamed_text.endswith((" ", "\n")):
                                    streamed_text += " "  # Add space before appending new chunk
                                
                                streamed_text += decoded_chunk  
                                final_response = fix_formatting(streamed_text)
                                response_text.markdown(f"{final_response}")

                        # Store full response in session state
                        st.session_state.messages.append({"role": "assistant", "content": streamed_text})

                except Exception as e:
                    st.error(f"Error: {str(e)}")

def fix_formatting(text):
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)

    # Ensure space after numbers when followed by a letter
    text = re.sub(r'(?<=[0-9])(?=[A-Za-z])', ' ', text)

    # Fix extra spaces in words (handle words like "Fre quenturination")
    text = re.sub(r'(\b\w{2,})\s+(\w{2,})\s+(\w{2,}\b)', r'\1 \2 \3', text)

    # Ensure proper formatting for Markdown headers
    text = text.replace("** ", "**").replace(" **", "**")  # Fix bold formatting

    # Add newline before Markdown sections (Title, Description, etc.)
    text = re.sub(r'(\*\*.*?\*\*)', r'\n\n\1\n\n', text)

    # Remove unwanted spaces around hyphens in lists
    text = re.sub(r'-\s+', '- ', text)

    # Ensure newlines between bullet points
    text = text.replace("- ", "\n- ")

    return text.strip()
   
if __name__ == "__main__":
    print("Starting MediBot Frontend...")
    main()
