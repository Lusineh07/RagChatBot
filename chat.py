import os
import pickle
import requests
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import Document
import json

# Set NVIDIA API key (replace with your actual API key)
os.environ['NVIDIA_API_KEY'] = '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'

# Feedback file path
FEEDBACK_FILE = "feedback.json"

# Function to fetch and extract text content and images from a specific div class on a website using Requests
def fetch_website_content(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find the specific div with class 'page-body-wrap'
        page_body_wrap = soup.find('div', class_='page-body-wrap')
        
        if not page_body_wrap:
            st.warning(f"No content found in div 'page-body-wrap' on {url}.")
            return None, [], url  # Return the URL along with content and images
        
        # Extract text content from the div
        text_content = page_body_wrap.get_text(separator='\n\n').strip()
        
        # Extract image URLs
        image_urls = []
        images = page_body_wrap.find_all('img')
        for img in images:
            src = img.get('src')
            if src:
                image_urls.append(src)
        
        return text_content, image_urls, url
    
    except Exception as e:
        st.error(f"Error fetching {url}: {str(e)}")
        return None, [], url

# Load feedback from file
def load_feedback():
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            return json.load(f)
    return {}

# Save feedback to file
def save_feedback(feedback):
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(feedback, f)

# Main function to process the website and chat interface
def main():

    # Load feedback
    feedback = load_feedback()

    # Load URLs from file
    website_urls = []
    try:
        with open("urls.txt", "r") as file:
            for line in file:
                url = line.strip()
                if url:
                    website_urls.append(url)
    except FileNotFoundError:
        st.error("File 'urls.txt' not found.")
        return
    except Exception as e:
        st.error(f"Error loading URLs from 'urls.txt': {str(e)}")
        return

    # Initialize lists to store fetched content and URLs
    documents = []

    # Fetch content from each website in parallel
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_website_content, url) for url in website_urls]
        for future in futures:
            text_content, image_urls, fetched_url = future.result()
            if text_content:
                documents.append(Document(page_content=text_content, metadata={"source": fetched_url, "images": image_urls}))

    # Ensure at least one document is fetched
    if not documents:
        st.error("No content fetched from the websites.")
        return

    # Load or create vector store
    vector_store_path = "vectorstore.pkl"
    if os.path.exists(vector_store_path):
        with open(vector_store_path, "rb") as f:
            vectorstore = pickle.load(f)
    else:
        document_embedder = NVIDIAEmbeddings(model="nvolveqa_40k", model_type="passage")
        vectorstore = FAISS.from_documents(documents, document_embedder)
        with open(vector_store_path, "wb") as f:
            pickle.dump(vectorstore, f)
        st.success("Vector store created successfully.")

    # Initialize NVIDIA models
    llm = ChatNVIDIA(model="mixtral_8x7b")
    prompt_template = ChatPromptTemplate.from_messages(
        [("system", "You are a helpful AI Assistant named..."), ("user", "{input}")]
    )
    chain = prompt_template | llm | StrOutputParser()

    # Chat interface
    st.subheader("Chat with your AI Assistant!")

    # Input form fixed at the top
    with st.form(key='chat_form', clear_on_submit=True):
        user_input = st.text_input("Ask your question:", key="user_input")
        submitted = st.form_submit_button("Send")

    # Handle user input and generate response
    if submitted and user_input and vectorstore is not None:
        st.session_state.messages.insert(0, {"role": "user", "content": user_input})

        # Retrieve relevant documents from vectorstore
        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(user_input)

        # Combine relevant texts into context
        context = "\n\n".join(doc.page_content for doc in relevant_docs)
        augmented_user_input = f"Context: {context}\n\nQuestion: {user_input}"

        # Invoke the AI assistant with augmented input
        response = chain.invoke({"input": augmented_user_input})

        # Find the URL corresponding to the most relevant fetched content
        if relevant_docs:
            relevant_url = relevant_docs[0].metadata["source"]
            response_with_url = f"{response}\n\nInformation from: [{relevant_url}]({relevant_url})"
        else:
            response_with_url = response

        st.session_state.messages.insert(0, {"role": "assistant", "content": response_with_url})
        st.experimental_rerun()

    # Display chat messages
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                col1, col2 = st.columns([1, 1], gap="small")
                with col1:
                    if st.button("👍", key=f"like_{i}"):
                        st.success("You liked the response!")
                        feedback[str(i)] = "like"
                        save_feedback(feedback)
                with col2:
                    if st.button("👎", key=f"dislike_{i}"):
                        st.error("You disliked the response.")
                        feedback[str(i)] = "dislike"
                        save_feedback(feedback)

# Run the main function
if __name__ == "__main__":
    main()
