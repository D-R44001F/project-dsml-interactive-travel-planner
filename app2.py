import chromadb
import openai

# Load ChromaDB clients for each folder
news_client = chromadb.PersistentClient(path="./chromadb")
municipalities_client = chromadb.PersistentClient(path="./chromadb_municipalities")
landmarks_client = chromadb.PersistentClient(path="./chromadb_landmarks")

# Function to load collections
def load_chromadb_collection(client, collection_name):
    try:
        collection = client.get_collection(collection_name)
        print(f"Successfully loaded collection: {collection_name}")
        return collection
    except Exception as e:
        print(f"Error loading collection {collection_name}: {e}")
        return None

# Load collections
news_collection = load_chromadb_collection(news_client, "news_articles")
municipalities_collection = load_chromadb_collection(municipalities_client, "municipalities")
landmarks_collection = load_chromadb_collection(landmarks_client, "landmarks")

# Set up OpenAI API Key
# Read API key from file
api_key_path = "API_Key.txt"
with open(api_key_path, "r") as file:
    OPENAI_API_KEY = file.read().strip()

openai.api_key = OPENAI_API_KEY

# Function to perform retrieval from the collections (RAG)
def retrieve_relevant_info(query, collection):
    # Query the collection with the user's question to get relevant documents
    results = collection.query(query_texts=[query], n_results=3)  # Adjust n_results as needed
    return results

# Example function for querying collections and getting responses from the LLM
def chat_with_llm(user_input):
    context = []

    # Retrieve relevant info from each collection
    if municipalities_collection:
        municipalities_info = retrieve_relevant_info(user_input, municipalities_collection)
        context.append(municipalities_info)
    
    if landmarks_collection:
        landmarks_info = retrieve_relevant_info(user_input, landmarks_collection)
        context.append(landmarks_info)
    
    if news_collection:
        news_info = retrieve_relevant_info(user_input, news_collection)
        context.append(news_info)

    # Extract text from results while filtering out None values
    combined_context = "\n".join(
        str(doc) for docs in context if docs and "documents" in docs 
        for doc in docs["documents"][0] if doc is not None
    )

    # Call OpenAI API
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Use "gpt-4" if needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": combined_context},
            ]
        )
        model_reply = response.choices[0].message.content
        return model_reply
    except Exception as e:
        return f"Error: {e}"



# Streamlit UI for interaction
import streamlit as st

st.title("Chat with ChatGPT using RAG system")

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": "You are a helpful assistant, always happy to help."}]

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] != "system":  # Skip displaying system message
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

# User input
user_input = st.chat_input("Ask me anything...")
if user_input:
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Get the response from the LLM
    model_reply = chat_with_llm(user_input)

    # Append AI response
    st.session_state.messages.append({"role": "assistant", "content": model_reply})

    # Display AI response
    with st.chat_message("assistant"):
        st.markdown(model_reply)

