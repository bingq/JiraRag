import openai
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core import StorageContext

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize the embedding model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# Setup Chroma VectorStore and collection
db = chromadb.PersistentClient(path="./storage/chroma_cr_jira")
collection = db.get_or_create_collection("cr_jira_db")

# Setup storage context and vector store
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def chatbot():
    while True:
        # Step 1: Get input from the user
        user_query = input("Ask your question (or type 'exit' to quit): ")

        if user_query.lower() == "exit":
            print("Goodbye!")
            break

        # Step 2: Get embedding for the user query and search the vector database
        embedding = Settings.embed_model.get_text_embedding(user_query)
        results = collection.query(query_embeddings=[embedding], n_results=1)

        # Step 3: Extract context and metadata from the search result
        metadata = results['metadatas'][0][0]  # Get metadata for the first result
        context_str = "\n".join(results['documents'][0])  # Extract context (chunks)
        context_str_with_metadata = f"URL: {metadata['url']}\nTitle: {metadata['title']}\n\n{context_str}"

        #print(f"Extracted Context:\n{context_str_with_metadata}")

        # Step 4: Form the prompt for ChatGPT
        prompt = f"""
       "You have very comprehensive knowledge and deep insights into cybersecurity, network, and operating system domains.\n"
        "Always answer the query using only the provided context information, and not prior knowledge.\n"
        "Some rules to follow: \n"
        "1. Always provide the Jira ticket URL when answering with information from any Jira ticket.\n"
        "2. Using both the context information and your own knowledge.\n"
        "3. Always make sure to include the URL of the Jira ticket provided in the context.\n"
        "4. When there's no context, use your own knowledge.\n"
        Context information is below:
        -----------------
        {context_str_with_metadata}
        -----------------
        Answer the question: {user_query}
        """

        # Step 5: Send the prompt to ChatGPT using the new API
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": prompt}]
        )
        
        # Step 6: Print the response from ChatGPT
        #print(f"Response from ChatGPT:\n{response['choices'][0]['message']['content'].strip()}\n")
        print(response.choices[0].message.content)

# Ensure OpenAI API key is set
openai.api_key = "REMOVED"

# Now, run the chatbot
chatbot()
