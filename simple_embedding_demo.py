import streamlit as st
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

openai.api_key =  st.secrets["api_secret"]



# Function to generate embeddings using OpenAI
def get_embedding(question, openai_api_key):
    openai.api_key = openai_api_key
    response = openai.Embedding.create(
        input=question,
        engine="text-embedding-ada-002"
    )
    return np.array(response['data'][0]['embedding'])

# Function to calculate cosine similarity
def calculate_similarity(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Streamlit UI
st.title("Simple FAQ Chatbot")



# Text boxes for FAQ
faq_question = st.text_input("Enter FAQ Question")
faq_answer = st.text_area("Enter FAQ Answer")

# Text box for user's question
user_question = st.text_input("Ask your question")

if st.button("Submit"):
    if faq_question and user_question:
        try:
            # Get embeddings
            faq_embedding = get_embedding(faq_question, openai_api_key)
            user_embedding = get_embedding(user_question, openai_api_key)

            # Calculate similarity
            similarity = calculate_similarity(faq_embedding, user_embedding)

            # Determine if the question is relevant and respond
            if similarity > 0.8:  # Threshold for relevance
                st.write(faq_answer)
            else:
                st.write("Please ask a relevant question.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please fill in all fields and submit.")

