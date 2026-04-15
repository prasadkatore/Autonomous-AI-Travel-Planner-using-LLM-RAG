from rag import load_documents, create_vector_store, retrieve
import streamlit as st
import ollama

st.title("✈️ AI Travel Planner")

# User inputs
destination = st.text_input("Enter destination")
budget = st.number_input("Enter budget (€)", min_value=100)
days = st.number_input("Number of days", min_value=1)
docs = load_documents("data")
index, stored_docs = create_vector_store(docs)

if st.button("Generate Plan"):

    query = f"{destination} travel plan budget {budget} for {days} days"

    retrieved_info = retrieve(query, index, stored_docs)

    prompt = f"""
You are a travel planner.

Use the following travel information:
{retrieved_info}

Plan a {days}-day trip to {destination} with a total budget of EUR {budget}.

STRICT RULES:
- Total cost must NOT exceed EUR {budget}

FORMAT:
1. Budget Breakdown
2. Day-by-Day Itinerary
3. Tips
"""

    response = ollama.chat(
        model="llama3",
        messages=[{"role": "user", "content": prompt}]
    )

    st.subheader("Your Travel Plan")
    st.write(response['message']['content'])