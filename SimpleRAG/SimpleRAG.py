#import logging
import heapq
import numpy as np
import ollama

#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def embed_text(text: str, embedding_model="nomic-embed-text"):
    """Embed text using Ollama's specified embedding model (default: nomic-embed-text)."""
    response = ollama.embeddings(model=embedding_model, prompt=text)
    embedding = response["embedding"]
    return np.array(embedding)

def cosine_similarity(embedding1, embedding2):
    """Compute cosine similarity between two embedding vectors."""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)

def topk(arr, k):
    """Get the k largest elements and their indexes."""
    topk_indices = heapq.nlargest(k, range(len(arr)), key=lambda i: arr[i])
    return topk_indices

def search(query, documents, document_embeddings, top_k=5):
    """Search for the most relevant document based on the cosine similarity of embeddings."""
    query_embedding = embed_text(query)

    similarities = []
    for doc_embedding in document_embeddings:
        similarity = cosine_similarity(query_embedding, doc_embedding)
        similarities.append(similarity)

    topk_indices = topk(similarities, top_k)

    # Print all top_k documents and their similarity scores
    # [logging.info(f"Document: {documents[i]}\nSimilarity Score: {similarities[i]}\n") for i in topk_indices]

    result = '\n'.join([documents[i] for i in topk_indices])
    return result, similarities[topk_indices[0]]

def generate_answer(system_prompt, user_prompt, model='llama3.2'):
    """Generate a response using the provided system and user prompts."""
    response = ollama.chat(
        model=model,
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]
    )

    return response['message']['content']

def rag(query):
    """Main function to perform document search and answer generation."""
    context, similarity = search(query, documents, document_embeddings)
    print (f'Retreived context: \n {context}')
    
    system_prompt = '''You are an AI assistant for a hotel booking website. You have access to a collection of detailed information about the hotel's services, booking, cancellation, and policies. Your task is to help users by providing them with relevant and accurate answers to their queries.

    The information you have includes:

    Hotel Details: Information about the hotel's location, amenities, room features, check-in/check-out times, etc.
    Booking Policy: Rules for securing and modifying bookings, payment requirements, minimum stay periods, etc.
    Cancellation Policy: Information about free cancellations, penalties for late cancellations, no-shows, and how to cancel bookings.
    Refund Policy: Details about refund processing times, conditions for refunds, and how refunds are calculated.
    Food and Dining: Details about the food services at the hotel, including breakfast times, restaurant offerings, room service availability, and bar hours.
    Additional Amenities: Information about parking, airport shuttle services, laundry, business center, and other hotel services.

    If a user's question cannot be answered based on the provided context, always guide them to the helpdesk by saying:

    "I cannot answer this based on the provided context. For more information, please contact the helpdesk at the number provided on our website."

    Respond in a clear, concise, and polite manner. When a user asks a question, refer to the relevant details from the list below to provide a well-informed answer. If unsure, always direct the user to the helpdesk for further assistance.
    '''
    
    user_prompt = f"Based on the following context, please answer the question.\nIf the answer cannot be derived from the context, say \"I cannot answer this based on the provided context.\" \n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:\n"
    
    answer = generate_answer(system_prompt, user_prompt)
    return answer

def handle_user_input():
    """Function to handle user input and append to messages."""
    return input("Please enter your query regarding the hotel: ")

def main():
    """
    Main function to perform semantic search in a loop.
    """
    while True:
        print("\n" + "="*50)
        print("Basic Retrieval-Augmented Generation")
        print("="*50)
        print("\n" + "-"*50)
        user_input = handle_user_input()
        if user_input:
            # Perform search and generate the answer
            answer = rag(user_input)
            print(f"\nAnswer: {answer}")

        print("-"*50)
        continue_query = input("\nDo you want to search again? (y/Y to continue, any other key to quit): ")
        if continue_query.lower() != "y":
            print("\nExiting!")
            break

if __name__ == "__main__":
    documents = [
        # hotel_details
        "The Grand Palace Hotel offers luxury rooms with a view of the city skyline.",
        "Our hotel is located in the heart of downtown, just a 10-minute walk from the main shopping district.",
        "Each room comes equipped with a flat-screen TV, free Wi-Fi, and premium toiletries.",
        "The hotel features a rooftop pool, fitness center, and spa services for all guests.",
        "Check-in time starts at 3:00 PM and check-out is at 11:00 AM.",
        # booking_policy
        "To secure your reservation, a credit card is required at the time of booking.",
        "Rooms can be booked up to six months in advance on our website.",
        "All reservations must be guaranteed with a valid payment method.",
        "The hotel accepts bookings for a minimum stay of two nights during peak seasons.",
        "If you'd like to modify your booking, please contact the front desk at least 24 hours before your check-in date.",
        # cancellation_policy
        "Guests can cancel their reservation free of charge up to 48 hours before the scheduled check-in date.",
        "Cancellations made within 24 hours of check-in will incur a 50% charge of the total booking cost.",
        "No-show guests will be charged the full amount of their stay.",
        "To cancel your reservation, you can either call the hotel directly or cancel through our website.",
        # refund_policy
        "Refunds are processed to the original payment method within 7 business days.",
        "Refunds are only available for cancellations made within the allowed time frame, as per the hotel's cancellation policy.",
        "If a guest has prepaid for their stay and cancels within the eligible period, a full refund will be issued.",
        "Refunds for partial bookings will be calculated based on the number of nights stayed and the cancellation policy.",
        # food_and_dining
        "Our hotel offers complimentary breakfast every morning from 7:00 AM to 10:00 AM.",
        "The on-site restaurant serves a range of international dishes, with vegetarian and vegan options available.",
        "Room service is available 24/7 for all guests.",
        "Guests can enjoy afternoon tea in the lobby lounge from 3:00 PM to 5:00 PM.",
        "The hotel bar serves a variety of cocktails, wine, and snacks from 12:00 PM to midnight.",
        # additional_amenities
        "Free parking is available for all guests staying at the hotel.",
        "We offer airport shuttle services for an additional fee. Please book in advance.",
        "The hotel provides laundry services, including dry cleaning and pressing.",
        "There is a business center on-site, with printing and copying services available for guests."
    ]

    document_embeddings = [embed_text(doc) for doc in documents]
    main()
