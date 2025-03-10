import heapq
import numpy as np
import ollama
from typing import List, Dict, Tuple, Any

class ConvoRAG:
    def __init__(self, documents: List[str], embedding_model: str = "nomic-embed-text", llm_model: str = "llama3.2"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.document_embeddings = [self.embed_text(doc) for doc in documents]
        self.conversation_history = []
        
    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using Ollama's specified embedding model."""
        response = ollama.embeddings(model=self.embedding_model, prompt=text)
        embedding = response["embedding"]
        return np.array(embedding)
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embedding vectors."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
    
    def topk(self, arr: List[float], k: int) -> List[int]:
        """Get the k largest elements and their indexes."""
        topk_indices = heapq.nlargest(k, range(len(arr)), key=lambda i: arr[i])
        return topk_indices
    
    def search(self, query: str, top_k: int = 5) -> Tuple[str, float]:
        """Search for the most relevant documents based on cosine similarity of embeddings."""
        query_embedding = self.embed_text(query)
        
        similarities = []
        for doc_embedding in self.document_embeddings:
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
        
        topk_indices = self.topk(similarities, top_k)
        
        result = '\n'.join([self.documents[i] for i in topk_indices])
        return result, similarities[topk_indices[0]]
    
    def generate_answer(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response using the provided system and user prompts."""
        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
        )
        return response['message']['content']
    
    def detect_query_type(self, query: str) -> str:
        """LLM to classify the query type into hotel-related, compliment, complaint, chitchat, or off-topic."""
        system_prompt = """
        You are an expert at classifying hotel customer queries. Your task is to categorize each query into EXACTLY ONE of these categories:
        
        1. "hotel-related" - Questions about hotel amenities, services, policies, bookings, cancellations, etc.
           Examples:
           - "What time does breakfast start?"
           - "Is there a fitness center?"
           - "Is the pool open year-round?"
           - "Is it complementary?" (when referring to a hotel service)
           - "BTW, Is it available for Airport?" (when referring to a hotel shuttle or taxi service)
           - "Whats the cost of it ?"
           - "Whats the price ?"
        
        2. "compliment" - Positive feedback or appreciation
           Examples:
           - "Thank you for the information"
           - "That's very helpful"
           - "You're doing a great job"
           - "You're very good assistant"
           - "You're providing relavant information"
        
        3. "complaint" - Negative feedback or dissatisfaction
           Examples:
           - "That's terrible service"
           - "I'm not happy with your answer"
           - "Not good job"
           - "You are responding slow"
           - "That's not the answer I expected"
        
        4. "chitchat" - General conversation not directly related to hotel inquiries
           Examples:
           - "How are you today?"
           - "What's your name?"
           - "Tell me a joke"
           - "How was your day?"
           - "Hey, nice to meet you!!"
        
        5. "off-topic" - Questions completely unrelated to hotels
           Examples:
           - "How do I fix my car?"
           - "What's the capital of France?"
           - "Can you write me a poem?"
           - "Who is current Indian Prime Minister?"
           - "Who is hosting next Olympics?"
        
        IMPORTANT: 
        - Follow-up questions about hotel services should be classified as "hotel-related" even if they are brief
        - Questions starting with "Is it..." or "Does it..." often refer to previously mentioned hotel services
        - Treat ambiguous queries that could reasonably be about hotel services as "hotel-related"
        
        Return ONLY the category name, with no explanation.
        """
        
        user_prompt = f"Classify this customer query: \"{query}\""
        
        response = self.generate_answer(system_prompt, user_prompt).lower().strip()
        
        # Extract the category from the response
        if "hotel" in response:
            return "hotel-related"
        elif "compliment" in response:
            return "compliment"  
        elif "complaint" in response:
            return "complaint"
        elif "chitchat" in response:
            return "chitchat"
        else:
            return "off-topic"
    
    def contextualize_query(self, current_query: str) -> str:
        """Enhance the current query with context from conversation history."""
        if not self.conversation_history:
            return current_query
        
        # If query is very detailed already, don't modify it
        if len(current_query.split()) > 10:
            return current_query
        
        # Build conversation history context with clear ordering
        history_context = "The conversation history is ordered from oldest to most recent:\n\n"
        for idx, (q, a) in enumerate(self.conversation_history[-5:]):
            history_context += f"Exchange {idx+1} (older):\nUser: {q}\nAssistant: {a}\n\n"
        
        system_prompt = """
        You are an expert at understanding conversational context in hotel customer service interactions.
        
        Your task is to analyze the current query in the context of the conversation history and determine:
        1. IF the current query is a follow-up question that needs context from previous exchanges
        2. If it is, WHICH previous topic it most likely refers to
        3. HOW to reformulate the query to be completely self-contained
        
        IMPORTANT RULES:
        - The conversation history is ordered from oldest to most recent
        - Brief queries like "What's the cost?" or "Is it available?" are almost always follow-ups
        - Use ONLY information explicitly mentioned in the conversation history
        - NEVER include information not present in the conversation
        - NEVER answer the query - just reformulate it
        - NEVER provide explanations outside the reformulated query
        - Keep reformulations concise (under 20 words when possible)
        - Reply with ONLY the reformulated query or the original query if no reformulation is needed
        
        EXAMPLES:
        1. History: User asked about fitness center, then laundry service. Current query: "What's the cost?"
           Response: "What is the cost of the laundry service?"
        
        2. History: User asked about airport shuttle, then breakfast, then pool. Current query: "Is it heated?"
           Response: "Is the hotel pool heated?"
        """
        
        user_prompt = f"""
        Conversation history:
        {history_context}
        
        Current query: "{current_query}"
        
        Reformulated query (if needed):
        """
        
        reformulated_query = self.generate_answer(system_prompt, user_prompt).strip()
        
        # Filter out any explanatory text the LLM might add
        if ":" in reformulated_query:
            reformulated_query = reformulated_query.split(":", 1)[1].strip()
        
        # If the reformulation seems excessive or contains phrases like "based on the conversation",
        # or if it looks like an answer rather than a question, fallback to original query
        problematic_phrases = [
            "based on", "according to", "from our conversation", 
            "as mentioned", "earlier you asked", "you asked about"
        ]
        
        if (len(reformulated_query.split()) > 25 or 
            any(phrase in reformulated_query.lower() for phrase in problematic_phrases) or
            "." in reformulated_query and "?" not in reformulated_query):
            return current_query
        
        # If reformulation is too similar to original, keep original
        if reformulated_query.lower() == current_query.lower():
            return current_query
            
        #print(f"Original query: {current_query}")
        #print(f"Reformulated query: {reformulated_query}")
        return reformulated_query
    
    def handle_non_hotel_query(self, query_type: str, query: str) -> str:
        """Handle compliments, complaints, chitchat, and off-topic queries using LLM."""
        system_prompt = """
        You are an AI assistant for a luxury hotel. Respond to the customer message appropriately based on its category.
        
        Keep your response:
        1. Professional and courteous
        2. Relatively brief (2-3 sentences)
        3. Empathetic when needed
        4. Gently redirecting to hotel topics when appropriate
        
        Do not include the category in your response.
        """
        
        user_prompt = f"""
        Message category: {query_type}
        Customer message: "{query}"
        
        Please respond appropriately to this message.
        """
        
        return self.generate_answer(system_prompt, user_prompt)
    
    def rag(self, query: str) -> str:
        """Main function to perform conversational document search and answer generation."""
        query_type = self.detect_query_type(query)
        #print(f"Query type: {query_type}")
        
        if query_type != "hotel-related":
            response = self.handle_non_hotel_query(query_type, query)
            self.conversation_history.append((query, response))
            return response
        
        # Contextualize the query based on conversation history
        contextualized_query = self.contextualize_query(query)
        
        # Retrieve relevant context
        context, _ = self.search(contextualized_query)
        #print(f'Retrieved context: \n{context}')
        
        # Generate the answer
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
        
        user_prompt = f"Based on the following context, please answer the question.\nIf the answer cannot be derived from the context, say \"I cannot answer this based on the provided context.\" \n\nContext:\n{context}\n\nQuestion: {contextualized_query}\n\nAnswer:\n"
        
        answer = self.generate_answer(system_prompt, user_prompt)
        
        # Update conversation history
        self.conversation_history.append((query, answer))
        
        return answer

def main():
    """
    Main function to create and run the conversational RAG system.
    """
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
    
    rag_system = ConvoRAG(documents)
    
    print("\n" + "="*50)
    print("Conversational Retrieval-Augmented Generation")
    print("="*50)
    
    while True:
        print("\n" + "-"*50)
        user_input = input("Please enter your query regarding the hotel: ")
        
        if user_input:
            answer = rag_system.rag(user_input)
            print(f"\nAssistant: {answer}")
        
        print("-"*50)
        continue_query = input("\nDo you want to ask another question? (y/Y to continue, any other key to quit): ")
        if continue_query.lower() != "y":
            print("\nExiting!")
            break

if __name__ == "__main__":
    main()
