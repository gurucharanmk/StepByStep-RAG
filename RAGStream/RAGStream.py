import streamlit as st
import heapq
import numpy as np
import ollama
import time
import re
from typing import List, Dict, Tuple, Any, Iterator, Union
from datetime import datetime


class ConvoRAG:
    def __init__(
        self,
        documents: List[str],
        embedding_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2",
    ):
        self.documents = documents
        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Add debug print
        st.write(f"Initializing with {len(documents)} document chunks")
        st.write(
            f"Using embedding model: {embedding_model}, LLM model: {llm_model}"
        )

        with st.spinner("Embedding documents..."):
            self.document_embeddings = [self.embed_text(doc) for doc in documents]

        self.conversation_history = []

    def embed_text(self, text: str) -> np.ndarray:
        """Embed text using Ollama's specified embedding model."""
        try:
            response = ollama.embeddings(model=self.embedding_model, prompt=text)
            embedding = response["embedding"]
            #Print embedding dimension
            embedding_array = np.array(embedding)
            st.write(
                f"Embedded text '{text[:50]}...' → {embedding_array.shape} dimensions"
            )
            return embedding_array
        except Exception as e:
            st.error(f"Error in embedding: {str(e)}")
            # Return a zero vector as fallback
            return np.zeros(768)  # Typical embedding size 1536 for text-davinci, but we are usng nomic

    def cosine_similarity(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two embedding vectors."""
        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)

            # Handle zero vectors
            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return similarity
        except Exception as e:
            st.error(f"Error in cosine similarity calculation: {str(e)}")
            return 0.0

    def topk(self, arr: List[float], k: int) -> List[int]:
        """Get the k largest elements and their indexes."""
        if not arr:
            return []

        # Ensure k is not larger than the array length
        k = min(k, len(arr))

        try:
            topk_indices = heapq.nlargest(k, range(len(arr)), key=lambda i: arr[i])
            #Print top scores
            top_scores = [arr[i] for i in topk_indices]
            st.write(
                f"Top {k} similarity scores: {[f'{score:.4f}' for score in top_scores]}"
            )
            return topk_indices
        except Exception as e:
            st.error(f"Error in topk calculation: {str(e)}")
            # Below return is wrong, however prevents app from crashing
            return list(range(min(k, len(arr))))

    def search(self, query: str, top_k: int = 5) -> Tuple[str, float]:
        """Search for the most relevant documents based on cosine similarity of embeddings."""
        if not self.documents or not self.document_embeddings:
            return "No documents available.", 0.0

        try:
            st.write(f"Searching for: '{query}'")
            query_embedding = self.embed_text(query)

            similarities = []
            for doc_embedding in self.document_embeddings:
                similarity = self.cosine_similarity(query_embedding, doc_embedding)
                similarities.append(similarity)

            if not similarities:
                return "No similarities found.", 0.0

            topk_indices = self.topk(similarities, top_k)

            if not topk_indices:
                return "No relevant documents found.", 0.0
            
            result = "\n".join([self.documents[i] for i in topk_indices])
            return result, similarities[topk_indices[0]] if topk_indices else 0.0
        except Exception as e:
            st.error(f"Error in document search: {str(e)}")
            return "Error occurred while searching documents.", 0.0

    def generate_answer(
        self, system_prompt: str, user_prompt: str, is_stream: bool = True
    ) -> Any:
        """Generate a response using the provided system and user prompts."""
        try:
            #Show abbreviated system prompt and user prompt
            st.write(f"Generating response with model: {self.llm_model}")

            generation_start = time.time()
            response = ollama.chat(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=is_stream,
            )

            # For non-streaming responses, show timing
            if not is_stream:
                generation_time = time.time() - generation_start
                st.write(
                    f"Response generated in {generation_time:.2f} seconds"
                )

            return response if is_stream else (response["message"]["content"])
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            if is_stream:
                return iter([{"message": {"content": error_msg}}])
            else:
                return error_msg

    def detect_query_type(self, query: str, relevance_threshold: float = 0.75) -> str:
        """LLM to classify the query type into hotel-related, compliment, complaint, chitchat, or off-topic."""
        st.write(f"Detecting query type for: '{query}'")

        # Use LLM for initial classification
        system_prompt = """
        You are an expert at classifying hotel customer queries. Your task is to categorize each query into EXACTLY ONE of these categories:
        
        1. "hotel-related" - Questions about hotel amenities, services, policies, bookings, cancellations, etc.
        Examples:
        - "What time does breakfast start?"
        - "Is there a fitness center?"
        - "Is the pool open year-round?"
        - "Is it complementary?" (when referring to a hotel service)
        - "BTW, Is it available for Airport?" (when referring to a hotel shuttle or taxi service)
        - "Whats the cost of it?"
        - "Whats the price?"
        - "How much is the room?"
        - "Where can I park?"
        - "Do I need to pay extra for wifi?"
        
        2. "compliment" - Positive feedback or appreciation
        Examples:
        - "Thank you for the information"
        - "That's very helpful"
        - "You're doing a great job"
        - "You're very good assistant"
        - "You're providing relevant information"
        
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
        
        IMPORTANT RULES: 
        - Follow-up questions about hotel services should be classified as "hotel-related" even if they are brief
        - Questions starting with "Is it..." or "Does it..." or "How much..." often refer to previously mentioned hotel services
        - Treat ambiguous queries that could reasonably be about hotel services as "hotel-related"
        - WHEN IN DOUBT, classify as "hotel-related"
        
        Return ONLY the category name, with no explanation.
        """

        user_prompt = f'Classify this customer query: "{query}"'

        try:
            initial_classification = (
                self.generate_answer(system_prompt, user_prompt, is_stream=False)
                .lower()
                .strip()
            )

            # Extract the category from the response
            if "hotel" in initial_classification:
                classification = "hotel-related"
            elif "compliment" in initial_classification:
                classification = "compliment"
            elif "complaint" in initial_classification:
                classification = "complaint"
            elif "chitchat" in initial_classification:
                classification = "chitchat"
            else:
                classification = "off-topic"

            #Show classification
            st.write(f"Initial classification: '{classification}'")

            # If already classified as hotel-related, no need for semantic verification
            if classification == "hotel-related":
                return classification

            # For potentially misclassified queries (chitchat and off-topic),
            # do semantic search to see if we have relevant documents
            if classification in ["chitchat", "off-topic"]:
                # Try semantic search on the original query
                relevant_docs, top_similarity = self.search(query, top_k=1)

                # Log the classification and relevance score for debugging
                st.write(
                    f"Query: '{query}' | Initial classification: {classification} | Top similarity: {top_similarity:.4f}"
                )

                # If we find highly relevant documents above threshold, override to hotel-related
                if top_similarity >= relevance_threshold:
                    st.write(
                        f"Reclassifying to hotel-related due to high document similarity: {top_similarity:.4f}"
                    )
                    return "hotel-related"

            return classification

        except Exception as e:
            st.error(f"Error in query classification: {str(e)}")
            # Default to hotel-related when classification fails
            return "hotel-related"

    def contextualize_query(self, current_query: str) -> str:
        """Enhance the current query with context from conversation history."""
        if not self.conversation_history:
            return current_query

        st.write(f"Contextualizing query: '{current_query}'")

        # If query is very detailed already, don't modify it
        if len(current_query.split()) > 10:
            st.write(f"Query is detailed enough, no contextualization needed")
            return current_query

        # Build conversation history context with clear ordering and numbering
        history_context = (
            "The conversation history is ordered from oldest to most recent:\n\n"
        )
        for idx, (q, a) in enumerate(self.conversation_history[-5:]):
            history_context += f"Exchange {idx + 1}:\nUser: {q}\nAssistant: {a}\n\n"
 
        system_prompt = """You are a query reformulation system. Your job is to take a user's query and reformulate it to be self-sufficient by incorporating relevant context from the conversation history, while preserving the original intent.

RULES:
1. Create a single, concise sentence that captures both the current query and relevant context
2. Never add your own opinions, reasoning, explanations, or commentary
3. Preserve all details from the user's current query
4. Be sensitive to topic changes - when a topic changes, do not carry over unrelated context
5. For ambiguous references, prioritize the most relevant (not necessarily the most recent) conversation
6. Never answer the question - only reformulate it
7. If the query is already self-sufficient, make minimal changes or return it as is
8. Use natural language as if the user had included all relevant context themselves
9. If the user's intent is a question, always ensure the reformulated text is also a question

EXAMPLES:

Example 1 - Basic context addition:
User: "Do we have laundry in Hotel?"
Reformulated query: "Does the Hotel provide laundry service during the stay?"

Example 2 - Topic change recognition:
User: "Do we have laundry in Hotel?"
Assistant: [Reply on laundry ...]
User: "Does hotel have fitness centre?"
Reformulated query: "During the stay, does the hotel provide a gym or fitness centre?"
DO NOT: "Does the hotel with laundry service have a fitness centre?" [INCORRECT - This forcefully combines the previous laundry topic with the new fitness center topic when the user has clearly switched topics]

Example 3 - Pronoun resolution:
User: "Does The Grand Palace Hotel offer laundry services?"
Assistant: [Reply on laundry ...]
User : "Is it for free?"
Reformulated query: "Is the laundry service provided by The Grand Palace Hotel complimentary?"

Example 4 - Topic switch with pronoun:
User: "Does The Grand Palace Hotel offer laundry services?"
Assistant: [Reply on laundry ...]
"Is it for free?"
Reformulated query: "Is the laundry service provided by The Grand Palace Hotel complimentary?"
Assistant: [Reply on laundry pricing...]
User: "Does the hotel have a fitness center?"
Reformulated query: "Is the fitness center/ gym service provided by The Grand Palace Hotel complimentary?"
Assistant: [Reply on fitness center ...]
User: "Is it for free?"
Reformulated query: "Is the fitness center at The Grand Palace Hotel free to use?"
DO NOT: "Is the laundry service at The Grand Palace Hotel complimentary?" [INCORRECT - This refers to the older laundry topic despite the conversation having moved on to discuss the fitness center, which is the current relevant topic]

Example 5 - Relevant (not just recent) context:
User: "Any football matches today?"
Assistant: [Reply on today's football matches ...]
User: "BTW Will it rain today?"
Assistant: [Reply on possibility of rain today ...]
User: "Is it streaming now?"
Reformulated query: "Is the football match streaming now?"
DO NOT: "Is the rain streaming now?" [INCORRECT - This assumes "it" refers to rain simply because rain was mentioned most recently, but "streaming" is semantically related to football matches, not rain, so football is the more relevant context]

Example 6 - Preserve question format:
User: "I'm planning to visit Paris."
Assistant: [Reply on Paris visit plans ...]
User: "When should I go?"
Reformulated query: "When is the best time to visit Paris?"
DO NOT: "The best time to visit Paris." [INCORRECT - This transforms the question into a statement, failing to preserve the interrogative intent of the original query]

DO NOT add commentary or explanations:
User: "What time does the museum close?"
DO NOT: "Based on your interest in visiting the museum, you'd like to know what time it closes today." [INCORRECT - This adds unnecessary commentary and explanation rather than simply reformulating the query]
DO INSTEAD: "What time does the museum close today?"

DO NOT exceed one sentence unless absolutely necessary:
User: "Can I cancel my reservation?"
DO NOT: "Can I cancel my hotel reservation? I'd like to know if there are any cancellation fees involved." [INCORRECT - This uses two sentences when one would suffice, making the reformulation unnecessarily verbose]
DO INSTEAD: "Can I cancel my hotel reservation and what are the associated fees?"

IMPORTANT: When a new topic is introduced (like fitness center after discussing laundry), subsequent references like "it" should refer to the most recently discussed topic, not older topics. Always pay close attention to topic shifts and pronoun references.

Remember: Focus on making the query self-contained while preserving its original intent. Never exceed a single sentence unless absolutely necessary. Always maintain the question format when the user's intent is a question."""
        user_prompt = f"""
Conversation history:
{history_context}

Current query: "{current_query}"

Reformulated query:
        """
        try:
            reformulated_query = self.generate_answer(
                system_prompt, user_prompt, is_stream=False
            ).strip()
            st.write(f"Reformulated query: '{reformulated_query}'")

            # Filter out any explanatory text the LLM might add
            if ":" in reformulated_query:
                parts = reformulated_query.split(":", 1)
                if len(parts) > 1:
                    reformulated_query = parts[1].strip()

            # Clean up quotes if present
            reformulated_query = reformulated_query.strip("\"'")

            # If the reformulation seems excessive or contains phrases like "based on the conversation",
            # or if it looks like an answer rather than a question, fallback to original query
            problematic_phrases = [
                "based on",
                "according to",
                "from our conversation",
                "as mentioned",
                "earlier you asked",
                "you asked about",
            ]

            if (
                len(reformulated_query.split()) > 25
                or any(
                    phrase in reformulated_query.lower()
                    for phrase in problematic_phrases
                )
                or ("." in reformulated_query and "?" not in reformulated_query)
            ):
                st.write(
                    f"Reformulation looks problematic, falling back to original query"
                )
                return current_query

            # If reformulation is too similar to original, keep original
            if reformulated_query.lower() == current_query.lower():
                st.write(f"Reformulation is identical to original query")
                return current_query

            return reformulated_query
        except Exception as e:
            st.error(f"Error in query contextualization: {str(e)}")
            return current_query

    def handle_non_hotel_query(
        self, query_type: str, query: str
    ) -> Union[Iterator[Dict], str]:
        """Handle compliments, complaints, chitchat, and off-topic queries using LLM."""
        st.write(f"Handling {query_type} query: '{query}'")

        # For compliments, we can provide a friendly response
        if query_type == "compliment":
            system_prompt = """
            You are an AI assistant for The Grand Palace Hotel. A guest has given you a compliment or thanked you.
            
            Respond with:
            1. A brief, gracious acknowledgment of their thanks
            2. An offer to help with any hotel-related questions they might have
            
            Keep your response to 1-2 sentences maximum.
            """

            user_prompt = f'Guest compliment: "{query}"'

        # For complaints, acknowledge and redirect
        elif query_type == "complaint":
            system_prompt = """
            You are an AI assistant for The Grand Palace Hotel. A guest has expressed dissatisfaction or complained.
            
            Respond with:
            1. A brief apology for any inconvenience
            2. An offer to help with specific hotel-related questions
            
            Keep your response to 1-2 sentences maximum.
            """

            user_prompt = f'Guest complaint: "{query}"'

        # For chitchat, be polite but redirect to hotel topics
        elif query_type == "chitchat":
            system_prompt = """
            You are an AI assistant for The Grand Palace Hotel. A guest has engaged in general conversation not related to hotel services.
            
            Respond with:
            1. A brief, friendly response
            2. A polite redirection to hotel-related topics
            
            Keep your response to 1-2 sentences maximum.
            """

            user_prompt = f'Guest chitchat: "{query}"'

        # For off-topic, clearly state the purpose and redirect
        else:  # off-topic
            system_prompt = """
            You are an AI assistant for The Grand Palace Hotel. A guest has asked about a topic unrelated to The Grand Palace Hotel or hotel services.
            
            Respond with:
            1. A polite explanation that you're designed to assist with questions about The Grand Palace Hotel
            2. An offer to help with hotel-related questions
            
            Keep your response to 1-2 sentences maximum.
            """

            user_prompt = f'Off-topic question: "{query}"'

        return self.generate_answer(system_prompt, user_prompt)

    def rag(self, query: str) -> Union[Iterator[Dict], str]:
        """Main function to perform conversational document search and answer generation."""
        try:
            # Create a debug expander for this query
            with st.expander(f" Debug info for query: '{query}'", expanded=False):
                st.write(f"Processing query: '{query}'")

                # Detect the type of query
                query_type = self.detect_query_type(query)
                st.write(f"Query type determined: '{query_type}'")

                # Handle non-hotel queries
                if query_type != "hotel-related":
                    st.write(f"Handling non-hotel query of type: {query_type}")
                    response_stream = self.handle_non_hotel_query(query_type, query)

                    # For non-streaming case, store the full response
                    if not isinstance(response_stream, Iterator):
                        self.conversation_history.append((query, response_stream))
                        return iter([{"message": {"content": response_stream}}])

                    # For streaming, we'll update conversation history after collecting full response
                    return response_stream

                # Contextualize the query based on conversation history
                contextualized_query = self.contextualize_query(query)
                st.write(f"Contextualized query: '{contextualized_query}'")

                # Retrieve relevant context
                context, relevance_score = self.search(contextualized_query)
                st.write(f"Top relevance score: {relevance_score:.4f}")

                # If no relevant context found, provide a fallback response
                if relevance_score < 0.35 and context != "No documents available.":
                    st.write(
                        f"Low relevance score ({relevance_score:.4f}), using fallback response"
                    )
                    system_prompt = """
                    You are an AI assistant for The Grand Palace Hotel. You're designed to answer questions about the hotel's services and amenities.
                    
                    The guest has asked a question that doesn't seem to match our available information. Respond with:
                    
                    1. A polite explanation that you don't have enough information to answer their specific question
                    2. An offer to help with other hotel-related questions
                    3. A suggestion to contact the hotel directly for more specific information
                    
                    Keep your response brief and professional.
                    """

                    user_prompt = f'Guest question without matching context: "{query}"'

                    return self.generate_answer(system_prompt, user_prompt)

                # Generate the answer using the retrieved context
                st.write("Generating final response with retrieved context")
                system_prompt = """You are an AI assistant for a hotel booking website. You have access to a collection of detailed information about the hotel's services, booking, cancellation, and policies. Your task is to help users by providing them with relevant and accurate answers to their queries.

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
                """

                user_prompt = f'Based on the following context, please answer the question.\nIf the answer cannot be derived from the context, say "I cannot answer this based on the provided context." \n\nContext:\n{context}\n\nQuestion: {contextualized_query}\n\nAnswer:\n'

                return self.generate_answer(system_prompt, user_prompt)

        except Exception as e:
            error_msg = f"I apologize, but I encountered an error while processing your question. Please try asking again or contact the Front Desk for immediate assistance."
            st.error(f"Error in RAG process: {str(e)}")
            return iter([{"message": {"content": error_msg}}])


def chunk_text_with_overlap(
    text: str, chunk_size: int = 750, overlap_size: int = 150
) -> List[str]:
    """Split text into chunks with overlap."""
    if not text:
        return ["No text provided"]

    words = text.split()
    chunks = []

    # Calculate number of words per chunk
    words_per_chunk = chunk_size
    overlap_words = overlap_size

    if not words:
        return ["No text provided"]

    if len(words) <= words_per_chunk:
        return [text]

    # Debug info
    st.write(
        f"Chunking text with {len(words)} words into chunks of ~{words_per_chunk} words with {overlap_words} word overlap"
    )

    i = 0
    while i < len(words):
        # Get the chunk
        chunk_end = min(i + words_per_chunk, len(words))
        chunk = " ".join(words[i:chunk_end])
        chunks.append(chunk)

        # Move to next position with overlap
        i += words_per_chunk - overlap_words

        # Avoid creating very small chunks at the end
        if i + words_per_chunk > len(words) and i < len(words):
            chunks.append(" ".join(words[i:]))
            break

    # Try to preserve paragraph and section boundaries
    # This improves the coherence of the chunks
    # TODO: Need some more investigations
    improved_chunks = []
    for chunk in chunks:
        # Try to break at paragraph or section markers
        if len(chunk.split()) > (words_per_chunk / 2):  # Only for chunks big enough
            for marker in ["\n\n", "\n", ". ", "! ", "? "]:
                if marker in chunk:
                    # Find the last occurrence of the marker
                    last_marker_pos = chunk.rfind(marker)
                    # if last_marker_pos > int(len(chunk) * 0.6):  # Only if marker is past 60% of chunk
                    if last_marker_pos > int(
                        len(chunk) - overlap_size
                    ):  # Only if marker is past overlap_size
                        improved_chunks.append(chunk[: last_marker_pos + len(marker)])
                        break
            else:  # No suitable marker found
                improved_chunks.append(chunk)
        else:
            improved_chunks.append(chunk)

    # Debug info
    st.write(
        f"Created {len(chunks)} initial chunks, {len(improved_chunks)} improved chunks"
    )

    return improved_chunks if improved_chunks else chunks


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "document_uploaded" not in st.session_state:
        st.session_state.document_uploaded = False
    if "document_text" not in st.session_state:
        st.session_state.document_text = ""
    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "document_embeddings" not in st.session_state:
        st.session_state.document_embeddings = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "rag_system" not in st.session_state:
        st.session_state.rag_system = None
    if "debug_mode" not in st.session_state:
        st.session_state.debug_mode = True


def display_chat_messages():
    """Display all messages in the chat history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def handle_user_input():
    """Process user input from the chat interface."""
    prompt = st.chat_input("Please enter your query regarding The Grand Palace Hotel:")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        return prompt
    return None


def handle_document_upload() -> str:
    """Handle document upload and processing."""
    uploaded_file = st.file_uploader(
        "Upload hotel information document",
        type=["txt"],
        disabled=st.session_state.document_uploaded,
    )
    if uploaded_file is not None:
        return process_uploaded_document(uploaded_file)
    return None


def process_uploaded_document(uploaded_file) -> str:
    """Process the uploaded document and extract its text."""
    file_extension = uploaded_file.name.split(".")[-1].lower()

    if file_extension == "txt":
        document_text = uploaded_file.getvalue().decode("utf-8")
        st.write(f"Processed document with {len(document_text)} characters")
        return document_text
    else:
        return "Unsupported file type."


def stream_parser(stream):
    """Parse the streaming response from Ollama."""
    full_response = ""
    stream_start = time.time()
    chunk_count = 0

    for chunk in stream:
        chunk_count += 1
        if "message" in chunk and "content" in chunk["message"]:
            content = chunk["message"]["content"]
            full_response += content
            yield content

    # Debug info for stream completion
    stream_duration = time.time() - stream_start
    if st.session_state.debug_mode:
        st.write(
            f"Response streaming completed in {stream_duration:.2f} seconds ({chunk_count} chunks)"
        )

    return full_response


def rag(query: str):
    """Perform RAG on the query using the initialized ConvoRAG system."""
    if st.session_state.rag_system:
        return st.session_state.rag_system.rag(query)
    else:
        return iter(
            [
                {
                    "message": {
                        "content": "Document not processed yet. Please upload a document first."
                    }
                }
            ]
        )


def main():
    """Main function to run the Streamlit app."""
    st.title("The Grand Palace Hotel Information Assistant")

    initialize_session_state()

    with st.sidebar:
        st.title("Configuration")
        model = st.selectbox("Select LLM Model", ["llama3.2", "phi4-mini"], index=0)
        embed_model = st.selectbox(
            "Select Embedding Model", ["nomic-embed-text"], index=0
        )

        # Add debug mode toggle
        st.session_state.debug_mode = st.checkbox("Enable Debug Mode", value=True)
        if st.session_state.debug_mode:
            st.info(
                "Debug mode is enabled. You'll see detailed information about the system's operations."
            )

        # Display basic hotel information in the sidebar
        if st.session_state.document_uploaded:
            st.markdown("---")
            st.subheader("About The Grand Palace Hotel")
            st.markdown("""
            * Luxurious accommodation in the heart of the city
            * Check-in: 3:00 PM, Check-out: 11:00 AM
            * Complimentary breakfast from 7:00 AM to 10:00 AM
            * Amenities: Rooftop pool, spa, fitness center
            * Pet-friendly in selected rooms
            * Free parking & airport shuttle (fees apply)
            
            *Ask the assistant for more specific information.*
            """)

    # Add a welcome message when the app starts
    if not st.session_state.messages and not st.session_state.document_uploaded:
        st.info(
            "👋 Welcome to The Grand Palace Hotel Information Assistant. Please upload the hotel information document to begin."
        )

    if not st.session_state.document_uploaded:
        document_text = handle_document_upload()
        if document_text:
            with st.spinner("Processing hotel information..."):
                start_time = time.time()
                # Add debug info
                st.write(f"Starting document processing")

                # Process the document with improved chunking
                documents = chunk_text_with_overlap(
                    text=document_text, chunk_size=250, overlap_size=75
                )
                st.session_state.documents = documents
                st.session_state.document_text = document_text

                # Initialize the RAG system with the processed documents
                st.write(
                    f"Initializing RAG system with {len(documents)} document chunks"
                )
                st.session_state.rag_system = ConvoRAG(
                    documents, embedding_model=embed_model, llm_model=model
                )
                st.session_state.document_uploaded = True

                # Add welcome message to conversation
                welcome_msg = "👋 Welcome to The Grand Palace Hotel Information Assistant! I'm here to help you with any questions about our amenities, services, policies, or booking information. How may I assist you today?"
                st.session_state.messages.append(
                    {"role": "assistant", "content": welcome_msg}
                )
                #time.sleep(20)
                st.success("Hotel information processed successfully!")
                st.rerun()  # Trigger a rerun to switch to chat interface
    else:
        # Show a reset button
        if st.sidebar.button("Reset Conversation"):
            st.session_state.messages = []
            st.session_state.conversation_history = []
            # Re-add welcome message
            welcome_msg = "👋 Welcome to The Grand Palace Hotel Information Assistant! I'm here to help you with any questions about our amenities, services, policies, or booking information. How may I assist you today?"
            st.session_state.messages.append(
                {"role": "assistant", "content": welcome_msg}
            )
            st.rerun()

        if st.sidebar.button("Restart Conversation"):
            st.session_state.clear()
            st.rerun()

        # Display the chat interface
        display_chat_messages()
        user_input = handle_user_input()

        if user_input:
            # Add the user query to conversation history
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""

                try:
                    # Show typing indicator
                    message_placeholder.markdown("⏳")

                    # Get the streaming response
                    llm_stream = rag(user_input)

                    # Process and display the streaming response
                    # Use a custom stream parser to handle the specific format
                    collected_response = ""
                    for part in stream_parser(llm_stream):
                        full_response += part
                        collected_response += part
                        message_placeholder.markdown(full_response + "▌")
                        time.sleep(0.01)

                    # Display final response
                    message_placeholder.markdown(full_response)

                    # Update conversation history with the contextualized response
                    st.session_state.rag_system.conversation_history.append(
                        (user_input, collected_response)
                    )
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": collected_response,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )

                except Exception as e:
                    error_msg = f"I apologize, but I encountered an error. Please try asking again or contact our Front Desk for assistance."
                    message_placeholder.markdown(error_msg)
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_msg,
                            "timestamp": datetime.now().isoformat(),
                        }
                    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="The Grand Palace Hotel Assistant", page_icon="🏨", layout="wide"
    )
    main()
