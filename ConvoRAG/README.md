# ConvoRAG


**ConvoRAG** is an educational and experimental Python tool developed to demonstrate **Conversational Retrieval-Augmented Generation (RAG)** from scratch. It is designed to answer user queries based on a predefined set of documents, while also maintaining and utilizing conversation context for more dynamic and personalized interactions. 

This tool builds upon the traditional RAG architecture by incorporating advanced features such as conversation history tracking, query classification, and query contextualization. By doing so, **ConvoRAG** ensures that responses are not only relevant to the current query but also take into account the ongoing conversation, improving the accuracy and coherence of answers.

The goal of **ConvoRAG** is to provide an easy-to-understand and fully transparent implementation of RAG that can be used as an educational resource. The tool is powered by open-source components and free local LLMs (Large Language Models), ensuring that anyone can experiment with and learn from it without worrying about costs. 

Whether you're new to RAG or looking to deepen your understanding, **ConvoRAG** offers a hands-on approach to learning how conversational AI systems can combine document retrieval and generation to produce more context-aware responses.

Key Features:
- **Conversational Context**: Remembers previous interactions and uses that context to improve answers.
- **Query Classification**: Automatically classifies queries to optimize the retrieval process.
- **Query Contextualization**: Enhances the relevance of responses by considering conversation history.
- **Educational & Open Source**: Designed to help users learn the inner workings of RAG systems using free, accessible resources.


## Features

- **Conversational Retrieval-Augmented Generation (RAG)**: Retrieves relevant documents based on the input query and maintains conversation context to generate well-informed responses.
- **Query Classification**: Automatically categorizes user queries into different types (hotel-related, compliment, complaint, chitchat, or off-topic) and handles each appropriately.
- **Query Contextualization**: Enhances brief follow-up queries with context from previous conversation exchanges to make them more specific and useful for retrieval.
- **Semantic Search**: Searches the provided documents by embedding the query and the documents into vectors and calculating cosine similarity to find the most relevant documents.
- **Conversational Memory**: Maintains a history of the conversation to provide context for future queries.
- **Generative AI Response**: Generates responses using an AI model (Llama3.2) based on the retrieved context and conversation history.
- **Context-Based Answers**: Provides answers derived directly from the document content. If no answer can be generated from the context, guides the user to contact the helpdesk.
- **User-Friendly Console Interface**: A simple command-line interface that allows users to input their queries and get contextually relevant answers.
- **Iterative Querying**: Users can continue the conversation naturally with follow-up questions.

## Requirements

To run **ConvoRAG**, you'll need the following Python packages:

- `numpy` (version 2.2.2 or higher)
- `ollama` (version 0.4.7 or higher)
- Python 3.12 or higher

You can install the required dependencies using `pip`:

```bash
pip install numpy ollama
```

## How to Run

1. Clone or download the code.
2. Install the required dependencies.
3. Run the script:

```bash
python ConvoRAG.py
```

4. After running the script, enter your query when prompted, and the system will retrieve the most relevant hotel-related documents based on the query.

### Example Results with Positive and Negative Scenarios

```bash

==================================================
Conversational Retrieval-Augmented Generation
==================================================

--------------------------------------------------
Please enter your query regarding the hotel: Is there a laundry service available at the hotel?

Assistant: Yes, the hotel provides laundry services, including dry cleaning and pressing. You can use their laundry facilities to get your clothes cleaned and pressed during your stay.
--------------------------------------------------

Do you want to ask another question? (y/Y to continue, any other key to quit): y

--------------------------------------------------
Please enter your query regarding the hotel: Is it complimentary with hotel room booking ?

Assistant: No, based on the provided context, it does not appear that the laundry service is complimentary with a standard hotel room booking. The context only mentions that the hotel provides laundry services, including dry cleaning and pressing, but it does not specify whether this service is included in the cost of a standard room or if there is an additional fee for it.

For more information on our laundry policies, please contact the helpdesk at the number provided on our website.
--------------------------------------------------

Do you want to ask another question? (y/Y to continue, any other key to quit): y

--------------------------------------------------
Please enter your query regarding the hotel: You are a very good assistant.

Assistant: Thank you so much for your kind words! We're thrilled to hear that our team has been able to provide you with excellent service during your stay. I'll be sure to pass along your compliment to the entire team, and we appreciate your loyalty to our hotel.
--------------------------------------------------

Do you want to ask another question? (y/Y to continue, any other key to quit): y

--------------------------------------------------
Please enter your query regarding the hotel: Sometimes you are frustratingly slow

Assistant: I apologize for any inconvenience caused by our speed, and I appreciate you bringing it to my attention. We're taking steps to optimize our services to ensure a more efficient experience for all guests. If you'd like, I can also provide you with some recommendations for nearby restaurants or attractions that may help pass the time while we work on improving our response times.
--------------------------------------------------

Do you want to ask another question? (y/Y to continue, any other key to quit): y
```
## How it Works

ConvoRAG enhances traditional RAG systems by adding conversational awareness. Here's how the system works:

1. **Query Classification**: When a user submits a query, the system first uses the LLM to classify it into one of five categories:
   - Hotel-related: Questions about amenities, services, policies, bookings, etc.
   - Compliment: Positive feedback or appreciation
   - Complaint: Negative feedback or dissatisfaction
   - Chitchat: General conversation not directly related to hotel inquiries
   - Off-topic: Questions completely unrelated to hotels

2. **Query Handling Based on Type**:
   - For non-hotel-related queries (compliments, complaints, chitchat, off-topic), the system generates appropriate responses without using the document retrieval mechanism.
   - For hotel-related queries, the system continues with the RAG process.

3. **Query Contextualization**: For hotel-related queries, the system analyzes the conversation history to enhance brief follow-up questions.
   - For example, if a user first asks "Does the hotel have a pool?" and then follows up with "Is it heated?", the system will reformulate the second query to "Is the hotel pool heated?"
   - This contextualization makes retrieval more effective for brief follow-up questions.

4. **Text Embedding**: The contextualized query and documents are converted into embeddings using the Ollama embedding model (default: nomic-embed-text).

5. **Cosine Similarity Calculation**: The system calculates the cosine similarity between the query embedding and each document's embedding.

6. **Top-K Retrieval**: The system identifies the K most similar documents based on cosine similarity scores.

7. **Answer Generation**: The retrieved documents are used as context to generate a response using a language model (default: llama3.2).

8. **Conversation History Update**: The query and response are added to the conversation history to provide context for future queries.

9. **Iterative Process**: The process continues as the user asks additional questions, with each new query benefiting from the accumulated conversation context.

## Disclaimer

**Important Note on Educational Purpose and Query Processing Limitations**:

This code is provided for educational purposes only and is not intended for production use. Using it as-is in production may introduce unforeseen risks or issues. ConvoRAG serves as a demonstration of conversational RAG techniques and should be used primarily for learning and experimentation.

While ConvoRAG strives to accurately process all user queries, the query classification and contextualization components may occasionally fail to properly categorize or contextualize certain inputs. This can happen when:

- Queries contain ambiguous wording or multiple intents
- Follow-up questions lack clear references to previous exchanges
- Unusual phrasing or domain-specific terminology is used
- Very brief queries without sufficient context are submitted

If you notice these limitations affecting system performance, consider adjusting the prompts in the `detect_query_type()` and `contextualize_query()` methods to better handle your specific use cases. The current prompts are designed for general hotel-related queries but may need fine-tuning for optimal performance in other domains or for edge cases within the hotel domain.

### Key Components

- **ConversationalRAG Class**: The main class that encapsulates all functionality.
- **embed_text**: Converts text to vector embeddings using the Ollama API.
- **cosine_similarity**: Calculates similarity between embeddings.
- **search**: Finds the most relevant documents for a query.
- **detect_query_type**: Classifies queries into different categories.
- **contextualize_query**: Enhances queries with conversation context.
- **handle_non_hotel_query**: Generates responses for non-hotel-related queries.
- **rag**: The main function that orchestrates the entire process.

## Customization

You can customize this tool by:

- **Replacing the hardcoded**  Modify the documents list in the script to include your own set of documents. For example:
```python
documents = [
    "Your custom document 1",
    "Your custom document 2",
    "Your custom document 3"
]
```
- **Dynamic Document Loading**: Modify the script to load documents from a file or a database if needed.
- **Adjusting the Number of Results**: The top_k parameter in the search function controls how many relevant documents to retrieve. You can modify it to return more or fewer results:
```python
relevant_documents = search(user_input, documents, document_embeddings, top_k=3)
```
- **Embedding Model Customization**: You can choose which embedding model to use dynamically by passing a custom embedding model name. For example: Lets say "mxbai-embed-large"
```python
# Use other embedding model
embedding = embed_text("Your text here", embedding_model="mxbai-embed-large")
```
- **Chat Model Customization**: The generate_answer function allows you to select a different model for generating responses. You can specify the model as needed. For example: Lets say "mistral"
```python
# Use a different chat model
answer = generate_answer(system_prompt, user_prompt, model='mistral')
```
- **Customizing the System Prompt**: The system prompt can be adjusted dynamically. Here's an example where you customize the assistant's role and the context it works with:
```python
system_prompt = '''You are an AI assistant for a tech product support website. 
You assist customers with issues like product setup, troubleshooting, warranty, and returns. 
Respond to questions based on the context provided below, and if you cannot answer, direct them to customer support.'''
```

## Cosine Similarity

Cosine similarity is a metric used to measure how similar two vectors are. It ranges from -1 (completely opposite vectors) to 1 (identical vectors). The cosine similarity is calculated as follows:

```
Cosine Similarity = (A ⋅ B) / (|A| * |B|)
```

Where:
- `A` and `B` are the two vectors.
- `A ⋅ B` is the dot product of the two vectors.
- `|A|` and `|B|` are the magnitudes (or norms) of the vectors.

## License

This project is licensed under the `MIT License with Attribution` - see the [LICENSE](../LICENSE) file for details.

## Author

- **Gurucharan MK** (2025)

---
