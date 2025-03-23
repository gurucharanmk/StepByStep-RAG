# RAGStream Hotel Information Assistant

**RAGStream** is an educational Streamlit application that demonstrates **Conversational Retrieval-Augmented Generation (RAG)** for a hotel information system. This application helps hotel guests find information about The Grand Palace Hotel's amenities, services, policies, and booking information through a conversational interface.

This tool builds upon traditional RAG architecture by incorporating advanced features such as conversation history tracking, query classification, and query contextualization. It ensures responses are not only relevant to the current query but also take into account the ongoing conversation, improving the accuracy and coherence of answers.

## Features

- **Web-Based Interface**: User-friendly Streamlit application with chat interface and configuration options
- **Conversational RAG**: Retrieves relevant document chunks based on user queries while maintaining conversation context
- **Query Classification**: Automatically categorizes queries into different types (hotel-related, compliment, complaint, chitchat, or off-topic)
- **Query Contextualization**: Enhances brief follow-up queries with conversation history context
- **Document Chunking**: Processes uploaded documents into manageable chunks with overlap for better context retention
- **Semantic Search**: Uses vector embeddings and cosine similarity to find the most relevant document chunks
- **Streaming Responses**: Displays AI responses in real-time as they're generated
- **Debug Mode**: Optional detailed logging of the system's internal operations for educational purposes
- **Model Selection**: Choose between different LLMs (Llama3.2, Phi4-mini) for response generation

## How It Works

The application enhances traditional RAG systems by adding conversational awareness:

1. **Document Processing**:
   - Upload a text document containing hotel information
   - The system chunks the document into smaller pieces with overlap
   - Each chunk is embedded using the specified embedding model

2. **Query Classification**:
   - When a user submits a query, the system classifies it into one of five categories:
     - Hotel-related: Questions about amenities, services, policies, etc.
     - Compliment: Positive feedback or appreciation
     - Complaint: Negative feedback or dissatisfaction
     - Chitchat: General conversation not directly related to hotel inquiries
     - Off-topic: Questions completely unrelated to hotels

3. **Query Handling Based on Type**:
   - For non-hotel queries, the system generates appropriate responses without using document retrieval
   - For hotel-related queries, the system proceeds with the RAG process

4. **Query Contextualization**:
   - Analyzes conversation history to enhance brief follow-up questions
   - For example, if a user asks "Does the hotel have a pool?" and then "Is it heated?", the system reformulates the second query to "Is the hotel pool heated?"

5. **Semantic Search**:
   - The contextualized query is embedded using the same model as the documents
   - Cosine similarity is calculated between the query embedding and each document chunk
   - The most relevant document chunks are retrieved

6. **Answer Generation**:
   - The retrieved document chunks are used as context to generate a response using the selected LLM
   - Responses are streamed to the UI as they're generated

7. **Conversation History Update**:
   - The query and response are added to the conversation history
   - This history is used to provide context for future queries

## Requirements

To run RAGStream Hotel Information Assistant, you'll need:

- Python 3.6 or higher
- Streamlit
- Numpy
- Ollama
- A text document containing hotel information

### Python Dependencies

```bash
pip install streamlit numpy ollama
```

### Ollama Configuration

The application requires [Ollama](https://ollama.ai/) to be installed and the following models to be available:

- `llama3.2` or `phi4-mini` for text generation
- `nomic-embed-text` for embeddings

## How to Run

1. Clone the repository or download the source code
2. Install the required dependencies
3. Run the Streamlit application:

```bash
streamlit run app.py
```

4. Open your web browser and navigate to the URL displayed in the terminal
5. Upload a text document containing hotel information
6. Start asking questions about the hotel through the chat interface

## Customization

You can customize the application through the Streamlit interface:

- Select different LLM models for response generation
- Toggle debug mode to view detailed system operations
- Adjust document chunking parameters in the code (chunk size and overlap)
- Modify system prompts for different response styles

## Debug Mode

Enable debug mode in the sidebar to see detailed information about:

- Document chunking process
- Embedding dimensions
- Similarity scores during search
- Query classification and contextualization
- Response generation timing

This information is valuable for understanding how the RAG system works and for troubleshooting issues.

## Educational Purpose

This application is provided for educational purposes to demonstrate:

- How to implement a conversational RAG system
- The impact of conversation history on query understanding
- Query classification techniques
- Document chunking strategies
- Vector-based semantic search
- Integration with local LLMs via Ollama

## Limitations

- The system relies on the quality of document content for accurate answers
- Very brief follow-up questions might not be contextualized correctly
- Performance depends on the capabilities of the selected LLM
- The current implementation only supports text documents

## Disclaimer

This code is provided for educational purposes only and is not intended for production use. It serves as a demonstration of conversational RAG techniques and should be used primarily for learning and experimentation.

## License

This project is licensed under the MIT License with Attribution.

## Author

Based on the original RAGStream concept by Gurucharan MK (2025)
