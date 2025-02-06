# SimpleRAG

**SimpleRAG** is a Python tool designed to perform **Retrieval-Augmented Generation (RAG)** for answering queries based on a predefined set of documents. It combines the power of semantic search with generative AI to provide relevant and accurate answers to user queries. The tool is ideal for applications like customer service, hotel booking inquiries, or other contexts where detailed, document-based information is needed.

## Features

- **Retrieval-Augmented Generation (RAG)**: The tool retrieves relevant documents based on the input query and uses this context to generate a well-informed response.
- **Semantic Search**: Searches the provided documents by embedding the query and the documents into vectors and calculating cosine similarity to find the most relevant documents.
- **Generative AI Response**: Generates a response using an AI model (like LLaMA) based on the retrieved context.
- **Context-Based Answers**: The system provides answers derived directly from the document content. If no answer can be generated from the context, the system guides the user to contact the helpdesk.
- **User-Friendly Console Interface**: A simple command-line interface that allows users to input their query and get an answer.
- **Iterative Querying**: Users can continue querying the system as needed.

## Requirements

To run **SimpleRAG**, you'll need the following Python packages:

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
python SimpleRAG.py
```

4. After running the script, enter your query when prompted, and the system will retrieve the most relevant hotel-related documents based on the query.

### Example Results with Positive and Negative Scenarios

```bash

==================================================
Basic Retrieval-Augmented Generation
==================================================

--------------------------------------------------
Please enter your query regarding the hotel: How soon do I need to cancel to avoid penalties?
Retreived context:
 Guests can cancel their reservation free of charge up to 48 hours before the scheduled check-in date.
Cancellations made within 24 hours of check-in will incur a 50% charge of the total booking cost.
Refunds are only available for cancellations made within the allowed time frame, as per the hotel's cancellation policy.
If a guest has prepaid for their stay and cancels within the eligible period, a full refund will be issued.
To cancel your reservation, you can either call the hotel directly or cancel through our website.

Answer: According to the provided context, to avoid penalties, you should cancel your reservation at least 48 hours before the scheduled check-in date. Cancellations made within this time frame will not incur any charges.
--------------------------------------------------

Do you want to search again? (y/Y to continue, any other key to quit): y

==================================================
Basic Retrieval-Augmented Generation
==================================================

--------------------------------------------------
Please enter your query regarding the hotel: Does the hotel provide a mini-fridge in the rooms?
Retreived context:
 The hotel provides laundry services, including dry cleaning and pressing.
The hotel accepts bookings for a minimum stay of two nights during peak seasons.
Free parking is available for all guests staying at the hotel.
Room service is available 24/7 for all guests.
The hotel bar serves a variety of cocktails, wine, and snacks from 12:00 PM to midnight.

Answer: Unfortunately, I don't have any information about room features or specific amenities provided in the guest rooms.

I cannot answer this based on the provided context. For more information, please contact the helpdesk at the number provided on our website.
--------------------------------------------------

Do you want to search again? (y/Y to continue, any other key to quit): y
```
## How it Works

1. **Text Embedding**: The text (query and documents) is converted into embeddings using the Ollama nomic-embed-text model.
2. **Cosine Similarity**: The cosine similarity between the query embedding and each document's embedding is calculated.
3. **Top-K Retrieval**: The topk function identifies the K most similar documents, based on cosine similarity scores, and displays them to the user.
 > **Note**: The hotel-related documents used in the example are hardcoded in the script under the documents variable. You can easily modify this list or load your own documents to suit your needs. and other related to customization of documents etc.
4. **Answer Generation**: The retrieved documents are used as context to generate a response using a language model llama3.2.

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
