# SemanticSeek

**SemanticSeek** is a Python tool designed for exploring semantic search capabilities using natural language queries. It utilizes embeddings to find the most relevant documents based on cosine similarity. While the tool can be used for various domains, in this implementation, we focus on hotel-related documents to demonstrate how semantic search can be applied in real-world scenarios like customer service, booking information, and amenities descriptions.

## Features

- **Semantic Search**: Perform searches using natural language queries and retrieve the most relevant documents from a collection.
- **Cosine Similarity Calculation**: Measures the relevance of each document by calculating cosine similarity between the query and document embeddings.
- **Document Embedding**: Uses Ollama's embedding model(nomic-embed-text) to convert text into vectors for similarity comparisons.
- **Top-K Retrieval**: Retrieves the top K most relevant documents based on similarity scores.
- **User-Friendly Interface**: A simple console-based interface that allows users to input their query and view relevant documents.
- **Iterative Querying**: Users can continue querying the model for as long as they need.

## Requirements

To run **SemanticSeek**, you'll need the following Python packages:

- `numpy` (version 2.2.2 or higher)
- `ollama` (version 0.4.7 or highern)
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
python SemanticSeek.py
```

4. After running the script, enter your query when prompted, and the system will retrieve the most relevant hotel-related documents based on the query.

### Example

```bash

==================================================
Semantic Search
==================================================
Enter your query:Can I get my clothes cleaned at the hotel?

Relevant Documents for your query: 'Can I get my clothes cleaned at the hotel?'

--------------------------------------------------
Document: The hotel provides laundry services, including dry cleaning and pressing.
Similarity Score: 0.7981

Document: To cancel your reservation, you can either call the hotel directly or cancel through our website.
Similarity Score: 0.6013

Document: Free parking is available for all guests staying at the hotel.
Similarity Score: 0.5863

Document: Refunds are only available for cancellations made within the allowed time frame, as per the hotel's cancellation policy.
Similarity Score: 0.5731

Document: The hotel features a rooftop pool, fitness center, and spa services for all guests.
Similarity Score: 0.5657
--------------------------------------------------

Do you want to search again? (y/Y to continue, any other key to quit): y

```
## How it Works

1. **Text Embedding**: The text (query and documents) is converted into embeddings using the Ollama nomic-embed-text model.
2. **Cosine Similarity**: The cosine similarity between the query embedding and each document's embedding is calculated.
3. **Top-K Retrieval**: The topk function identifies the K most similar documents, based on cosine similarity scores, and displays them to the user.
 > **Note**: The hotel-related documents used in the example are hardcoded in the script under the documents variable. You can easily modify this list or load your own documents to suit your needs. and other related to customization of documents etc.

## Customization

You can customize this tool by:

- Replacing the hardcoded `documents` list with your own set of documents. Simply modify the `documents` variable in the script to include your custom texts.
- Loading documents dynamically by modifying the script to read text from a file or a database if needed.
- Changing the number of top results returned by adjusting the top_k parameter in the search function. This allows you to retrieve more or fewer results based on your preference.

For example, you can replace the hotel-related documents with your own documents like this:
```python
documents = [
    "Your custom document 1",
    "Your custom document 2",
    "Your custom document 3"
]
```
You can also change how many results are shown by adjusting the `top_k` argument(below its 3) in the `search` function call.
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
