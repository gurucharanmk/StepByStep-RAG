# StepByStep-RAG - Build a Simple Retrieval-Augmented Generation (RAG) System

**StepByStep-RAG** project is designed to guide you through building a simple Retrieval-Augmented Generation (RAG) system, starting from scratch. The goal is to teach you how to implement various components of RAG **locally** and **for free** by building it step-by-step, without relying on third-party frameworks.

Each folder in this repository contains one step of the process, with detailed instructions and code to help you learn as you build. Start with the first component and move forward sequentially to learn the concepts and implement them one at a time.

## Project Structure:

### 1. [CosineExplorer](./CosineExplorer) 
   - **What It Is**: In this module, you'll learn about **Cosine Similarity**, a key concept for measuring the similarity between texts. To make this concept easier to grasp, we'll start by using 2D vectors and visualizing them on a Cartesian plane. This simplified approach helps you understand the core idea before extending it to higher-dimensional vector spaces, which is how real-world text similarity is typically computed.
   - **What’s Inside**:
     - `CosineExplorer.py`: The code for calculating cosine similarity between 2D text vectors.
     - `README.md`: Detailed instructions on how to use the code and understand cosine similarity.
   - **Article Link**: [Unlocking the Power of Cosine Similarity: The Heart of Text Understanding](https://medium.com/@charan4u/unlocking-the-power-of-cosine-similarity-the-heart-of-text-understanding-eed427df745a)
   - **Sequence**: This is the first step in building your RAG system.


### 2. [SemanticSeek](./SemanticSeek)
   - **What It Is**: This module covers **Semantic Search**, which takes the concept of simple keyword matching a step further. You'll build a search system that understands the meaning behind the text, not just the words.
   - **What’s Inside**:
     - `SemanticSeek.py`: The code for semantic search, using techniques like vectorization and similarity to find meaningful results.
     - `README.md`: Instructions for setting up and running semantic search.
   - **Article Link**: [Building Semantic Search: Beyond Keywords for Better Text Retrieval](https://medium.com/@charan4u/building-semantic-search-beyond-keywords-for-better-text-retrieval-b7a27d9d4f8f)
   - **Sequence**: This is the second step after cosine similarity, where you learn how to retrieve relevant text.

### 3. [SimpleRAG](./SimpleRAG)
   - **What It Is**: In this module, you'll implement the **core RAG system**, combining the retrieval and generation components. This is where you’ll integrate your semantic search with a simple text generation model to build the RAG architecture.
   - **What’s Inside**:
     - `SimpleRAG.py`: The main code for implementing a simple RAG system using retrieval and generation.
     - `README.md`: Instructions for setting up and running the RAG system.
   - **Article Link**: [The Basics of Retrieval-Augmented Generation: Bringing Search and Generation Together](https://medium.com/@charan4u/the-basics-of-retrieval-augmented-generation-bringing-search-and-generation-together-ee043a8effa5)
   - **Sequence**: This is the third step in the process, after completing the search module.

### 4. [ConvoRAG](./ConvoRAG) 
   - **What It Is**: Finally, in this module, you’ll add **conversational capabilities** to your RAG system. This makes your system capable of handling ongoing interactions, simulating a chatbot-like experience.
   - **What’s Inside**:
     - `ConvoRAG.py`: The code for integrating conversational flow into your RAG system.
     - `README.md`: Instructions for implementing and running the conversational RAG system.
   - **Article Link**: [Creating Conversational RAG: Taking Your AI Interactions to the Next Level](https://medium.com/@charan4u/creating-conversational-rag-taking-your-ai-interactions-to-the-next-level-6b188f945cfe)
   - **Sequence**: This is the last step in your journey, where you complete the RAG system with conversational features.

### 5. [RAGStream](./RAGStream) (Bonus)
   - **What It Is**: RAGStream is an interactive application that leverages Retrieval-Augmented Generation (RAG) to deliver context-aware, adaptive dialogues in real-time. Built with Streamlit, it supports streaming features for dynamic conversations. This module takes the previous concepts of semantic search and RAG systems to the next level by enabling real-time, coherent conversations that evolve with each interaction. The app also includes doc upload and chunking capabilities for an enhanced user experience, providing a more detailed and responsive chat flow based on uploaded content.
   - **What’s Inside**:
     - `RAGStream.py`: The main code that implements the RAG system with streaming support and conversational features. This code integrates the RAG architecture with real-time conversation and document chunking.
     - `README.md`: Instructions on how to run and set up RAGStream, including detailed information on doc uploads, Streamlit features, and debugging options.
     - `GrandPalaceHotelDocument.txt`: A sample document that can be uploaded to the RAGStream app to showcase the chunking and retrieval features of the system. The app processes this document for adaptive conversations based on its content.
   - **Article Link**: [Building a Free, Open-Source RAG System: A Deep Dive]()
   - **Sequence**: This is a **bonus module** that takes your RAG journey a step further. After completing **ConvoRAG**, where you integrated conversational flow into the system, **RAGStream** builds on this by adding real-time interactivity, streaming support, and document upload capabilities for a fully immersive and adaptable AI-driven conversation experience.

---

## How to Get Started:

1. **Clone this repository** to your local machine:
   ```bash
   git clone https://github.com/gurucharanmk/StepByStep-RAG.git
   ```
2. **Follow the sequence of modules:**
    - Start with CosineExplorer to understand the foundation of similarity measurement.
    - Move on to SemanticSeek to learn about semantic search and how it improves text retrieval.
    - Then, explore SimpleRAG to build the core RAG system.
    - Finally, add conversational capabilities with ConvoRAG.
3. **Run the code** in each folder according to the instructions in the `README.md` files inside each folder.

---

##  Learn from Doing:
This project emphasizes a hands-on approach, so you'll be building and understanding each part of the RAG system step-by-step. The goal is to help you understand each component of the system and how they fit together. By the end of this journey, you'll have a fully functional, local, and free RAG system that you can modify and expand as needed.

---

##   License:
This project is licensed under the `MIT License with Attribution` - see the [LICENSE](LICENSE) file for details.
