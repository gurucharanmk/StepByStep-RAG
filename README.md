# StepByStep-RAG - Build a Simple Retrieval-Augmented Generation (RAG) System

**StepByStep-RAG** project is designed to guide you through building a simple Retrieval-Augmented Generation (RAG) system, starting from scratch. The goal is to teach you how to implement various components of RAG **locally** and **for free** by building it step-by-step, without relying on third-party frameworks.

Each folder in this repository contains one step of the process, with detailed instructions and code to help you learn as you build. Start with the first component and move forward sequentially to learn the concepts and implement them one at a time.

## Project Structure:

### 1. [CosineExplorer](./CosineExplorer) (WIP)
   - **What It Is**: In this module, you'll learn about **Cosine Similarity**, a key concept for measuring the similarity between texts. To make this concept easier to grasp, we'll start by using 2D vectors and visualizing them on a Cartesian plane. This simplified approach helps you understand the core idea before extending it to higher-dimensional vector spaces, which is how real-world text similarity is typically computed.
   - **What’s Inside**:
     - `CosineExplorer.py`: The code for calculating cosine similarity between 2D text vectors.
     - `README.md`: Detailed instructions on how to use the code and understand cosine similarity.
   - **Article Link**: [Unlocking the Power of Cosine Similarity: The Heart of Text Understanding](#link-to-your-article)
   - **Sequence**: This is the first step in building your RAG system.


### 2. [SemanticSeek](./SemanticSeek) (WIP)
   - **What It Is**: This module covers **Semantic Search**, which takes the concept of simple keyword matching a step further. You'll build a search system that understands the meaning behind the text, not just the words.
   - **What’s Inside**:
     - `SemanticSeek.py`: The code for semantic search, using techniques like vectorization and similarity to find meaningful results.
     - `README.md`: Instructions for setting up and running semantic search.
   - **Article Link**: [Building Semantic Search: Beyond Keywords for Better Text Retrieval](#link-to-your-article)
   - **Sequence**: This is the second step after cosine similarity, where you learn how to retrieve relevant text.

### 3. [SimpleRAG](./SimpleRAG) (WIP)
   - **What It Is**: In this module, you'll implement the **core RAG system**, combining the retrieval and generation components. This is where you’ll integrate your semantic search with a simple text generation model to build the RAG architecture.
   - **What’s Inside**:
     - `SimpleRAG.py`: The main code for implementing a simple RAG system using retrieval and generation.
     - `README.md`: Instructions for setting up and running the RAG system.
   - **Article Link**: [The Basics of Retrieval-Augmented Generation: Bringing Search and Generation Together](#link-to-your-article)
   - **Sequence**: This is the third step in the process, after completing the search module.

### 4. [ConvoRAG](./ConvoRAG) (WIP)
   - **What It Is**: Finally, in this module, you’ll add **conversational capabilities** to your RAG system. This makes your system capable of handling ongoing interactions, simulating a chatbot-like experience.
   - **What’s Inside**:
     - `ConvoRAG.py`: The code for integrating conversational flow into your RAG system.
     - `README.md`: Instructions for implementing and running the conversational RAG system.
   - **Article Link**: [Creating Conversational RAG: Taking Your AI Interactions to the Next Level](#link-to-your-article)
   - **Sequence**: This is the last step in your journey, where you complete the RAG system with conversational features.

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
