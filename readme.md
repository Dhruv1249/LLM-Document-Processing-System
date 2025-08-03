# LLM Document Processing System ⚙️

This project implements a Retrieval-Augmented Generation (RAG) system designed to process and answer natural language queries based on a large collection of unstructured documents. Using a powerful Large Language Model (LLM), it can parse complex queries, perform semantic search across documents like policies, contracts, and emails, and deliver precise, structured, and justifiable answers.

The system is built with an interactive web interface using Streamlit, making it easy to upload documents and get instant insights.



## ✨ Features

-   **Multi-Format Document Ingestion**: Natively handles a wide range of document types, including **PDFs, Word files (DOCX), emails (EML), and plain text**.
-   **Semantic Search**: Moves beyond simple keyword matching. It understands the contextual meaning of your query to find the most relevant clauses and information.
-   **Intelligent Query Parsing**: Automatically identifies and structures key details from conversational queries (e.g., age, location, procedure).
-   **Structured JSON Output**: Delivers answers in a clean, predictable JSON format, containing the decision, any applicable amount, and a detailed justification.
-   **Explainable AI (XAI)**: Provides full transparency by mapping every part of its decision back to the specific clauses from the source documents.
-   **Interactive Web UI**: A user-friendly interface built with Streamlit for easy document uploading and query processing.

---

## 🏗️ How It Works (Architecture)

The system follows a Retrieval-Augmented Generation (RAG) pipeline to ensure that the LLM's responses are grounded in the provided documents.

```mermaid
graph TD
    subgraph "Phase 1: Ingestion & Indexing"
        A[📄 User Uploads Documents <br>(PDF, DOCX, EML...)] --> B{Parse & Chunk <br>(Unstructured & LangChain)};
        B --> C{Generate Embeddings <br>(Sentence Transformers)};
        C --> D[Vector Store <br>(In-memory index of chunks & embeddings)];
    end

    subgraph "Phase 2: Query & Response"
        E[🧑‍💻 User Submits Query] --> F{Generate Query Embedding};
        F -- Cosine Similarity --> D;
        D -- Top-K Relevant Chunks --> G{Augmented Prompt <br>(Context + Query)};
        G --> H[🧠 Gemini LLM];
        H --> I[✅ Structured JSON Response <br>(Decision, Amount, Justification)];
    end

    style A fill:#D6EAF8,stroke:#333,stroke-width:2px
    style E fill:#D5F5E3,stroke:#333,stroke-width:2px
    style I fill:#FCF3CF,stroke:#333,stroke-width:2px
    style H fill:#FADBD8,stroke:#333,stroke-width:2px