LLM Document Processing System ⚙️
This project implements a Retrieval-Augmented Generation (RAG) system designed to process and answer natural language queries based on a large collection of unstructured documents. Using a powerful Large Language Model (LLM), it can parse complex queries, perform semantic search across documents like policies, contracts, and emails, and deliver precise, structured, and justifiable answers.

The system is built with an interactive web interface using Streamlit, making it easy to upload documents and get instant insights.

✨ Features
Multi-Format Document Ingestion: Natively handles a wide range of document types, including PDFs, Word files (DOCX), emails (EML), and plain text.

Semantic Search: Moves beyond simple keyword matching. It understands the contextual meaning of your query to find the most relevant clauses and information.

Intelligent Query Parsing: Automatically identifies and structures key details from conversational queries (e.g., age, location, procedure).

Structured JSON Output: Delivers answers in a clean, predictable JSON format, containing the decision, any applicable amount, and a detailed justification.

Explainable AI (XAI): Provides full transparency by mapping every part of its decision back to the specific clauses from the source documents.

Interactive Web UI: A user-friendly interface built with Streamlit for easy document uploading and query processing.

🏗️ How It Works (Architecture)
The system follows a Retrieval-Augmented Generation (RAG) pipeline to ensure that the LLM's responses are grounded in the provided documents.

graph TD
    subgraph "Phase 1: Ingestion & Indexing"
        A["📄 User Uploads Documents <br>(PDF, DOCX, EML...)"] --> B{"Parse & Chunk <br>(Unstructured & LangChain)"};
        B --> C{"Generate Embeddings <br>(Sentence Transformers)"};
        C --> D["Vector Store <br>(In-memory index of chunks & embeddings)"];
    end

    subgraph "Phase 2: Query & Response"
        E["🧑‍💻 User Submits Query"] --> F{"Generate Query Embedding"};
        F -- "Cosine Similarity" --> D;
        D -- "Top-K Relevant Chunks" --> G{"Augmented Prompt <br>(Context + Query)"};
        G --> H["🧠 Gemini LLM"];
        H --> I["✅ Structured JSON Response <br>(Decision, Amount, Justification)"];
    end

    style A fill:#D6EAF8,stroke:#333,stroke-width:2px
    style E fill:#D5F5E3,stroke:#333,stroke-width:2px
    style I fill:#FCF3CF,stroke:#333,stroke-width:2px
    style H fill:#FADBD8,stroke:#333,stroke-width:2px

Ingestion & Indexing: Documents are parsed by unstructured, split into chunks by LangChain, and converted into semantic embeddings by Sentence-Transformers.

Query & Response: A user's query is also converted into an embedding to find the most relevant document chunks. These chunks are then fed to the Google Gemini model along with the query to generate a precise, source-based answer.

📂 Project Structure
Here's an overview of the key files and directories in this project:

.
├── main.py           # The core Streamlit application script
├── requirements.txt  # List of Python dependencies
├── .env              # Your local environment variables (API Key)
├── env.example       # Template for the .env file
├── rag_outputs/      # Directory where logs and results are saved
└── README.md         # You are here!

🚀 Getting Started: Full Installation Guide
Follow these instructions to set up and run the project locally.

Step 1: Get Your Google API Key
Before you begin, you need a Google API Key with access to the Gemini models.

Go to Google AI Studio.

Sign in with your Google account.

Click "Create API key".

Copy the generated key. You will need it in a later step.

Step 2: Clone the Repository
Open your terminal or command prompt and run the following command to clone the project to your local machine:

git clone [https://github.com/your-username/llm-document-processing-system.git](https://github.com/your-username/llm-document-processing-system.git)
cd llm-document-processing-system

Step 3: Set Up a Virtual Environment
It's a best practice to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

# Create the virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate

# On Windows (Command Prompt):
.\venv\Scripts\activate.bat

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

You'll know it's active when you see (venv) at the beginning of your terminal prompt.

Step 4: Install Dependencies
The requirements.txt file lists all the Python packages this project needs. Install them using pip:

pip install -r requirements.txt

Step 5: Configure Your API Key
Your API key should be kept secret. We'll use an .env file to store it securely.

First, create a copy of the example file and name it .env:

# On macOS/Linux:
cp env.example .env

# On Windows:
copy env.example .env

Next, open the new .env file with a text editor and paste your Google API Key. The file content should look like this:

.env:

# Replace YOUR_GOOGLE_API_KEY_HERE with your actual key from Step 1
API_KEY="YOUR_GOOGLE_API_KEY_HERE"

The application will automatically load this key when it starts.

Step 6: Run the Application
You're all set! With your virtual environment active and your .env file configured, launch the Streamlit app:

streamlit run main.py

Open your web browser and navigate to the local URL provided by Streamlit (usually http://localhost:8501).

📋 Usage Instructions
Upload Documents: In the sidebar, click "Browse files" to upload one or more documents. The system will automatically process and index them, showing a success message when done.

Ask a Question: Type your query into the text box labeled "Enter your query:". Be as specific or as conversational as you like.

Process Query: Click the "Process Query" button.

Get Results: The system will perform the RAG pipeline and display the structured JSON response. The justification array shows exactly which clauses from which documents were used to arrive at the decision.

Sample Query:
46M, knee surgery in Pune, 3-month old policy

Sample JSON Response:
{
  "decision": "Approved",
  "amount": 50000,
  "justification": [
    {
      "source": "health_insurance_policy.pdf",
      "clause": "Clause 4.1.2: Orthopedic procedures, including knee ligament repair, are covered for all policyholders above 18 years of age.",
      "reasoning": "The query for a 'knee surgery' matches the covered 'orthopedic procedures' mentioned in this clause, and the age '46' is above the minimum requirement of 18."
    }
  ]
}

🤔 Troubleshooting
Here are solutions to some common issues:

streamlit: command not found: This means your virtual environment is not activated. Run the activation command for your OS (e.g., source venv/bin/activate) from inside the project directory and try again.

API Key Errors (PermissionDenied, API_KEY_INVALID):

Ensure your .env file is in the project's root directory and is named correctly (it must start with a dot).

Double-check that you have copied the API key correctly into the .env file.

Make sure you have enabled the "Gemini API" in your Google Cloud project dashboard if you are using a project-based key.

Dependency Installation Fails: Some dependencies of unstructured might require system-level packages. For example, on Debian/Ubuntu, you might need to run sudo apt-get install libmagic-dev poppler-utils. Check the unstructured documentation for OS-specific requirements if you encounter errors during pip install.

App shows "Please upload/process documents first": This message appears if you try to query before uploading and processing files. Please use the sidebar to upload documents first. The system will confirm when they are ready.

📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.
