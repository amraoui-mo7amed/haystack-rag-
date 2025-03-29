# Project Documentation: Scientific RAG System

## Project Overview
This project implements a Retrieval-Augmented Generation (RAG) system using LangChain, designed to answer user questions based on provided scientific articles. The system is divided into two main components:

1. **Jupyter Notebook Backend**: Contains the core RAG implementation with these key features:
   - Document loading and processing
   - Text embedding and vector storage
   - Question answering chain setup
   - API endpoints for frontend communication

2. **ReactJS Frontend**: A user interface that:
   - Allows users to input questions
   - Displays answers from the RAG system
   - Shows relevant source documents
   - Provides a clean, interactive experience

The system works by processing scientific articles into a searchable knowledge base, then using AI to generate accurate answers based on both the stored knowledge and its general training.

## Jupyter Notebook Backend Implementation

### Step 1: Installation
The first step in setting up the backend is to install the necessary Python packages. These packages provide the tools for document processing, embedding, API creation, and environment management. Here's a breakdown of the key packages and their roles:

1. **`haystack-ai`**:  
   - **Purpose**: A framework for building search systems and question-answering pipelines.  
   - **Role**: Used to create the core RAG pipeline, including document retrieval and answer generation.  

2. **`mistral-haystack`**:  
   - **Purpose**: Integrates Mistral AI models with Haystack.  
   - **Role**: Enables the use of Mistral's language models for generating high-quality answers.  

3. **`sentence-transformers`**:  
   - **Purpose**: Provides pre-trained models for generating sentence embeddings.  
   - **Role**: Used to convert text into vector representations for efficient document retrieval.  

4. **`huggingface_hub`**:  
   - **Purpose**: Facilitates access to models and datasets from Hugging Face.  
   - **Role**: Used to download and manage pre-trained models for embedding and generation tasks.  

5. **`markdown-it-py` and `mdit_plain`**:  
   - **Purpose**: Tools for parsing and rendering Markdown content.  
   - **Role**: Used to process Markdown-formatted scientific articles into plain text for embedding.  

6. **`pypdf`**:  
   - **Purpose**: A library for reading and extracting text from PDF files.  
   - **Role**: Used to extract text from scientific articles provided in PDF format.  

7. **`Flask`**:  
   - **Purpose**: A lightweight web framework for Python.  
   - **Role**: Used to create API endpoints that allow the frontend to communicate with the backend.  

8. **`python-dotenv`**:  
   - **Purpose**: Manages environment variables from `.env` files.  
   - **Role**: Used to securely store and access sensitive configuration data, such as API keys.  

9. **`flask-cors`**:  
   - **Purpose**: Enables Cross-Origin Resource Sharing (CORS) for Flask applications.  
   - **Role**: Allows the React frontend to securely communicate with the Flask backend.

### Step 2: Indexing
The indexing step involves processing and storing documents in a searchable format. This is achieved using a pipeline of Haystack components, each responsible for a specific task. Here's how the indexing process works:

1. **File Type Routing**:  
   - **Component**: `FileTypeRouter`  
   - **Purpose**: Routes files to the appropriate converter based on their MIME type (e.g., PDF, Markdown, plain text).  
   - **Role**: Ensures that each file is processed by the correct converter.  

2. **Document Conversion**:  
   - **Components**: `MarkdownToDocument`, `PyPDFToDocument`, `TextFileToDocument`  
   - **Purpose**: Convert files of different formats into Haystack `Document` objects.  
   - **Role**: Standardizes documents for further processing.  

3. **Document Cleaning and Splitting**:  
   - **Components**: `DocumentCleaner`, `DocumentSplitter`  
   - **Purpose**: Clean the text (e.g., remove extra spaces, special characters) and split large documents into smaller chunks.  
   - **Role**: Prepares documents for embedding by breaking them into manageable pieces and ensuring clean input.  

4. **Document Joining**:  
   - **Component**: `DocumentJoiner`  
   - **Purpose**: Combines processed documents into a single output.  
   - **Role**: Ensures that all document chunks are unified for embedding and storage.  

5. **Document Embedding**:  
   - **Component**: `SentenceTransformersDocumentEmbedder`  
   - **Purpose**: Generates embeddings for documents using sentence-transformers models.  
   - **Role**: Converts text into vector representations for efficient retrieval.  

6. **Document Storage**:  
   - **Component**: `InMemoryDocumentStore`  
   - **Purpose**: Stores documents and their embeddings in memory.  
   - **Role**: Acts as the searchable knowledge base for the RAG system.  

7. **Pipeline Orchestration**:  
   - **Component**: `Pipeline`  
   - **Purpose**: Orchestrates the flow of data between components.  
   - **Role**: Defines the sequence of operations for document processing and retrieval.

### Step 3: Chunking
The chunking step is crucial for breaking down large documents into smaller, more manageable pieces. This ensures that the RAG system can efficiently process and retrieve relevant information. Here's how the chunking process works:

1. **Document Cleaning**:  
   - **Component**: `DocumentCleaner`  
   - **Purpose**: Cleans the text by removing extra spaces, special characters, and other noise.  
   - **Role**: Ensures that the input text is clean and ready for further processing.  

2. **Document Splitting**:  
   - **Component**: `DocumentSplitter`  
   - **Purpose**: Splits large documents into smaller chunks based on word count.  
   - **Parameters**:  
     - `split_by="word"`: Splits the document by word count.  
     - `split_length=150`: Each chunk contains up to 150 words.  
     - `split_overlap=50`: Ensures a 50-word overlap between consecutive chunks to maintain context.  
   - **Role**: Breaks down documents into smaller, semantically meaningful pieces for embedding and retrieval.

### Step 4: Embedding and Storage
The embedding and storage step involves converting the cleaned and chunked documents into vector representations and storing them in a searchable format. Here's how this step works:

1. **Document Embedding**:  
   - **Component**: `SentenceTransformersDocumentEmbedder`  
   - **Purpose**: Generates embeddings for documents using a pre-trained sentence-transformers model.  
   - **Model**: `sentence-transformers/all-MiniLM-L6-v2`  
   - **Role**: Converts text into vector representations, enabling efficient semantic search and retrieval.  

2. **Document Storage**:  
   - **Component**: `DocumentWriter`  
   - **Purpose**: Writes the embedded documents into the document store.  
   - **Role**: Stores documents and their embeddings in the `InMemoryDocumentStore`, making them searchable for the RAG system.

### Step 5: Pipeline Connections
The pipeline connections define the flow of data between components, ensuring that each step in the document processing workflow is executed in the correct order. Here's how the pipeline is connected:

1. **File Type Routing**:  
   - **Connection**: `file_type_router.text/plain` → `text_file_converter.sources`  
   - **Purpose**: Routes plain text files to the `TextFileToDocument` converter.  

2. **PDF Conversion**:  
   - **Connection**: `file_type_router.application/pdf` → `pypdf_converter.sources`  
   - **Purpose**: Routes PDF files to the `PyPDFToDocument` converter.  

3. **Markdown Conversion**:  
   - **Connection**: `file_type_router.text/markdown` → `markdown_converter.sources`  
   - **Purpose**: Routes Markdown files to the `MarkdownToDocument` converter.  

4. **Document Joining**:  
   - **Connections**:  
     - `text_file_converter` → `document_joiner`  
     - `pypdf_converter` → `document_joiner`  
     - `markdown_converter` → `document_joiner`  
   - **Purpose**: Combines the outputs of all converters into a single stream of documents.  

5. **Document Cleaning**:  
   - **Connection**: `document_joiner` → `document_cleaner`  
   - **Purpose**: Cleans the joined documents by removing noise and unnecessary characters.  

6. **Document Splitting**:  
   - **Connection**: `document_cleaner` → `document_splitter`  
   - **Purpose**: Splits the cleaned documents into smaller, manageable chunks.  

7. **Document Embedding**:  
   - **Connection**: `document_splitter` → `document_embedder`  
   - **Purpose**: Converts the document chunks into vector embeddings.  

8. **Document Writing**:  
   - **Connection**: `document_embedder` → `document_writer`  
   - **Purpose**: Writes the embedded documents into the `InMemoryDocumentStore` for retrieval.

### Step 6: Running the Pipeline
The final step is to execute the preprocessing pipeline on the documents stored in a specified directory. This step involves loading the files, processing them through the pipeline, and storing the results in the document store. Here's how it works:

1. **Directory Setup**:  
   - **Component**: `Path` from `pathlib`  
   - **Purpose**: Specifies the directory containing the documents to be processed.  
   - **Directory**: `'articles'`  
   - **Role**: Defines the location of the input files for the pipeline.  

2. **Pipeline Execution**:  
   - **Component**: `preprocessing_pipeline.run`  
   - **Purpose**: Runs the pipeline on all files in the specified directory.  
   - **Input**:  
     - `sources`: A list of all files in the `articles` directory (including subdirectories) using `Path(output_dir).glob("**/*")`.  
   - **Role**: Processes each file through the pipeline, converting, cleaning, splitting, embedding, and storing the documents in the `InMemoryDocumentStore`.  

This step ensures that all documents in the `articles` directory are processed and made available for retrieval in the RAG system.

### Step 7: Environment Setup and Query Pipeline
This step focuses on setting up the environment for the RAG system and building the query pipeline, which is responsible for answering user questions based on the processed documents. Here's what this step accomplishes:

1. **Environment Setup**:  
   - **Purpose**: Loads and manages environment variables, such as API keys, required for accessing external services (e.g., Hugging Face, Mistral).  
   - **Component**: `dotenv`  
   - **Role**: Loads environment variables from a `.env` file, ensuring secure access to sensitive data like API keys.  
   - **Default Values**: If the environment variables are not set, default values are used to ensure the system runs smoothly.  

2. **Query Pipeline Components**:  
   - **`SentenceTransformersTextEmbedder`**:  
     - **Purpose**: Generates embeddings for user queries using the same model (`sentence-transformers/all-MiniLM-L6-v2`) used for document embeddings.  
     - **Role**: Ensures that queries and documents are embedded in the same vector space for accurate retrieval.  

   - **`InMemoryEmbeddingRetriever`**:  
     - **Purpose**: Retrieves the most relevant documents from the `InMemoryDocumentStore` based on the query embedding.  
     - **Role**: Acts as the search engine for the RAG system, finding documents that match the user's query.  

   - **`ChatPromptBuilder`**:  
     - **Purpose**: Constructs a prompt for the language model based on the retrieved documents and the user's question.  
     - **Template**:  
       - **System Message**: Defines the assistant's behavior, including rules for answering in German and referencing document details.  
       - **User Message**: Combines the retrieved documents and the user's question into a structured prompt.  
     - **Role**: Ensures that the language model generates accurate and contextually relevant answers.  

   - **`MistralChatGenerator` or `HuggingFaceAPIChatGenerator`**:  
     - **Purpose**: Generates answers to user questions using a language model.  
     - **Options**:  
       - **Mistral**: Uses Mistral's language model for high-quality responses.  
       - **Hugging Face**: Uses Hugging Face's serverless inference API for model access.  
     - **Role**: Produces the final answer based on the prompt and retrieved documents.  

3. **Pipeline Connections**:  
   - **`embedder` → `retriever`**: Passes the query embedding to the retriever to find relevant documents.  
   - **`retriever` → `chat_prompt_builder`**: Provides the retrieved documents to the prompt builder.  
   - **`chat_prompt_builder` → `llm`**: Sends the constructed prompt to the language model for answer generation.  

4. **Interactive Querying**:  
   - **Purpose**: Allows users to interact with the RAG system by asking questions and receiving answers.  
   - **Role**: Enables real-time question answering based on the processed documents.  

This step ensures that the RAG system is fully operational, capable of retrieving relevant documents and generating accurate answers to user queries.

### Step 8: API Setup for Frontend Communication
This step involves setting up a Flask-based API to enable communication between the frontend and the RAG system. The API allows the frontend to send user questions and receive answers generated by the RAG pipeline. Here's what this step accomplishes:

1. **Flask Application**:  
   - **Purpose**: Creates a web server to handle HTTP requests from the frontend.  
   - **Component**: `Flask`  
   - **Role**: Provides an endpoint (`/api`) for receiving user questions and returning answers.  

2. **Cross-Origin Resource Sharing (CORS)**:  
   - **Purpose**: Allows the frontend (running on a different domain or port) to securely communicate with the Flask backend.  
   - **Component**: `flask_cors.CORS`  
   - **Role**: Ensures that the frontend can make requests to the API without being blocked by browser security policies.  

3. **API Endpoint (`/api`)**:  
   - **Purpose**: Handles POST requests containing user questions.  
   - **Method**: `POST`  
   - **Process**:  
     - Extracts the user's question from the request JSON.  
     - Passes the question to the RAG pipeline for processing.  
     - Returns the generated answer as a JSON response.  
   - **Error Handling**:  
     - Catches and logs any errors that occur during processing.  
     - Returns a 400 status code with error details if something goes wrong.  

4. **Flask Threading**:  
   - **Purpose**: Runs the Flask app in a separate thread to avoid blocking the main program.  
   - **Component**: `threading.Thread`  
   - **Role**: Ensures that the Flask app runs concurrently with other processes, allowing the system to remain responsive.  

5. **Main Program Loop**:  
   - **Purpose**: Keeps the main thread alive while the Flask app runs in the background.  
   - **Component**: `time.sleep`  
   - **Role**: Prevents the program from exiting immediately, ensuring the Flask app remains active.  

6. **Graceful Shutdown**:  
   - **Purpose**: Handles keyboard interrupts (e.g., Ctrl+C) to shut down the Flask app gracefully.  
   - **Role**: Ensures that the system can be stopped cleanly without leaving resources open.  

This step ensures that the RAG system can be accessed via a RESTful API, enabling seamless integration with the frontend and real-time question answering.



## Frontend Implementation: React with Vite

#### React Overview
React is a JavaScript library for building user interfaces that:
- Uses a component-based architecture
- Employs a virtual DOM for efficient updates
- Supports unidirectional data flow
- Utilizes JSX (JavaScript + XML) syntax
- Manages state through hooks or context

#### Vite Project Setup
1. **Project Creation**:
   ```bash
   npm create vite@latest scientific-rag-frontend --template react
   cd scientific-rag-frontend
   npm install

2. **Project Structure**
   ```bash 
    frontend/
        ├── node_modules/       # Dependencies
        ├── public/             # Static assets
        ├── src/
        │   ├── assets/         # Images, fonts
        │   ├── components/     # Reusable UI components
        │   ├── pages/          # Page components
        │   ├── services/       # API service layer
        │   ├── styles/         # Global styles
        │   ├── App.jsx         # Main app component
        │   ├── main.jsx        # Entry point
        │   └── index.css       # Base styles
        ├── .gitignore
        ├── index.html          # Root HTML
        ├── package.json
        ├── vite.config.js      # Build configuration
        └── README.md
   ```

## Frontend Implementation: Chat Interface Component

#### Component Overview
This React component creates an interactive chat interface that connects to our RAG backend, featuring:

1. **State Management**:
   - `showChat`: Toggles chat window visibility
   - `message`: Tracks current input message
   - `messages`: Stores chat history
   - `isLoading`: Manages loading states
   - `messagesEndRef`: Handles auto-scrolling

2. **Core Functionality**:
   - Toggleable chat window with animation
   - Message history display with user/bot differentiation
   - Markdown support for formatted responses
   - Auto-scrolling to newest messages
   - Loading indicators during API calls

3. **UI Structure**:
   - Header with welcome message and hero image
   - Floating chat toggle button
   - Chat window with:
     - Header bar with close button
     - Message display area
     - Input field with send button

#### Key Features

1. **Chat Operations**:
   ```javascript
   const handleSendMessage = async () => {
     // 1. Input validation
     // 2. Message state updates
     // 3. API request to RAG backend
     // 4. Response processing with markdown
     // 5. Error handling
   }```