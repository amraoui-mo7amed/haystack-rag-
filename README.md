# Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system using haystack, designed to answer user questions based on provided scientific articles. The system is divided into two main components:

**Jupyter Notebook Backend:** Holds the main code along side the  API

**ReactJS Frontend:** The ReactJS application serves as the user interface, communicating with the RAG API endpoints exposed by the Jupyter notebook backend.

## Setup and Activation

1. Clone the repository:
   ```bash
   git clone https://github.com/amraoui-mo7amed/haystack-rag-
   cd haystack-rag-
   ```

2. Create a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   # Linux/MacOS
   python3 -m venv venv
   ```

3. Activate the virtual environment:
   ```bash
   # Windows
   venv\Scripts\activate
   # After activation, your command prompt should show (venv) prefix

   # Linux/MacOS
   source venv/bin/activate
   # After activation, your terminal should show (venv) prefix
   ```

4. Verify the virtual environment is active:
   ```bash
   # Both Windows and Linux/MacOS
   which python
   # Should point to the virtual environment's Python executable
   ```

5. Upgrade pip in the virtual environment:
   ```bash
   # Both Windows and Linux/MacOS
   python -m pip install --upgrade pip
   ```

## Running the Application

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Jupyter notebook server:
   ```bash
   jupyter lab
   ```

3. In a separate terminal, navigate to the ReactJS app directory and start the frontend:
   ```bash
   cd frontend
   npm install
   npm start
   ```

4. The ReactJS application will automatically open in your browser and connect to the RAG API endpoints.

## Key Features
- **RAG Implementation**: Combines retrieval and generation for more accurate responses
- **LangChain Integration**: Enables advanced language model interactions and chaining
- **ReactJS Frontend**: Modern, responsive user interface for interacting with the RAG system
- **API Communication**: Seamless connection between frontend and backend components
