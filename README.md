# Heritage CV + RAG System

An AI-powered heritage assistant developed as a prototype for the **National Heritage Preservation Trust (NHPT)**. The system combines **Computer Vision (CV)** and **Retrieval-Augmented Generation (RAG)** to identify the architectural style of a heritage building from an uploaded image and answer visitor questions using a heritage knowledge base.

<img width="946" height="1015" alt="image" src="https://github.com/user-attachments/assets/169e36ec-914f-416c-8b47-53c70c7ba9f8" />


---

## Features

- **Architectural Style Classification**
  - Uses **EfficientNetB0** with transfer learning to classify images into six UK architectural styles.
  - Supported styles:
    - Gothic
    - Georgian
    - Tudor Revival
    - Edwardian
    - Romanesque
    - Queen Anne

- **Retrieval-Augmented Generation (RAG)**
  - Built using **LangChain**, **FAISS**, and **Ollama (Gemma)**.
  - Retrieves relevant information from a heritage knowledge base before generating responses.
  - Supports conversational memory for follow-up questions.

- **Integrated AI Pipeline**
  - The predicted architectural style from the CV model is automatically passed to the RAG system, allowing responses to be tailored to the detected style.

- **Interactive Web Application**
  - A **Streamlit** interface allows users to:
    - Upload an image of a heritage building
    - View the predicted architectural style
    - Ask questions about the building
    - Continue the conversation with contextual memory

---

## System Workflow

1. User uploads an image of a heritage building.
2. The EfficientNetB0 model predicts its architectural style.
3. The predicted style is provided as context to the RAG pipeline.
4. LangChain retrieves relevant information from the FAISS vector database.
5. The Ollama LLM generates a contextual response.
6. The Streamlit application displays both the prediction and the chatbot response.

---

## Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0
- LangChain
- FAISS
- Ollama (Gemma 3:12B)
- Streamlit
- NumPy
- Pandas

---

## Project Structure

```text
heritage-cv-rag-system/
│
├── README.md                      # Project documentation
├── requirements.txt               # Python dependencies
├── .gitignore                     # Git ignore rules
├── heritage_cv_rag_system.ipynb   # CV model training, evaluation, and RAG development
├── app.py                         # Streamlit web application
├── heritage_faiss_index/          # FAISS vector database
├── heritage_data/                 # Heritage knowledge base (JSON documents)
└── uk_heritage_architecture/      # Image dataset used for training
```
