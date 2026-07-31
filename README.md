# Heritage CV + RAG System

An AI prototype for the National Heritage Preservation Trust (NHPT). A visitor uploads a photo
of a building, a computer vision model classifies its architectural style, and a LangChain RAG
chatbot answers questions about it using a heritage knowledge base.

## How it works
1. **CV model** — EfficientNetB0 (transfer learning) classifies images into 6 UK architectural
   styles: Gothic, Georgian, Tudor Revival, Edwardian, Romanesque, Queen Anne.
2. **RAG pipeline** — LangChain + FAISS vector store + Ollama LLM (`gemma3:12b`) retrieve and
   answer from a heritage document knowledge base, with conversation memory for follow-up
   questions.
3. **Handoff** — the CV prediction is passed into the RAG prompt so answers are aware of the
   detected style.
4. **App** — a Streamlit-based chat interface (app.py) integrates all system components.

## Project structure
```
├── README.md                      # Project overview
├── .gitignore                     # Git ignore rules
├── heritage_cv_rag_system.ipynb   # CV model training/evaluation + RAG
├── app.py                         # Streamlit chat application
├── heritage_faiss_index/          # FAISS vector database for RAG
├── heritage_data/                 # Heritage site JSON knowledge documents
└── uk_heritage_architecture/      # Architectural style image dataset used for training
```
