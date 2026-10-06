# 🎓 UniGuide AI

### Your Intelligent University Academic Assistant

UniGuide AI is an AI-powered university academic assistant designed to help students quickly access academic information, understand university rules, analyze their performance, and receive personalized study guidance.

It combines **Retrieval-Augmented Generation (RAG)**, **Streamlit**, **Python**, and a **local Ollama LLM** to provide a practical and privacy-friendly academic assistant.

---

## ✨ Features

### 🤖 AI Academic Chatbot
- Answers questions about university academic rules
- Supports follow-up questions
- Uses uploaded university documents as its knowledge source
- Displays relevant document sources

### 📚 RAG-Based Knowledge System
- Searches across multiple university PDFs
- Uses TF-IDF and keyword-based retrieval
- Retrieves relevant document sections before generating answers
- Helps reduce unsupported AI responses

### 📄 AI Document Summarizer
- Summarizes uploaded academic documents
- Extracts key points and important rules
- Generates topics to remember
- Provides possible exam questions

### 🎓 Student Dashboard
- Student profile
- CGPA
- Attendance
- Exam performance
- Academic eligibility

### 📊 Performance Analysis
- Subject-wise marks
- Average marks
- Strongest and weakest subjects
- Subjects requiring improvement

### 📈 Academic Progress Tracker
- Compares previous and current marks
- Identifies improving subjects
- Identifies declining subjects
- Shows overall academic trend

### 🤖 Personalized Study Recommendations
- Uses student performance data
- Identifies study priorities
- Provides personalized improvement suggestions

### 🗓️ AI Study Planner
- Helps students organize their study activities
- Creates study plans based on academic needs

### ⚙️ Admin Panel
- Upload university PDF documents
- Manage the academic knowledge base
- Delete outdated documents

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web application interface |
| Ollama | Local AI/LLM |
| Gemma 3 1B | Local language model |
| Scikit-learn | TF-IDF and similarity search |
| PyPDF | PDF processing |
| Requests | Communication with Ollama |
| Git & GitHub | Version control |

---

## 🏗️ Architecture

```text
Student
   ↓
Streamlit Interface
   ↓
UniGuide AI
   ↓
Question Processing
   ↓
RAG Retrieval
   ↓
University PDF Knowledge Base
   ↓
Relevant Information
   ↓
Ollama Local LLM
   ↓
AI Response + Source

📂 Project Structure

UniGuide-AI/
│
├── app.py
├── pdf_reader.py
├── requirements.txt
├── .gitignore
│
├── documents/
│   ├── UniGuide_Exam_and_Grading_Policy.pdf
│   └── UniGuide_Sample_Academic_Rules.pdf
│
└── data/


🚀 Installation
1. Clone the repository
git clone https://github.com/smitajtope2004/UniGuide-AI.git

2. Open the project
cd UniGuide-AI

3. Install dependencies
pip install -r requirements.txt

4. Install Ollama
Install Ollama and download the required model:
ollama pull gemma3:1b

Make sure Ollama is running locally.
5. Run UniGuide AI
streamlit run app.py

The application will open in your browser.


🔐 Privacy
UniGuide AI is designed to use a local Ollama model, so the core AI interaction can run locally without requiring a paid cloud LLM API.
Do not add API keys, passwords, .env files, or private documents to the repository.


🎯 Project Goal
The goal of UniGuide AI is to create a practical academic assistant that can help students:
- Find university rules quickly
- Understand academic policies
- Analyze their academic performance
- Identify areas for improvement
- Plan their studies
- Interact with university documents using natural language


🔮 Future Improvements
Possible future improvements include:
- Advanced semantic embeddings
- Better document citation
- Student authentication
- University-specific deployment
- Database integration
- Faculty/admin dashboards
- Improved multilingual support


👩‍💻 Author
Smita Tope
AI & DS Student
GitHub:
https://github.com/smitajtope2004
⭐ Project
If you find this project useful, consider giving the repository a ⭐.

