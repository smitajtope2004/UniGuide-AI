import streamlit as st

import requests

import json

from pypdf import PdfReader

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

import re

import os

# =========================================================

# CONFIGURATION

# =========================================================

st.set_page_config(

    page_title="UniGuide AI",

    page_icon="🎓",

    layout="wide"

)

# =========================================================
# PROFESSIONAL UI THEME
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .main-subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 12px;
        padding: 12px;
    }

    .stButton > button {
        border-radius: 8px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.15);
    }
    </style>
    """,
    unsafe_allow_html=True
)

DOCUMENT_FOLDER = "documents"

OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL_NAME = "gemma3:1b"

def call_ai(payload, timeout=180):
    """
    Central AI backend for UniGuide AI.

    Local computer:
        Uses Ollama.

    Streamlit Cloud:
        Uses Gemini API.
    """

    try:
        ai_backend = st.secrets.get("AI_BACKEND")
    except Exception:
        ai_backend = None

    if not ai_backend:
        ai_backend = os.getenv("AI_BACKEND", "ollama")

    ai_backend = str(ai_backend).strip().lower()

    # =========================================================
    # LOCAL MODE - OLLAMA
    # =========================================================
    if ai_backend == "ollama":

        try:
            response = requests.post(
                OLLAMA_URL,
                json=payload,
                timeout=timeout
            )

            return response

        except requests.exceptions.ConnectionError:
            return None

        except requests.exceptions.Timeout:
            return None

        except Exception:
            return None

    # =========================================================
    # CLOUD MODE - GEMINI
    # =========================================================
    elif ai_backend == "gemini":

        try:
            try:
                gemini_api_key = st.secrets.get("GEMINI_API_KEY", "")
            except Exception:
                gemini_api_key = os.getenv("GEMINI_API_KEY", "")

            if not gemini_api_key:
                response = requests.Response()
                response.status_code = 500
                response._content = b'{"error":{"message":"GEMINI_API_KEY is not configured."}}'
                return response

            try:
                gemini_model = st.secrets.get("GEMINI_MODEL", "")
            except Exception:
                gemini_model = os.getenv("GEMINI_MODEL", "")

            if not gemini_model:
                gemini_model = "gemini-2.5-flash"

            messages = payload.get("messages", [])

            system_instruction = None
            contents = []

            for message in messages:

                role = message.get("role", "user")
                content = message.get("content", "")

                if role == "system":

                    if system_instruction is None:
                        system_instruction = content
                    else:
                        system_instruction += "\n\n" + content

                elif role == "assistant":

                    contents.append({
                        "role": "model",
                        "parts": [
                            {
                                "text": content
                            }
                        ]
                    })

                else:

                    contents.append({
                        "role": "user",
                        "parts": [
                            {
                                "text": content
                            }
                        ]
                    })

            gemini_payload = {
                "contents": contents
            }

            if system_instruction:

                gemini_payload["systemInstruction"] = {
                    "parts": [
                        {
                            "text": system_instruction
                        }
                    ]
                }

            gemini_url = (
                "https://generativelanguage.googleapis.com/"
                f"v1beta/models/{gemini_model}:generateContent"
                f"?key={gemini_api_key}"
            )

            gemini_response = requests.post(
                gemini_url,
                json=gemini_payload,
                timeout=timeout
            )

            if gemini_response.status_code == 200:

                data = gemini_response.json()

                generated_text = (
                    data["candidates"][0]
                    ["content"]["parts"][0]["text"]
                )

                # Convert Gemini response into the same
                # structure used by the existing UniGuide code.
                compatible_response = requests.Response()

                compatible_response.status_code = 200

                compatible_response._content = json.dumps({
                    "message": {
                        "content": generated_text
                    }
                }).encode("utf-8")

                compatible_response.headers["Content-Type"] = (
                    "application/json"
                )

                return compatible_response

            return gemini_response

        except requests.exceptions.Timeout:
            return None

        except requests.exceptions.ConnectionError:
            return None

        except Exception as e:

            response = requests.Response()
            response.status_code = 500

            response._content = json.dumps({
                "error": {
                    "message": str(e)
                }
            }).encode("utf-8")

            return response

    # =========================================================
    # UNKNOWN BACKEND
    # =========================================================
    else:

        response = requests.Response()
        response.status_code = 500

        response._content = json.dumps({
            "error": {
                "message": f"Unknown AI backend: {ai_backend}"
            }
        }).encode("utf-8")

        return response

os.makedirs(DOCUMENT_FOLDER, exist_ok=True)

# =========================================================

# SIDEBAR - STUDENT PROFILE

# =========================================================

st.sidebar.title("🎓 UniGuide AI")

st.sidebar.header("👤 Student Profile")

student_name = st.sidebar.text_input(

    "Student Name",

    placeholder="Enter your name"

)

course = st.sidebar.text_input(

    "Course",

    value="MSc AI & DS"

)

year = st.sidebar.selectbox(

    "Year",

    [

        "First Year",

        "Second Year",

        "Third Year",

        "Final Year"

    ]

)

cgpa = st.sidebar.number_input(

    "CGPA",

    min_value=0.0,

    max_value=10.0,

    value=8.84,

    step=0.01

)

# =========================================================

# ACADEMIC PERFORMANCE

# =========================================================

st.sidebar.header("📊 Academic Performance")

attendance = st.sidebar.number_input(

    "Attendance (%)",

    min_value=0.0,

    max_value=100.0,

    value=75.0,

    step=1.0

)

exam_marks = st.sidebar.number_input(

    "Exam Marks (%)",

    min_value=0.0,

    max_value=100.0,

    value=40.0,

    step=1.0

)

# =========================================================
# SUBJECT-WISE PERFORMANCE INSIGHTS
# =========================================================

st.sidebar.subheader("📚 Subject Performance")

subject_count = st.sidebar.number_input(
    "Number of Subjects",
    min_value=1,
    max_value=10,
    value=4,
    step=1,
    key="subject_count"
)

subject_marks = {}

for i in range(int(subject_count)):
    subject_name = st.sidebar.text_input(
        f"Subject {i + 1}",
        value=f"Subject {i + 1}",
        key=f"subject_name_{i}"
    )

    marks = st.sidebar.number_input(
        f"Marks - {subject_name}",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=1.0,
        key=f"subject_marks_{i}"
    )

    subject_marks[subject_name] = marks

if subject_marks:
    performance_average = sum(subject_marks.values()) / len(subject_marks)
    strongest_subject = max(subject_marks, key=subject_marks.get)
    weakest_subject = min(subject_marks, key=subject_marks.get)
    subjects_needing_improvement = [
        name for name, marks in subject_marks.items()
        if marks < 60
    ]
else:
    performance_average = 0
    strongest_subject = ""
    weakest_subject = ""
    subjects_needing_improvement = []

# =========================================================
# GRADE CALCULATION

# =========================================================

def calculate_grade(marks):

    if marks >= 90:

        return "A+"

    elif marks >= 80:

        return "A"

    elif marks >= 70:

        return "B+"

    elif marks >= 60:

        return "B"

    elif marks >= 50:

        return "C"

    elif marks >= 40:

        return "D"

    else:

        return "F"

grade = calculate_grade(exam_marks)

# =========================================================

# ELIGIBILITY

# =========================================================

attendance_ok = attendance >= 75

marks_ok = exam_marks >= 40

overall_eligible = (

    attendance_ok and marks_ok

)

st.sidebar.header("✅ Eligibility")

if st.sidebar.button("Check Eligibility"):

    if overall_eligible:

        st.sidebar.success(

            "Eligible for examination."

        )

    else:

        st.sidebar.error(

            "Not eligible. Check attendance and marks."

        )

# =========================================================

# ADMIN DOCUMENT MANAGEMENT

# =========================================================

st.sidebar.header("🛠️ Admin Panel")

st.sidebar.write(

    "Manage university documents used by UniGuide AI."

)

# =========================================================

# UPLOAD PDF

# =========================================================

uploaded_file = st.sidebar.file_uploader(

    "📤 Upload University PDF",

    type=["pdf"]

)

if uploaded_file is not None:

    file_path = os.path.join(

        DOCUMENT_FOLDER,

        uploaded_file.name

    )

    with open(file_path, "wb") as f:

        f.write(

            uploaded_file.getbuffer()

        )

    st.sidebar.success(

        f"Uploaded: {uploaded_file.name}"

    )

# =========================================================

# REFRESH KNOWLEDGE BASE

# =========================================================

if st.sidebar.button(

    "🔄 Refresh Knowledge Base"

):

    st.sidebar.success(

        "Knowledge base refreshed."

    )

    st.rerun()

# =========================================================

# DOCUMENT LIST

# =========================================================

document_files = [

    file

    for file in os.listdir(

        DOCUMENT_FOLDER

    )

    if file.lower().endswith(".pdf")

]

st.sidebar.write(

    f"📚 Documents: {len(document_files)}"

)

# =========================================================

# DELETE DOCUMENT

# =========================================================

if document_files:

    st.sidebar.subheader(

        "🗑️ Delete Document"

    )

    selected_document = st.sidebar.selectbox(

        "Select document",

        document_files

    )

    if st.sidebar.button(

        "Delete Selected Document"

    ):

        delete_path = os.path.join(

            DOCUMENT_FOLDER,

            selected_document

        )

        try:

            os.remove(

                delete_path

            )

            st.sidebar.success(

                f"Deleted: {selected_document}"

            )

            st.rerun()

        except Exception as e:

            st.sidebar.error(

                f"Error deleting document: {e}"

            )

# =========================================================

# =========================================================
# MAIN TITLE

# =========================================================

st.markdown('<div class="main-title">🎓 UniGuide AI</div>', unsafe_allow_html=True)

st.markdown('<div class="main-subtitle">Your Intelligent University Academic Assistant</div>', unsafe_allow_html=True)

st.subheader(

    "Your Intelligent University Academic Assistant"

)

st.write(

    "Ask questions about university rules, exams, "

    "attendance, grading, library policies and more."

)

# =========================================================

# LOAD ALL DOCUMENTS

# =========================================================

documents = []

for filename in os.listdir(

    DOCUMENT_FOLDER

):

    if filename.lower().endswith(".pdf"):

        file_path = os.path.join(

            DOCUMENT_FOLDER,

            filename

        )

        try:

            reader = PdfReader(

                file_path

            )

            for page_number, page in enumerate(

                reader.pages

            ):

                text = page.extract_text()

                if not text:

                    continue

                # Split numbered sections

                chunks = re.split(

                    r"(?=\d+\\.\s+[A-Za-z])",

                    text

                )

                for chunk in chunks:

                    chunk = chunk.strip()

                    if len(chunk) > 30:

                        documents.append(

                            {

                                "text": chunk,

                                "source": filename,

                                "page": page_number + 1

                            }

                        )

        except Exception as e:

            st.warning(

                f"Could not read {filename}: {e}"

            )

# =========================================================

# CREATE SEARCH INDEX

# =========================================================

if documents:

    document_texts = [

        doc["text"]

        for doc in documents

    ]

    vectorizer = TfidfVectorizer(

        stop_words="english",

        lowercase=True,

        ngram_range=(1, 2)

    )

    tfidf_matrix = vectorizer.fit_transform(

        document_texts

    )

else:

    vectorizer = None

    tfidf_matrix = None

# =========================================================

# KEYWORD EXTRACTION

# =========================================================

def get_keywords(text):

    words = re.findall(

        r"\b[a-zA-Z]{3,}\b",

        text.lower()

    )

    return set(words)

# =========================================================

# RECENT CONVERSATION

# =========================================================

def get_recent_context():

    if "messages" not in st.session_state:

        return ""

    recent_messages = (

        st.session_state.messages[-4:]

    )

    context = ""

    for message in recent_messages:

        if message["role"] == "user":

            context += (

                "Previous user question: "

                + message["content"]

                + "\n"

            )

        elif message["role"] == "assistant":

            context += (

                "Previous assistant answer: "

                + message["content"]

                + "\n"

            )

    return context

# =========================================================

# FOLLOW-UP QUESTION IMPROVEMENT

# =========================================================

def improve_question(question):

    lower_question = (

        question.lower().strip()

    )

    follow_up_phrases = [

        "what about",

        "how about",

        "and what about",

        "what is the rule for",

        "tell me about"

    ]

    is_follow_up = any(

        phrase in lower_question

        for phrase in follow_up_phrases

    )

    if not is_follow_up:

        return question

    recent_context = (

        get_recent_context()

    )

    if recent_context:

        return (

            recent_context

            + "\nCurrent question: "

            + question

        )

    return question

# =========================================================

# DOCUMENT SEARCH

# =========================================================

def search_documents(

    question,

    top_k=4

):

    if (

        not documents

        or vectorizer is None

    ):

        return []

    improved_question = (

        improve_question(question)

    )

    # TF-IDF similarity

    question_vector = vectorizer.transform(

        [improved_question]

    )

    tfidf_scores = cosine_similarity(

        question_vector,

        tfidf_matrix

    )[0]

    # Keyword similarity

    question_keywords = get_keywords(

        improved_question

    )

    keyword_scores = []

    for doc in documents:

        doc_keywords = get_keywords(

            doc["text"]

        )

        if question_keywords:

            overlap = len(

                question_keywords.intersection(

                    doc_keywords

                )

            )

            score = (

                overlap

                /

                len(question_keywords)

            )

        else:

            score = 0

        keyword_scores.append(

            score

        )

    # Hybrid score

    final_scores = [

        0.7 * tfidf_score

        +

        0.3 * keyword_score

        for tfidf_score, keyword_score

        in zip(

            tfidf_scores,

            keyword_scores

        )

    ]

    ranked_indexes = sorted(

        range(

            len(final_scores)

        ),

        key=lambda index:

        final_scores[index],

        reverse=True

    )

    results = []

    for index in ranked_indexes[:top_k]:

        results.append(

            {

                "text":

                documents[index]["text"],

                "source":

                documents[index]["source"],

                "page":

                documents[index]["page"],

                "score":

                final_scores[index]

            }

        )

    return results

# =========================================================

# EXACT GRADING SYSTEM

# =========================================================

def find_grading_system(question):

    lower_question = (

        question.lower()

    )

    grading_keywords = [

        "grading system",

        "grade system",

        "grading criteria",

        "grades",

        "grade ranges",

        "marks to grade"

    ]

    if any(

        keyword in lower_question

        for keyword in grading_keywords

    ):

        return """

The university grading system is:

90–100 → A+

80–89 → A

70–79 → B+

60–69 → B

50–59 → C

40–49 → D

Below 40 → F

"""

    return None

# =========================================================

# EXACT GRADE ANSWER

# =========================================================

def find_grade_answer(

    question,

    contexts

):

    match = re.search(

        r"\b(\d{1,3})\s*(?:%|marks?)?\b",

        question.lower()

    )

    if not match:

        return None

    marks = int(

        match.group(1)

    )

    if marks > 100:

        return None

    if marks >= 90:

        grade_result = "A+"

    elif marks >= 80:

        grade_result = "A"

    elif marks >= 70:

        grade_result = "B+"

    elif marks >= 60:

        grade_result = "B"

    elif marks >= 50:

        grade_result = "C"

    elif marks >= 40:

        grade_result = "D"

    else:

        grade_result = "F"

    grade_keywords = [

        "grade",

        "marks",

        "score",

        "percentage"

    ]

    if any(

        keyword in question.lower()

        for keyword in grade_keywords

    ):

        return (

            f"According to the university "

            f"grading system, {marks} marks "

            f"corresponds to grade "

            f"**{grade_result}**."

        )

    return None

# =========================================================

# DEVICE RULE CHECKER

# =========================================================

def find_device_answer(question):

    lower_question = (

        question.lower()

    )

    if (

        "mobile" in lower_question

        or "phone" in lower_question

    ):

        return (

            "Mobile phones are **not permitted** "

            "in the examination hall."

        )

    if (

        "smartwatch" in lower_question

        or

        "smart watch" in lower_question

    ):

        return (

            "Smart watches are **not permitted** "

            "in the examination hall."

        )

    return None

# =========================================================

# ATTENDANCE CHECKER

# =========================================================

def find_attendance_answer(question):

    lower_question = question.lower()

    if "attendance" not in lower_question:
        return None

    return (
        "The minimum required attendance "
        "is **75%**."
    )

# =========================================================

# PASSING MARK CHECKER

# =========================================================

def find_passing_answer(question):

    lower_question = (

        question.lower()

    )

    keywords = [

        "passing marks",

        "pass marks",

        "minimum marks",

        "passing percentage",

        "minimum passing"

    ]

    if not any(

        keyword in lower_question

        for keyword in keywords

    ):

        return None

    return (

        "The minimum passing requirement "

        "is **40%**."

    )

# =========================================================

# =========================================================

# =========================================================
# MAIN APPLICATION MODULES
# =========================================================

tab_dashboard, tab_study, tab_chat = st.tabs(
    [
        "🏠 Student Dashboard",
        "📚 Study Center",
        "💬 UniGuide AI"
    ]
)

with tab_dashboard:
    # PERFORMANCE INSIGHTS
    # =========================================================

    st.header("📊 AI Student Performance Insights")
    st.write(
        "Enter subject-wise marks in the sidebar to analyze "
        "your academic performance."
    )

    if subject_marks:
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Average Marks", f"{performance_average:.1f}%")

        with col2:
            st.metric("Strongest Subject", strongest_subject)

        with col3:
            st.metric("Needs Most Improvement", weakest_subject)

        with col4:
            st.metric(
                "Subjects Below 60%",
                len(subjects_needing_improvement)
            )

        st.subheader("📈 Subject-wise Performance")

        for name, marks in subject_marks.items():
            st.write(f"**{name}: {marks:.1f}%**")
            st.progress(int(marks))

        st.subheader("💡 Personalized Improvement Suggestions")

        if performance_average >= 80:
            st.success(
                "Excellent performance! Keep maintaining your "
                "current study routine and focus on consistency."
            )
        elif performance_average >= 60:
            st.info(
                "Good performance. Focus on the subjects with "
                "lower marks to improve your overall average."
            )
        else:
            st.warning(
                "Your overall performance needs improvement. "
                "Create a regular study schedule and give extra "
                "time to weaker subjects."
            )

        if subjects_needing_improvement:
            st.warning(
                "Focus more on: "
                + ", ".join(subjects_needing_improvement)
            )

        if performance_average >= 75:
            st.success("🎯 Performance Status: On track")
        else:
            st.warning("🎯 Performance Status: Needs attention")

    # =========================================================

    # ACADEMIC PROGRESS TRACKER
    # =========================================================

    st.header("📈 Academic Progress Tracker")

    st.write(
        "Compare previous and current marks to understand your "
        "academic progress."
    )

    if subject_marks:
        st.subheader("📝 Enter Previous Marks")

        previous_marks = {}

        for subject, current_marks in subject_marks.items():
            previous_marks[subject] = st.number_input(
                f"Previous Marks - {subject}",
                min_value=0.0,
                max_value=100.0,
                value=max(0.0, current_marks - 5),
                step=1.0,
                key=f"previous_marks_{subject}"
            )

        if st.button(
            "📊 Analyze Academic Progress",
            use_container_width=True,
            key="analyze_progress"
        ):
            progress_data = []

            for subject in subject_marks:
                previous = previous_marks[subject]
                current = subject_marks[subject]
                change = current - previous

                if change > 0:
                    trend = "📈 Improved"
                elif change < 0:
                    trend = "📉 Declined"
                else:
                    trend = "➡️ No Change"

                progress_data.append(
                    {
                        "Subject": subject,
                        "Previous": previous,
                        "Current": current,
                        "Change": change,
                        "Trend": trend
                    }
                )

            st.subheader("📊 Progress Summary")

            for item in progress_data:
                st.write(
                    f"**{item['Subject']}** — "
                    f"{item['Previous']:.1f}% → "
                    f"{item['Current']:.1f}% "
                    f"({item['Change']:+.1f}%) "
                    f"{item['Trend']}"
                )

            average_previous = (
                sum(previous_marks.values()) / len(previous_marks)
            )
            average_current = (
                sum(subject_marks.values()) / len(subject_marks)
            )
            overall_change = average_current - average_previous

            st.subheader("🎯 Overall Academic Trend")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Previous Average", f"{average_previous:.1f}%")

            with col2:
                st.metric("Current Average", f"{average_current:.1f}%")

            with col3:
                st.metric("Overall Change", f"{overall_change:+.1f}%")

            if overall_change > 0:
                st.success(
                    "📈 Your overall academic performance is improving."
                )
            elif overall_change < 0:
                st.warning(
                    "📉 Your overall performance has declined. "
                    "Focus on subjects showing a negative trend."
                )
            else:
                st.info(
                    "➡️ Your overall performance has remained stable."
                )

            improved_subjects = [
                subject
                for subject in subject_marks
                if subject_marks[subject] > previous_marks[subject]
            ]

            declining_subjects = [
                subject
                for subject in subject_marks
                if subject_marks[subject] < previous_marks[subject]
            ]

            if improved_subjects:
                st.success(
                    "📈 Improving subjects: "
                    + ", ".join(improved_subjects)
                )

            if declining_subjects:
                st.warning(
                    "📉 Subjects needing attention: "
                    + ", ".join(declining_subjects)
                )
    else:
        st.info(
            "Enter subject-wise marks above to use the Academic "
            "Progress Tracker."
        )

    # STUDENT DASHBOARD

    # =========================================================

    st.header(

        "🎓 Student Dashboard"

    )

    # ---------------------------------------------------------

    # Student Information

    # ---------------------------------------------------------

    st.subheader(

        "👤 Student Information"

    )

    info_col1, info_col2, info_col3 = (

        st.columns(3)

    )

    with info_col1:

        st.write(

            "**Student Name**"

        )

        st.write(

            student_name

            if student_name

            else "Not provided"

        )

    with info_col2:

        st.write(

            "**Course**"

        )

        st.write(

            course

        )

    with info_col3:

        st.write(

            "**Year**"

        )

        st.write(

            year

        )

    # ---------------------------------------------------------

    # Academic Performance

    # ---------------------------------------------------------

    st.subheader(

        "📊 Academic Performance"

    )

    metric1, metric2, metric3, metric4 = (

        st.columns(4)

    )

    with metric1:

        st.metric(

            "CGPA",

            f"{cgpa:.2f}/10"

        )

    with metric2:

        st.metric(

            "Attendance",

            f"{attendance:.0f}%"

        )

    with metric3:

        st.metric(

            "Exam Marks",

            f"{exam_marks:.0f}%"

        )

    with metric4:

        st.metric(

            "Current Grade",

            grade

        )

    # ---------------------------------------------------------

    # Eligibility

    # ---------------------------------------------------------

    st.subheader(

        "✅ Academic Eligibility"

    )

    if overall_eligible:

        st.success(

            "🎉 You are eligible for the examination."

        )

    else:

        st.warning(

            "⚠️ You currently need improvement "

            "to meet the examination requirements."

        )

    # ---------------------------------------------------------

    # Performance Overview

    # ---------------------------------------------------------

    st.subheader(

        "📈 Performance Overview"

    )

    progress_col1, progress_col2 = (

        st.columns(2)

    )

    with progress_col1:

        st.write(

            f"**Attendance: {attendance:.0f}%**"

        )

        st.progress(

            min(

                attendance / 100,

                1.0

            )

        )

        if attendance >= 75:

            st.caption(

                "✅ Attendance requirement satisfied"

            )

        else:

            st.caption(

                "⚠️ Attendance is below "

                "the required 75%"

            )

    with progress_col2:

        st.write(

            f"**Exam Performance: {exam_marks:.0f}%**"

        )

        st.progress(

            min(

                exam_marks / 100,

                1.0

            )

        )

        if exam_marks >= 40:

            st.caption(

                "✅ Minimum passing requirement satisfied"

            )

        else:

            st.caption(

                "⚠️ Marks are below "

                "the minimum passing requirement"

            )

    # =========================================================

    # PERSONALIZED SUGGESTIONS

    # =========================================================

    st.subheader(

        "💡 Personalized Suggestions"

    )

    suggestions = []

    if attendance < 75:

        suggestions.append(

            "📅 Improve your attendance "

            "to reach the required 75%."

        )

    elif attendance < 80:

        suggestions.append(

            "📅 Your attendance is eligible, "

            "but maintaining regular attendance "

            "is recommended."

        )

    else:

        suggestions.append(

            "✅ Your attendance is in a good range."

        )

    if exam_marks < 40:

        suggestions.append(

            "📚 Focus on improving your exam "

            "preparation to reach the 40% "

            "passing requirement."

        )

    elif exam_marks < 60:

        suggestions.append(

            "📚 You have passed, but improving "

            "your marks can strengthen your "

            "academic performance."

        )

    else:

        suggestions.append(

            "🌟 Your exam performance is good. "

            "Keep it up!"

        )

    if cgpa < 6:

        suggestions.append(

            "🎯 Consider creating a regular "

            "study schedule to improve your CGPA."

        )

    elif cgpa < 8:

        suggestions.append(

            "🎯 Your CGPA is good. Consistent "

            "preparation can help you improve "

            "it further."

        )

    else:

        suggestions.append(

            "🌟 Excellent CGPA. "

            "Maintain your performance!"

        )

    for suggestion in suggestions:

        st.write(

            f"- {suggestion}"

        )

    # =========================================================

    # AI ACADEMIC RECOMMENDATIONS

    # =========================================================

    st.subheader(

        "🤖 AI Academic Recommendations"

    )

    st.write(

        "Personalized recommendations based on "

        "your current academic performance."

    )

    recommendations = []

    if attendance < 75:

        recommendations.append(

            (

                "🚨 Attendance Priority",

                f"Your attendance is {attendance:.0f}%, "

                "which is below the required 75%. "

                "Prioritize attending upcoming classes "

                "to improve your eligibility."

            )

        )

    elif attendance < 80:

        recommendations.append(

            (

                "📅 Maintain Attendance",

                f"Your attendance is {attendance:.0f}%. "

                "You meet the minimum requirement, "

                "but maintaining regular attendance "

                "will give you a safer margin."

            )

        )

    else:

        recommendations.append(

            (

                "✅ Strong Attendance",

                f"Your attendance is {attendance:.0f}%. "

                "You are comfortably above the minimum "

                "attendance requirement."

            )

        )

    if exam_marks < 40:

        recommendations.append(

            (

                "🚨 Exam Performance Priority",

                f"Your exam score is {exam_marks:.0f}%, "

                "which is below the 40% passing requirement. "

                "Focus on revision, practice questions "

                "and difficult topics."

            )

        )

    elif exam_marks < 60:

        recommendations.append(

            (

                "📚 Improve Exam Performance",

                f"Your exam score is {exam_marks:.0f}%. "

                "You have passed, but additional practice "

                "could help you reach a stronger grade."

            )

        )

    elif exam_marks < 80:

        recommendations.append(

            (

                "📈 Good Performance",

                f"Your exam score is {exam_marks:.0f}%. "

                "Focus on consistent revision and "

                "practice to move toward an A-level grade."

            )

        )

    else:

        recommendations.append(

            (

                "🌟 Excellent Exam Performance",

                f"Your exam score is {exam_marks:.0f}%. "

                "Your performance is strong. "

                "Continue your current study strategy."

            )

        )

    if cgpa < 6:

        recommendations.append(

            (

                "🎯 CGPA Improvement",

                f"Your CGPA is {cgpa:.2f}. "

                "Create a consistent weekly study schedule "

                "and focus on subjects where you score lower."

            )

        )

    elif cgpa < 8:

        recommendations.append(

            (

                "🎯 CGPA Growth",

                f"Your CGPA is {cgpa:.2f}. "

                "You have a good foundation. "

                "Regular revision and practice can help "

                "you move toward 8+."

            )

        )

    elif cgpa < 9:

        recommendations.append(

            (

                "🌟 Strong CGPA",

                f"Your CGPA is {cgpa:.2f}. "

                "Your academic performance is strong. "

                "Consistent preparation can help you "

                "reach the 9+ range."

            )

        )

    else:

        recommendations.append(

            (

                "🏆 Excellent CGPA",

                f"Your CGPA is {cgpa:.2f}. "

                "Excellent academic performance. "

                "Focus on maintaining consistency."

            )

        )

    for title, message in recommendations:

        with st.expander(

            title,

            expanded=True

        ):

            st.write(

                message

            )

    # =========================================================

    # CURRENT ACADEMIC PRIORITY

    # =========================================================

    st.subheader(

        "🎯 Your Current Priority"

    )

    if (

        attendance < 75

        and exam_marks < 40

    ):

        st.error(

            "High Priority: Improve both attendance "

            "and examination performance."

        )

    elif attendance < 75:

        st.warning(

            "Priority: Improve your attendance "

            "to meet the 75% requirement."

        )

    elif exam_marks < 40:

        st.warning(

            "Priority: Improve your examination "

            "performance to reach the 40% passing mark."

        )

    elif exam_marks < 60:

        st.info(

            "Priority: Strengthen exam preparation "

            "to improve your grade."

        )

    else:

        st.success(

            "🎉 Your academic performance is on track. "

            "Focus on maintaining consistency."

        )

    st.divider()

    # =========================================================

with tab_study:
    # AI DOCUMENT SUMMARIZER

    # =========================================================

    def summarize_document(document_name):

        file_path = os.path.join(

            DOCUMENT_FOLDER,

            document_name

        )

        try:

            reader = PdfReader(file_path)

            full_text = ""

            for page in reader.pages:

                page_text = page.extract_text() or ""

                full_text += page_text + "\n"

            if not full_text.strip():

                return "❌ I couldn't extract readable text from this PDF."

            max_characters = 30000

            if len(full_text) > max_characters:

                full_text = full_text[:max_characters]

            summary_prompt = f"""

    You are UniGuide AI, a university academic assistant.

    Analyze the following university document.

    Document name:

    {document_name}

    DOCUMENT CONTENT:

    {full_text}

    Create a simple, student-friendly summary.

    Use these sections:

    ## 📝 Summary

    Give a concise overall summary.

    ## 🔑 Key Points

    List the most important points.

    ## 📌 Important Rules

    List important rules, requirements, percentages,

    deadlines, restrictions, and procedures.

    ## 📚 Topics to Remember

    List important topics students should remember

    for exams or academic activities.

    ## ❓ Possible Exam Questions

    Create 5 possible questions based only on this document.

    IMPORTANT:

    - Use only information present in the document.

    - Do not invent university rules.

    - Keep the explanation simple and useful for students.

    - Preserve important numbers and percentages.

    """

            response = call_ai(
    {

                    "model": MODEL_NAME,

                    "messages": [

                        {

                            "role": "system",

                            "content": (

                                "You are UniGuide AI. "

                                "Summarize university documents accurately "

                                "without inventing information."

                            )

                        },

                        {

                            "role": "user",

                            "content": summary_prompt

                        }

                    ],

                    "stream": False

                },

                timeout=180

            )

            if response.status_code == 200:

                return response.json()["message"]["content"]

            return f"❌ Ollama returned error {response.status_code}."

        except requests.exceptions.ConnectionError:

            return (

                "❌ Could not connect to Ollama. "

                "Please make sure Ollama is running."

            )

        except Exception as e:

            return f"❌ Error while summarizing document: {e}"

    st.header("📄 AI Document Summarizer")

    st.write(

        "Select a university PDF and let UniGuide AI "

        "generate a student-friendly summary, key points, "

        "important rules, topics to remember, and possible "

        "exam questions."

    )

    if document_files:

        summary_document = st.selectbox(

            "📄 Select a PDF to summarize",

            document_files,

            key="summary_document"

        )

        if st.button(

            "🤖 Generate AI Summary",

            use_container_width=True

        ):

            with st.spinner(

                "🤖 Reading and summarizing the document..."

            ):

                summary = summarize_document(

                    summary_document

                )

            st.markdown(summary)

            st.caption(

                f"📄 Source document: {summary_document}"

            )

    else:

        st.info(

            "No university PDF is available yet. "

            "Upload a PDF from the Admin Panel first."

        )

    st.divider()

    # SMART UNIVERSITY SEARCH

    # =========================================================

    st.header(

        "🔎 Smart University Search"

    )

    st.write(

        "Search university rules, policies, exams, "

        "attendance, library information and more."

    )

    search_query = st.text_input(

        "Search the university knowledge base",

        placeholder="Example: attendance rules"

    )

    if search_query:

        search_results = search_documents(

            search_query,

            top_k=5

        )

        if search_results:

            for i, result in enumerate(

                search_results,

                start=1

            ):

                with st.expander(

                    f"Result {i} — "

                    f"{result['source']} "

                    f"(Page {result['page']})"

                ):

                    st.write(

                        result["text"]

                    )

                    st.caption(

                        f"Relevance Score: "

                        f"{result['score']:.2f}"

                    )

        else:

            st.info(

                "No relevant information found."

            )

    # =========================================================

    # UNIVERSITY FAQ / QUICK ACTIONS

    # =========================================================

    st.header(

        "⚡ Quick University Questions"

    )

    st.write(

        "Click a topic to quickly get information "

        "from the university knowledge base."

    )

    quick_questions = {

        "📅 Attendance Rules":

            "What are the attendance rules?",

        "📝 Exam Rules":

            "What are the examination rules?",

        "📊 Grading System":

            "What is the grading system?",

        "📚 Library Rules":

            "What are the library rules?"

    }

    quick_col1, quick_col2, quick_col3, quick_col4 = (

        st.columns(4)

    )

    quick_columns = [

        quick_col1,

        quick_col2,

        quick_col3,

        quick_col4

    ]

    for column, (

        button_name,

        question

    ) in zip(

        quick_columns,

        quick_questions.items()

    ):

        with column:

            if st.button(

                button_name,

                use_container_width=True

            ):

                # Exact answers first

                quick_answer = None

                if "attendance" in question.lower():

                    quick_answer = (

                        find_attendance_answer(

                            question

                        )

                    )

                elif "grading" in question.lower():

                    quick_answer = (

                        find_grading_system(

                            question

                        )

                    )

                if quick_answer:

                    st.success(

                        quick_answer

                    )

                else:

                    results = search_documents(

                        question,

                        top_k=3

                    )

                    if results:

                        st.success(

                            f"Information about "

                            f"{button_name}"

                        )

                        for result in results:

                            st.write(

                                result["text"]

                            )

                            st.caption(

                                f"📄 Source: "

                                f"{result['source']} "

                                f"— Page {result['page']} "

                                f"— Relevance: "

                                f"{result['score']:.2f}"

                            )

                            st.divider()

                    else:

                        st.info(

                            "No relevant information "

                            "was found in the university "

                            "documents."

                        )

    st.divider()

    # =========================================================

    # AI STUDY PLANNER

    # =========================================================

    st.header(

        "📚 AI Study Planner"

    )

    st.write(

        "Create a personalized study plan using "

        "your local UniGuide AI model."

    )

    planner_col1, planner_col2 = (

        st.columns(2)

    )

    with planner_col1:

        study_subject = st.text_input(

            "📖 Subject / Topic",

            placeholder="Example: Machine Learning"

        )

        study_days = st.number_input(

            "📅 Number of Days",

            min_value=1,

            max_value=30,

            value=7,

            step=1

        )

    with planner_col2:

        study_hours = st.number_input(

            "⏰ Study Hours Per Day",

            min_value=1,

            max_value=12,

            value=3,

            step=1

        )

        target_marks = st.number_input(

            "🎯 Target Marks (%)",

            min_value=1,

            max_value=100,

            value=80,

            step=1

        )

    if st.button(

        "🤖 Generate AI Study Plan",

        use_container_width=True

    ):

        if not study_subject.strip():

            st.warning(

                "Please enter a subject or topic."

            )

        else:

            total_hours = (

                study_days * study_hours

            )

            study_prompt = f"""

    Create a practical and realistic study plan

    for a university student.

    Student Profile:

    Course: {course}

    Year: {year}

    Current CGPA: {cgpa}

    Current Exam Marks: {exam_marks}%

    Current Grade: {grade}

    Study Details:

    Subject / Topic:

    {study_subject}

    Number of Days:

    {study_days}

    Study Hours Per Day:

    {study_hours}

    Total Available Study Hours:

    {total_hours}

    Target Marks:

    {target_marks}%

    Create a day-by-day study plan.

    For every day include:

    1. Topics to study

    2. Suggested study time

    3. Practice or revision activity

    The final day should include:

    - Complete revision

    - Practice questions

    - Mock test or self-assessment

    Make the plan realistic and easy for

    a university student to follow.

    Do not make the response unnecessarily long.

    """

            with st.spinner(

                "🤖 Creating your personalized "

                "study plan..."

            ):

                try:

                    response = call_ai(
                        {
                            "model": MODEL_NAME,
                            "messages": [
                                {
                                    "role": "system",
                                    "content": (
                                        "You are UniGuide AI, "
                                        "a helpful university "
                                        "academic study planner."
                                    )
                                },
                                {
                                    "role": "user",
                                    "content": study_prompt
                                }
                            ],
                            "stream": False
                        },
                        timeout=180
                    )

                    if response is not None and response.status_code == 200:

                        data = response.json()

                        study_plan = (
                            data["message"]["content"]
                        )

                        st.success(
                            "🎉 Your personalized "
                            "study plan is ready!"
                        )

                        st.markdown(
                            study_plan
                        )

                        st.info(
                            f"📊 Total planned study time: "
                            f"**{total_hours} hours**"
                        )

                    else:

                        st.error(
                            "Unable to generate the study plan. "
                            "Please check the configured AI backend."
                        )

                except Exception as e:

                    st.error(
                        f"Error generating study plan: {e}"
                    )

    st.divider()

    # =========================================================

with tab_chat:
    # CHAT SECTION

    # =========================================================

    st.header(

        "💬 Chat with UniGuide AI"

    )

    if "messages" not in st.session_state:

        st.session_state.messages = []

    # =========================================================

    # DISPLAY CHAT HISTORY

    # =========================================================

    for message in (

        st.session_state.messages

    ):

        with st.chat_message(

            message["role"]

        ):

            st.markdown(

                message["content"]

            )

    # =========================================================

    # CHAT INPUT

    # =========================================================

    user_question = st.chat_input(

        "Ask UniGuide AI..."

    )

    if user_question:

        # -----------------------------------------------------

        # Save user message

        # -----------------------------------------------------

        st.session_state.messages.append(

            {

                "role":

                "user",

                "content":

                user_question

            }

        )

        with st.chat_message(

            "user"

        ):

            st.markdown(

                user_question

            )

        # -----------------------------------------------------

        # Exact rule checks

        # -----------------------------------------------------

        answer = find_grading_system(

            user_question

        )

        if not answer:

            answer = find_device_answer(

                user_question

            )

        if not answer:

            answer = find_attendance_answer(

                user_question

            )

        if not answer:

            answer = find_passing_answer(

                user_question

            )

        # -----------------------------------------------------

        # RAG SEARCH

        # -----------------------------------------------------

        contexts = []

        if not answer:

            contexts = search_documents(

                user_question,

                top_k=4

            )

            grade_answer = find_grade_answer(

                user_question,

                contexts

            )

            if grade_answer:

                answer = grade_answer

        # -----------------------------------------------------

        # ASK OLLAMA

        # -----------------------------------------------------

        if not answer:

            if contexts:

                context_text = "\n\n".join(

                    [

                        (

                            f"Source: "

                            f"{c['source']} "

                            f"(Page {c['page']})\n"

                            f"{c['text']}"

                        )

                        for c in contexts

                    ]

                )

            else:

                context_text = (

                    "No relevant university "

                    "document information "

                    "was found."

                )

            recent_context = (

                get_recent_context()

            )

            system_prompt = f"""

    You are UniGuide AI, an intelligent

    university academic assistant.

    Student Profile:

    Name: {

        student_name

        if student_name

        else "Not provided"

    }

    Course: {course}

    Year: {year}

    CGPA: {cgpa}

    Attendance: {attendance}%

    Exam Marks: {exam_marks}%

    Current Grade: {grade}

    Your job is to answer university-related

    questions using the provided university

    documents.

    IMPORTANT RULES:

    1. Use the provided university documents

       whenever possible.

    2. Do not invent university policies.

    3. If the answer is not available in the

       provided documents, clearly say:

       "I couldn't find this information

       in the university documents."

    4. Be concise and student-friendly.

    5. For follow-up questions, use the

       previous conversation context.

    University Documents:

    {context_text}

    Recent Conversation:

    {recent_context}

    """

            prompt = f"""

    Student question:

    {user_question}

    Provide the most accurate answer based

    on the university documents.

    """

            try:

                response = call_ai(
                    {
                        "model": MODEL_NAME,
                        "messages": [
                            {
                                "role": "system",
                                "content": system_prompt
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "stream": False
                    },
                    timeout=120
                )

                if response is not None and response.status_code == 200:
                    data = response.json()
                    answer = data["message"]["content"]
                else:
                    answer = "Sorry, I could not generate an AI response."

            except Exception as e:
                answer = f"AI error: {e}"

        # -----------------------------------------------------

        # -----------------------------------------------------

        # Display answer

        # -----------------------------------------------------

        with st.chat_message("assistant"):

            st.markdown(answer)

            if contexts:

                st.markdown("### 📚 Sources")

                for context in contexts:

                    st.caption(
                        f"📄 {context['source']} "
                        f"— Page {context['page']} "
                        f"— Relevance: "
                        f"{context['score']:.2f}"
                    )

        # -----------------------------------------------------

        # Save assistant message

        # -----------------------------------------------------

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

    # =========================================================

    # CLEAR CHAT

    if st.button(
        "🗑️ Clear Chat"
    ):
        st.session_state.messages = []
        st.rerun()
