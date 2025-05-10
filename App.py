import streamlit as st
import spacy
from spacy.matcher import PhraseMatcher
import fitz  # PyMuPDF
import tempfile
import os

# Load spaCy model
@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")

nlp = load_model()

# Pre-defined skill keywords list
SKILLS_DB = [
    "machine learning", "deep learning", "nlp", "natural language processing",
    "python", "java", "c++", "c#", "sql", "tensorflow", "pytorch", "keras",
    "scikit-learn", "data analysis", "data mining", "computer vision",
    "artificial intelligence", "cloud", "aws", "azure", "gcp", "docker",
    "kubernetes", "react", "angular", "node.js", "flask", "django", "linux",
    "git", "rest api", "microservices", "hadoop", "spark", "tableau",
    "matlab", "r", "excel", "power bi", "jira", "confluence", "tableau",
    "spark", "hive", "nosql", "mongodb", "agile", "scrum"
]

# Prepare matcher for skills
def create_skill_matcher(nlp, skills):
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(skill) for skill in skills]
    matcher.add("SKILLS", patterns)
    return matcher

skill_matcher = create_skill_matcher(nlp, SKILLS_DB)

def extract_text_from_pdf(file_path):
    text = ""
    try:
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
    return text

def extract_skills(text, nlp, matcher):
    doc = nlp(text)
    matches = matcher(doc)
    skills = set()
    for match_id, start, end in matches:
        span = doc[start:end]
        skills.add(span.text.lower())
    return sorted(skills)

def calc_match_score(candidate_skills, job_skills):
    if not job_skills:
        return 0.0
    matched = set(candidate_skills).intersection(set(job_skills))
    score = len(matched) / len(job_skills)
    return score

# Streamlit UI
st.set_page_config(page_title="AI-Powered Resume Parser", layout="centered")

st.title("📝 AI-Powered Resume Parser")
st.write("""
Upload your resumes (PDF or TXT) and get a list of extracted skills using NLP (spaCy).
""")

# Text box for job description
job_description = st.text_area("Enter Job Description", height=150)

uploaded_files = st.file_uploader("Choose your resume files", type=["pdf", "txt"], accept_multiple_files=True)

if uploaded_files:
    st.subheader("Extracted Skills and Candidate Ranking")

    # Extract job description skills once if job description is provided
    job_skills = []
    if job_description.strip():
        job_skills = extract_skills(job_description, nlp, skill_matcher)

    candidates = []
    for uploaded_file in uploaded_files:
        # Save uploaded file to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tfile:
            tfile.write(uploaded_file.read())
            temp_path = tfile.name

        file_ext = os.path.splitext(uploaded_file.name)[1].lower()

        with st.spinner(f"Processing {uploaded_file.name}..."):
            if file_ext == ".pdf":
                text = extract_text_from_pdf(temp_path)
            elif file_ext == ".txt":
                with open(temp_path, "r", encoding="utf-8") as f:
                    text = f.read()
            else:
                st.error("Unsupported file format. Please upload PDF or TXT files.")
                text = ""

            if text:
                skills = extract_skills(text, nlp, skill_matcher)
                if skills:
                    score = calc_match_score(skills, job_skills) if job_skills else None
                    candidates.append({
                        "filename": uploaded_file.name,
                        "skills": skills,
                        "score": score
                    })
                else:
                    st.warning(f"No skills found in {uploaded_file.name}.")
            else:
                st.error(f"No text found in {uploaded_file.name}.")

        # Cleanup temp file
        os.remove(temp_path)

    # Display candidates info and ranking in list format
    if candidates:
        # Sort candidates by score if job description provided, else order of upload
        if job_skills:
            candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)

        st.markdown("## Candidate Ranking")

        for idx, candidate in enumerate(candidates, start=1):
            st.markdown(f"**{idx}. {candidate['filename']}**")
            if candidate["score"] is not None:
                percent = candidate["score"] * 100
                st.write(f"- Match Percentage: {percent:.1f}%")
            st.write(f"- Extracted Skills: {', '.join([skill.capitalize() for skill in candidate['skills']])}")

st.markdown("---")
st.write("Made with using Streamlit & spaCy")
