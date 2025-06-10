import streamlit as st
import pandas as pd
import base64
import random
import time
import datetime
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams, LTTextBox
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
import io
import requests
import json
import re
import os
from streamlit_tags import st_tags
from PIL import Image
import pymysql
from Courses import resume_videos, interview_videos
import plotly.express as px
import nltk
import spacy
import yt_dlp
from yt_dlp import YoutubeDL
from resume_builder import ResumeBuilder
from streamlit_lottie import st_lottie
from pdf2image import convert_from_path
import pytesseract
import cv2
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Initialize resume builder and NLP
resume_builder = ResumeBuilder()
nlp = spacy.load('en_core_web_sm')
print("NLpSwakshan ", nlp.pipe_names)

nltk.download('stopwords')
os.environ["PAFY_BACKEND"] = "yt-dlp"
# import pafy

def load_lottie(url_or_path: str, sidebar: bool = False):
    if url_or_path.startswith(("http://", "https://")):
        response = requests.get(url_or_path)
        if response.status_code != 200:
            st.error(f"Failed to load Lottie animation from URL: {url_or_path}")
            return None
        lottie_json = response.json()
    else:
        try:
            with open(url_or_path, "r") as f:
                lottie_json = json.load(f)
        except Exception as e:
            st.error(f"Failed to load Lottie animation from file: {url_or_path}. Error: {e}")
            return None

    if sidebar:
        with st.sidebar:
            st_lottie(lottie_json, height=200, key="sidebar_lottie")
    else:
        st_lottie(lottie_json, height=300, key="main_lottie")

def fetch_yt_video(link):
    ydl_opts = {}
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(link, download=False)
        return info['title']

# Lightcast API credentials
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
TOKEN_URL = "https://auth.emsicloud.com/connect/token"

token_cache = {
    "access_token": None,
    "expiry_time": 0
}

def get_access_token():
    global token_cache
    if token_cache["access_token"] and time.time() < token_cache["expiry_time"]:
        return token_cache["access_token"]
    
    payload = {
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "grant_type": "client_credentials",
        "scope": "emsi_open"
    }
    
    response = requests.post(TOKEN_URL, data=payload)
    if response.status_code == 200:
        token_data = response.json()
        access_token = token_data.get("access_token")
        token_cache["access_token"] = access_token
        token_cache["expiry_time"] = time.time() + 3600
        return access_token
    else:
        st.error("Failed to get access token. Status code: " + str(response.status_code))
        st.error("Response: " + response.text)
        return None

def get_table_download_link(df, filename, text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def extract_text_from_scanned_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        extracted_text = ""
        for image in images:
            img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, thresh1 = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)
            rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 18))
            dilation = cv2.dilate(thresh1, rect_kernel, iterations=1)
            contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cropped = img[y:y + h, x:x + w]
                text = pytesseract.image_to_string(cropped, lang='eng')
                extracted_text += text + "\n"
        return extracted_text
    except Exception as e:
        st.error(f"Error extracting text from scanned PDF: {e}")
        return None

def pdf_reader(file):
    try:
        resource_manager = PDFResourceManager()
        fake_file_handle = io.StringIO()
        converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
        page_interpreter = PDFPageInterpreter(resource_manager, converter)
        
        with open(file, 'rb') as fh:
            for page in PDFPage.get_pages(fh, caching=True, check_extractable=True):
                page_interpreter.process_page(page)
            text = fake_file_handle.getvalue()

        if not text.strip():
            st.warning("No text found. Trying OCR...")
            with st.spinner("Extracting text using OCR..."):
                ocr_text = extract_text_from_scanned_pdf(file)
                if ocr_text:
                    text = ocr_text
                else:
                    st.error("Failed to extract text using OCR.")
                    return None
        
        converter.close()
        fake_file_handle.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return None

def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)

def course_recommender(course_list):
    st.subheader("**Courses & Certificates Recommendations 🎓**")
    c = 0
    rec_course = []
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5, key=f"slider_{random.randint(0, 100000)}")
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course

def fetch_related_skills(skill, access_token):
    SKILLS_API_URL = "https://emsiservices.com/skills/versions/latest/skills"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    params = {
        "q": skill,
        "limit": 3
    }
    try:
        response = requests.get(SKILLS_API_URL, headers=headers, params=params)
        if response.status_code == 200:
            data = response.json()
            related_skills = [skill['name'] for skill in data.get('data', [])]
            return related_skills
        else:
            st.error(f"Failed to fetch data from API. Status code: {response.status_code}")
            st.error("Response: " + response.text)
            return []
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return []

def extract_name_from_email(email):
    if not email:
        return None
    local_part = email.split('@')[0] if '@' in email else email
    name_parts = re.split(r'[._0-9]+', local_part)
    name = ' '.join([part.capitalize() for part in name_parts if part])
    return name if name else None

def extract_experience(resume_text):
    experience_section = []
    total_experience_months = 0
    experience_patterns = [
        r'(?i)(?:experience|work history|professional experience|employment history|work experience)',
        r'(?i)(?:jobs?|positions?|roles?) held',
        r'(?i)(?:professional|work|employment) background'
    ]
    exp_section_text = ""
    for pattern in experience_patterns:
        match = re.search(pattern, resume_text)
        if match:
            start = match.start()
            end = re.search(r'\n\s*\n|\b(?:education|skills|projects)\b', resume_text[start:], re.I)
            if end:
                exp_section_text = resume_text[start:start+end.start()]
            else:
                exp_section_text = resume_text[start:]
            break
    
    if not exp_section_text:
        return [], 0
    
    job_patterns = [
        r'(?i)([\w\s\-\.]+)\s*(?:at|@|in)\s*([\w\s\-\.,&]+)\s*\((\d{1,2}/\d{4}|\w+\s\d{4})\s*-\s*(\d{1,2}/\d{4}|\w+\s\d{4}|present|current)\)',
        r'(?i)([\w\s\-\.,&]+)\s*[-–]\s*([\w\s\-\.]+)\s*\((\d{1,2}/\d{4}|\w+\s\d{4})\s*-\s*(\d{1,2}/\d{4}|\w+\s\d{4}|present|current)\)',
        r'(?i)^\s*[-•*]\s*([\w\s\-\.]+)\s*(?:at|@|in)\s*([\w\s\-\.,&]+)\s*,\s*(\d{1,2}/\d{4}|\w+\s\d{4})\s*-\s*(\d{1,2}/\d{4}|\w+\s\d{4}|present|current)',
    ]
    
    for pattern in job_patterns:
        for match in re.finditer(pattern, exp_section_text):
            job_title, company, start_date, end_date = match.groups()
            try:
                if len(start_date.split('/')) == 2:
                    start_date = datetime.datetime.strptime(start_date, '%m/%Y')
                else:
                    start_date = datetime.datetime.strptime(start_date, '%B %Y')
                
                if end_date.lower() in ['present', 'current']:
                    end_date = datetime.datetime.now()
                elif len(end_date.split('/')) == 2:
                    end_date = datetime.datetime.strptime(end_date, '%m/%Y')
                else:
                    end_date = datetime.datetime.strptime(end_date, '%B %Y')
                
                duration = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
                total_experience_months += duration
                
                experience_section.append({
                    'job_title': job_title.strip(),
                    'company': company.strip(),
                    'start_date': start_date.strftime('%b %Y'),
                    'end_date': end_date.strftime('%b %Y'),
                    'duration_months': duration,
                    'duration_years': round(duration / 12, 1)
                })
            except ValueError as e:
                continue
    
    total_experience_years = round(total_experience_months / 12, 1)
    return experience_section, total_experience_years

def determine_experience_level(total_experience_years):
    if total_experience_years == 0:
        return "Fresher"
    elif total_experience_years <= 3:
        return "Intermediate"
    elif total_experience_years <= 7:
        return "Experienced"
    else:
        return "Senior"

def fetch_youtube_courses(query, max_results=1):
    ydl_opts = {
        'extract_flat': True,
        'quiet': True,
        'default_search': 'ytsearch',
    }
    with YoutubeDL(ydl_opts) as ydl:
        try:
            search_results = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            courses = []
            for entry in search_results['entries']:
                title = entry.get('title', 'No Title')
                url = entry.get('url', '')
                courses.append([title, url])
            return courses
        except Exception as e:
            st.error(f"Failed to fetch YouTube courses: {e}")
            return []

def detect_sections(resume_text):
    doc = nlp(resume_text)
    sections = {
        'Objective': ['objective', 'summary', 'career goal'],
        'Skills': ['skills', 'technical skills', 'proficiencies'],
        'Experience': ['experience', 'work history', 'employment'],
        'Education': ['education', 'academic'],
        'Projects': ['projects', 'portfolio'],
        'Achievements': ['achievements', 'awards', 'honors'],
        'Certifications': ['certifications', 'certificates', 'credentials']
    }
    section_scores = {}
    for section, keywords in sections.items():
        section_scores[section] = any(keyword.lower() in resume_text.lower() for keyword in keywords)
    return section_scores

# Hardcode your Groq API key
# os.environ['GROQ_API_KEY']
GROQ_API_KEY = os.environ['GROQ_API_KEY']

# Updated function to extract resume details using Groq API
def extract_resume_info_with_groq(resume_text, groq_api_key):
    if not groq_api_key:
        st.error("🔑 Groq API key is missing!")
        return None

    # Clean resume text to avoid JSON issues
    safe_resume_text = resume_text.replace('\\', '\\\\').replace('"', '\\"')

    # Improved prompt with explicit JSON instructions and example
    prompt = (
        "**Task**: Extract the following information from the resume text and return it in valid JSON format.\n"
        "**Fields to Extract**:\n"
        "1. **Name**: Full name of the candidate.\n"
        "2. **Email**: Email address.\n"
        "3. **Contact**: Phone number (include country code if present).\n"
        "4. **Experience**: List of work experiences, each with job_title, company, start_date, end_date, and duration (in months).\n"
        "5. **Total Experience Years**: Total years of experience (rounded to 1 decimal place).\n"
        "6. **Experience Level**: Classify as 'Fresher' (0 years), 'Intermediate' (<=3 years), 'Experienced' (<=7 years), or 'Senior' (>7 years).\n\n"
        "**Instructions**:\n"
        "- Return *only* a valid JSON object. Do not include any additional text, explanations, or markdown (e.g., ```json).\n"
        "- Use null for missing fields.\n"
        "- For dates, use 'MM/YYYY' format or 'Present' for current roles.\n"
        "- Calculate duration in months for each experience and sum for total experience years.\n"
        "- Example output:\n"
        "{\n"
        "  \"Name\": \"John Doe\",\n"
        "  \"Email\": \"john.doe@example.com\",\n"
        "  \"Contact\": \"+1-555-123-4567\",\n"
        "  \"Experience\": [\n"
        "    {\"job_title\": \"Software Engineer\", \"company\": \"Tech Corp\", \"start_date\": \"01/2020\", \"end_date\": \"Present\", \"duration\": 64},\n"
        "    {\"job_title\": \"Junior Developer\", \"company\": \"Startup Inc\", \"start_date\": \"06/2018\", \"end_date\": \"12/2019\", \"duration\": 18}\n"
        "  ],\n"
        "  \"Total Experience Years\": 6.8,\n"
        "  \"Experience Level\": \"Experienced\"\n"
        "}\n\n"
        f"**Resume Text**:\n{safe_resume_text[:50000]}"
    )

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.2,
        "max_tokens": 2000
    }

    try:
        with st.spinner("🔍 Extracting resume details"):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
        if response.status_code == 200:
            response_content = response.json()["choices"][0]["message"]["content"]
            # Try to extract JSON if wrapped in markdown or extra text
            json_match = re.search(r'\{[\s\S]*\}', response_content)
            if json_match:
                response_content = json_match.group(0)
            try:
                extracted_data = json.loads(response_content)
                return extracted_data
            except json.JSONDecodeError as e:
                st.error(f"❌ Failed to parse Groq response as JSON: {e}")
                st.write("Raw response:", response_content)
                return None
        else:
            st.warning(f"❌ Primary model failed: {response.status_code} - {response.text}. Trying fallback model...")
            payload["model"] = "llama3-70b-8192"
            with st.spinner("🔍 Extracting with Llama3-70B-8192 (fallback)..."):
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
            if response.status_code == 200:
                response_content = response.json()["choices"][0]["message"]["content"]
                json_match = re.search(r'\{[\s\S]*\}', response_content)
                if json_match:
                    response_content = json_match.group(0)
                try:
                    extracted_data = json.loads(response_content)
                    return extracted_data
                except json.JSONDecodeError as e:
                    st.error(f"❌ Failed to parse fallback Groq response as JSON: {e}")
                    st.write("Raw response:", response_content)
                    return None
            else:
                st.error(f"❌ Fallback model failed: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        st.error(f"❌ Failed to call Groq API: {e}")
        return None

# Groq ATS Scoring Function with Fallback
def analyze_with_groq(resume_text, groq_api_key):
    if not groq_api_key:
        st.error("🔑 Groq API key is missing!")
        return None

    safe_resume_text = resume_text.replace('\\', '\\\\').replace('"', '\\"')
    prompt = (
        "**Task:** Analyze this resume for ATS compatibility and provide:\n"
        "1. **ATS Score (0-100)** - Start with: \"ATS Score: XX/100\"\n"
        "2. **Key Strengths** (3 bullet points)\n"
        "3. **Weaknesses** (3 bullet points)\n"
        "4. **Top 3 Optimization Tips** (short bullets)\n\n"
        f"**Resume Text:**\n{safe_resume_text[:50000]}"
    )

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.3,
        "max_tokens": 2000
    }

    try:
        with st.spinner("🔍 Analyzing with Llama-3.3-70B-Versatile..."):
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        else:
            st.warning(f"❌ Primary model failed: {response.status_code} - {response.text}. Trying fallback model...")
            payload["model"] = "llama3-70b-8192"
            with st.spinner("🔍 Analyzing with Llama3-70B-8192 (fallback)..."):
                response = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=30
                )
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                st.error(f"❌ Fallback model failed: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        st.error(f"❌ Failed to call Groq API: {e}")
        return None

# Database Connection (aligned with repository)
connection = pymysql.connect(host='localhost', user='root', password='Dhiraj@123', db='cv')
cursor = connection.cursor()

def insert_data(name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses):
    DB_table_name = 'user_data'
    insert_sql = "INSERT INTO " + DB_table_name + """
    (Name, Email_ID, resume_score, Timestamp, Page_no, Predicted_Field, User_level, Actual_skills, Recommended_skills, Recommended_courses)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    rec_values = (name, email, res_score, timestamp, no_of_pages, reco_field, cand_level, skills, recommended_skills, courses)
    cursor.execute(insert_sql, rec_values)
    connection.commit()
    
st.set_page_config(
    page_title="ResumeForge.ai",
    page_icon='./Logo/Logo/logo2.png',
)

def run():
    github_url = "https://github.com/DhirajWankhede"
    st.sidebar.markdown(f'<a href="{github_url}" target="_blank"><img src="https://cdn-icons-png.flaticon.com/512/25/25231.png" width="40" height="40" alt="GitHub Logo"></a>', unsafe_allow_html=True)
    load_lottie("https://assets10.lottiefiles.com/packages/lf20_3rwasyjy.json")
    load_lottie("https://assets10.lottiefiles.com/packages/lf20_qp1q7mct.json", sidebar=True)
    st.title("ResumeForge.ai")

    activities = ["Resume Analyzer", "Developer", "Resume Builder"]
    choice = st.sidebar.segmented_control('Choose the Mode', options=activities)
    link = '[©Developed With ♥️](https://www.linkedin.com/in/swakshan-tayade-3a0a5b235/)'
    st.sidebar.markdown(link, unsafe_allow_html=True)

    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)

    DB_table_name = 'user_data'
    table_sql = """
        CREATE TABLE IF NOT EXISTS user_data (
            ID INT NOT NULL AUTO_INCREMENT,
            Name VARCHAR(100) NOT NULL,
            Email_ID VARCHAR(50) NOT NULL,
            resume_score VARCHAR(8) NOT NULL,
            Timestamp VARCHAR(50) NOT NULL,
            Page_no VARCHAR(5) NOT NULL,
            Predicted_Field VARCHAR(100) NOT NULL,
            User_level VARCHAR(30) NOT NULL,
            Actual_skills VARCHAR(300) NOT NULL,
            Recommended_skills VARCHAR(300) NOT NULL,
            Recommended_courses VARCHAR(600) NOT NULL,
            PRIMARY KEY (ID)
        );
    """
    cursor.execute(table_sql)

    if choice == 'Resume Analyzer':
        st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload your resume, and get smart recommendations</h5>''',
                    unsafe_allow_html=True)
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        
        if pdf_file is not None:
            with st.spinner('Uploading your Resume...'):
                time.sleep(4)
            save_image_path = './Uploaded_Resumes/'+pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            resume_text = pdf_reader(save_image_path)
            
            if resume_data and resume_text:
                st.text_area("Extracted Resume Text:", resume_text, height=300)
                st.header("**Resume Analysis**")

                # Extract details using Groq API
                groq_extracted_data = extract_resume_info_with_groq(resume_text, GROQ_API_KEY)
                if groq_extracted_data:
                    # Update resume_data with Groq-extracted fields
                    resume_data['name'] = groq_extracted_data.get('Name', resume_data.get('name', extract_name_from_email(resume_data.get('email'))))
                    resume_data['email'] = groq_extracted_data.get('Email', resume_data.get('email', 'Not provided'))
                    resume_data['mobile_number'] = groq_extracted_data.get('Contact', resume_data.get('mobile_number', 'Not provided'))
                    resume_data['experience'] = groq_extracted_data.get('Experience', [])
                    resume_data['total_experience_years'] = groq_extracted_data.get('Total Experience Years', 0)
                    resume_data['experience_level'] = groq_extracted_data.get('Experience Level', 'Fresher')

                    # Store in session state
                    st.session_state['resume_data'] = resume_data

                    # Display extracted information
                    st.subheader("**Your Basic Info**")
                    st.text(f"Name: {resume_data['name'] or 'Not provided'}")
                    st.text(f"Email: {resume_data['email'] or 'Not provided'}")
                    st.text(f"Contact: {resume_data['mobile_number'] or 'Not provided'}")
                    st.text(f"Resume pages: {str(resume_data['no_of_pages'] or 'Not provided')}")
                    st.text(f"Total Experience: {resume_data['total_experience_years']} years")
                    st.text(f"Experience Level: {resume_data['experience_level']}")

                    if resume_data['experience']:
                        st.subheader("**Work Experience Details**")
                        exp_df = pd.DataFrame(resume_data['experience'])
                        st.dataframe(exp_df)

                    # Display greeting
                    if resume_data['name'] and resume_data['name'] != 'Not provided':
                        st.success(f"Hello {resume_data['name']}!")
                    else:
                        st.success("Hello!")

                else:
                    st.error("Failed to extract resume details with Groq API. Falling back to ResumeParser.")
                    # Fallback to existing extraction methods
                    experience_section, total_experience_years = extract_experience(resume_text)
                    resume_data['experience'] = experience_section
                    resume_data['total_experience_years'] = total_experience_years
                    resume_data['experience_level'] = determine_experience_level(total_experience_years)
                    resume_data['name'] = extract_name_from_email(resume_data.get('email')) or 'Not provided'
                    resume_data['email'] = resume_data.get('email', 'Not provided')
                    resume_data['mobile_number'] = resume_data.get('mobile_number', 'Not provided')

                    # Store in session state
                    st.session_state['resume_data'] = resume_data

                    st.subheader("**Your Basic Info**")
                    st.text(f"Name: {resume_data['name']}")
                    st.text(f"Email: {resume_data['email']}")
                    st.text(f"Contact: {resume_data['mobile_number']}")
                    st.text(f"Resume pages: {str(resume_data['no_of_pages'] or 'Not provided')}")
                    st.text(f"Total Experience: {total_experience_years} years")
                    st.text(f"Experience Level: {resume_data['experience_level']}")

                    if experience_section:
                        st.subheader("**Work Experience Details**")
                        exp_df = pd.DataFrame(experience_section)
                        st.dataframe(exp_df)

                    if resume_data['name'] != 'Not provided':
                        st.success(f"Hello {resume_data['name']}!")
                    else:
                        st.success("Hello!")

                # Existing skills and recommendations
                keywords = st_tags(label='### Your Current Skills',
                                   text='See our skills recommendation below',
                                   value=resume_data['skills'], key='1')

                keywords_dict = {
                    'ds_dev': ['tensorflow', 'keras', 'pytorch', 'machine learning', 'deep learning', 'flask', 'streamlit'],
                    'ui_ux_designer': ["UI/UX", "Wireframing", "Prototyping", "User Research", "Adobe XD",
                                      "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'backend_dev': ["Python", "Java", "Node.js", "SQL", "APIs", "Django", "Flask", "Database Design",
                                   "Microservices", "Docker", "Analytical thinking", "Problem-solving", "Team collaboration", "Communication"],
                    'mobile_app_dev': ["Swift", "Kotlin", "React Native", "Flutter", "Mobile UI/UX", "App Store Deployment",
                                     "User-centric thinking", "Problem-solving", "Attention to detail"],
                    'ios_dev': ["Swift", "Objective-C", "Xcode", "iOS SDK", "App Store Deployment", "User-centric thinking",
                               "Problem-solving", "Attention to detail"],
                    'game_dev': ["Unity", "Unreal Engine", "C++", "C#", "3D Graphics", "Game Physics", "Graphics Programming",
                                "Physics Simulation", "Multiplayer", "Creativity", "Problem-solving", "Team collaboration"],
                    'machine_learning': ["Python", "R", "Machine Learning", "Statistics", "SQL", "Deep Learning",
                                       "Analytical thinking", "Research", "Problem-solving"],
                    'data_analyst': ["SQL", "Excel", "Python", "Data Visualization", "Statistics",
                                    "Analytical thinking", "Research", "Problem-solving"],
                    'cloud_engineer': ["AWS", "Azure", "Google Cloud", "DevOps", "Kubernetes", "Docker",
                                     "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'devops_engineer': ["CI/CD", "Docker", "Kubernetes", "Jenkins", "Ansible", "Git",
                                       "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'site_reliability_engineer': ["Linux", "Networking", "Monitoring", "Incident Response", "Automation", "Scripting",
                                                "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'cybersecurity_engineer': ["Firewalls", "Intrusion Detection", "Incident Response", "Encryption", "Penetration Testing",
                                             "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'penetration_tester': ["Penetration Testing", "Ethical Hacking", "Vulnerability Assessment", "Security Tools", "Networking",
                                         "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'network_engineer': ["Networking", "Cisco", "Routing & Switching", "Firewalls", "Troubleshooting",
                                       "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'software_engineer': ["Algorithms", "Data Structures", "Object-Oriented Programming", "Design Patterns", "Testing",
                                        "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'database_admin': ["SQL", "Database Design", "Performance Tuning", "Backup & Recovery", "Security",
                                      "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'business_analyst': ["SQL", "Excel", "Data Analysis", "Requirements Gathering", "Data Visualization",
                                        "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'product_manager': ["Product Development", "Market Research", "Roadmapping", "Stakeholder Management", "Agile",
                                      "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'technical_writer': ["Technical Writing", "Documentation", "Editing", "Proofreading", "Research",
                                        "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'it_support': ["Troubleshooting", "Help Desk", "Networking", "Hardware", "Software",
                                  "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'data_engineer': ["ETL", "Data Warehousing", "Big Data", "SQL", "Python",
                                     "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'ai_engineer': ["Machine Learning", "Deep Learning", "Neural Networks", "Python", "TensorFlow",
                                   "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'blockchain_engineer': ["Blockchain", "Smart Contracts", "Cryptocurrency", "DApps", "Solidity",
                                           "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'qa_engineer': ["Testing", "Automation", "Selenium", "JIRA", "Bug Tracking",
                                   "Analytical thinking", "Problem-solving", "Team collaboration"],
                    'system_admin': ["Linux", "Windows", "Networking", "Security", "Troubleshooting",
                                    "Analytical thinking", "Problem-solving", "Team collaboration"]
                }

                recommended_skills = []
                reco_field = ''
                rec_course = ''
                all_recommended_skills = []
                category_match_count = {cat: 0 for cat in keywords_dict.keys()}

                for i in resume_data['skills']:
                    for cat, skills in keywords_dict.items():
                        if i.lower() in [s.lower() for s in skills]:
                            category_match_count[cat] += 1

                max_matches = max(category_match_count.values())
                if max_matches > 0:
                    category = max(category_match_count, key=category_match_count.get)
                    st.success(f"** Our analysis says you are looking for {category.replace('_', ' ').title()} Jobs **")
                    for i in resume_data['skills']:
                        if i.lower() in [s.lower() for s in keywords_dict[category]]:
                            access_token = get_access_token()
                            if access_token:
                                related_skills = fetch_related_skills(i.lower(), access_token)
                            else:
                                related_skills = []
                                st.error("Failed to fetch related skills from the API.")
                            recommended_skills = related_skills
                            all_recommended_skills.extend(recommended_skills)

                if all_recommended_skills:
                    unique_key = f"all_recommended_skills"
                    recommended_keywords = st_tags(label='### Recommended skills for you.',
                                                  text='Recommended skills generated from System',
                                                  value=all_recommended_skills, key=unique_key)
                    st.markdown('''<h4 style='text-align: left; color: #1ed760;'>Adding these skills to your resume will boost🚀 your chances of getting a Job</h4>''',
                                unsafe_allow_html=True)
                else:
                    st.warning("No recommended skills found for this job role.")

                # Fetch YouTube course recommendations
                with st.spinner('Fetching YouTube course recommendations...'):
                    youtube_courses = []
                    max_videos = 5
                    video_count = 0
                    for skill in all_recommended_skills[:5]:
                        if video_count >= max_videos:
                            break
                        query = f"{skill} course"
                        courses = fetch_youtube_courses(query, max_results=1)
                        if courses:
                            youtube_courses.extend(courses)
                            video_count += 1
                    rec_course = str(youtube_courses)

                if youtube_courses:
                    st.subheader(f"**YouTube Course Recommendations 🎓**")
                    for title, url in youtube_courses:
                        st.markdown(f"- [{title}]({url})")
                else:
                    st.warning("No YouTube courses found for the recommended skills.")

                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)

                st.subheader("**Resume Tips & Ideas💡**")
                section_scores = detect_sections(resume_text)
                for section, present in section_scores.items():
                    if present:
                        st.markdown(f'''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added {section}</h5>''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''<h5 style='text-align: left; color: red;'>[-] Please add {section} to improve your resume.</h5>''', unsafe_allow_html=True)

                # ATS Scoring
                st.subheader("**ATS Resume Evaluation**")
                analysis = analyze_with_groq(resume_text, GROQ_API_KEY)
                if analysis:
                    st.success("✅ Analysis Complete!")
                    st.markdown("---")
                    
                    if "ATS Score:" in analysis:
                        score_line = analysis.split("ATS Score:")[1].split("\n")[0].strip()
                        score = int(score_line.split("/")[0].strip())
                        score = min(max(score, 0), 100)
                        st.progress(score / 100.0)
                        st.success(f'**ATS Score: {score}/100**')
                        if score >= 80:
                            st.success("🎉 **Excellent!** Highly ATS-optimized.")
                        elif score >= 60:
                            st.warning("⚠️ **Good, but needs minor improvements.**")
                        else:
                            st.error("❌ **Needs major optimization.**")

                    strengths_section = re.search(r'\*\*Key Strengths\*\*.*?((?:- .+?\n){3})', analysis, re.DOTALL)
                    if strengths_section:
                        strengths = strengths_section.group(1).strip().split('\n')
                        st.subheader("**Key Strengths**")
                        for strength in strengths:
                            st.markdown(strength)

                    weaknesses_section = re.search(r'\*\*Weaknesses\*\*.*?((?:- .+?\n){3})', analysis, re.DOTALL)
                    if weaknesses_section:
                        weaknesses = weaknesses_section.group(1).strip().split('\n')
                        st.subheader("**Weaknesses**")
                        for weakness in weaknesses:
                            st.markdown(weakness)

                    tips_section = re.search(r'\*\*Top 3 Optimization Tips\*\*.*?((?:- .+?\n){3})', analysis, re.DOTALL)
                    if tips_section:
                        tips = tips_section.group(1).strip().split('\n')
                        st.subheader("**Top 3 Optimization Tips**")
                        for tip in tips:
                            st.markdown(tip)

                    st.subheader("**Full ATS Resume Evaluation**")
                    st.markdown(analysis)

                # Insert data into database (aligned with repository schema)
                insert_data(
                    resume_data['name'],
                    resume_data['email'],
                    str(score if 'score' in locals() else 0),
                    timestamp,
                    str(resume_data['no_of_pages'] or '1'),
                    reco_field,
                    resume_data['experience_level'],
                    str(resume_data['skills']),
                    str(all_recommended_skills),
                    rec_course
                )

                st.header("**Bonus Video for Resume Writing Tips💡**")
                resume_vid = random.choice(resume_videos)
                res_vid_title = fetch_yt_video(resume_vid)
                st.subheader("✅ **"+res_vid_title+"**")
                st.video(resume_vid)

                st.header("**Bonus Video for Interview Tips💡**")
                interview_vid = random.choice(interview_videos)
                int_vid_title = fetch_yt_video(interview_vid)
                st.subheader("✅ **" + int_vid_title + "**")
                st.video(interview_vid)

                connection.commit()
            else:
                st.error('Something went wrong..')
    elif choice == 'Resume Builder':
        if st.sidebar.button("📝 Build Your Resume"):
            st.session_state['resume_builder'] = True
            
        if 'resume_builder' in st.session_state and st.session_state['resume_builder']:
            st.header("Lets build a great resume")
            template = st.selectbox("Choose a Template", ["Modern", "Professional", "Minimal Creative"])
            personal_info = {}
            personal_info['full_name'] = st.text_input("Full Name")
            personal_info['email'] = st.text_input("Email")
            personal_info['phone'] = st.text_input("Phone")
            personal_info['location'] = st.text_input("Location")
            personal_info['linkedin'] = st.text_input("LinkedIn Profile")
            personal_info['portfolio'] = st.text_input("Portfolio Website")
            personal_info['title'] = st.text_input("Professional Title")
            
            summary = st.text_area("Professional Summary")
            
            experience = []
            st.subheader("Experience")
            num_experience = st.number_input("Number of Experience Entries", min_value=0, max_value=10, value=1)
            
            for i in range(num_experience):
                exp = {}
                exp['position'] = st.text_input(f"Position {i+1}")
                exp['company'] = st.text_input(f"Company {i+1}")
                exp['start_date'] = st.text_input(f"Start Date {i+1}")
                exp['end_date'] = st.text_input(f"End Date {i+1}")
                exp['description'] = st.text_area(f"Description {i+1}")
                experience.append(exp)
                
            education = []
            st.header("Education")
            num_education = st.number_input("Education", min_value=1, max_value=10, value=1)
            
            for i in range(num_education):
                edu = {}
                edu['school'] = st.text_input(f"School {i+1}")
                edu['degree'] = st.text_input(f"Degree {i+1}")
                edu['field'] = st.text_input(f"Field of Study {i+1}")
                edu['graduation_date'] = st.text_input(f"Graduation Date {i+1}")
                edu['gpa'] = st.text_input(f"GPA {i+1}")
                education.append(edu)
            
            skills = {}
            st.subheader("Skills")
            skills['technical'] = st.text_area("Technical Skills (comma separated)")
            skills['soft'] = st.text_area("Soft Skills (comma separated)")
            skills['languages'] = st.text_area("Languages (comma separated)")
            skills['tools'] = st.text_area("Tools & Technologies (comma separated)")
            
            resume_data = None
            if st.button("Generate Resume"):
                resume_data = {
                    'personal_info': personal_info,
                    'summary': summary,
                    'experience': experience,
                    'education': education,
                    'skills': skills,
                    'template': template
                }
                
            if resume_data is not None:
                resume_buffer = resume_builder.generate_resume(resume_data)
                st.success("Resume generated successfully")
                st.download_button(
                    label="Download Resume",
                    data=resume_buffer,
                    file_name="resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
    else:
        st.success('Welcome to Admin Side (FYP_G3)')
        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')
        if st.button('Login'):
            if ad_user == 'FYP_G3' and ad_password == '123':
                st.success("Welcome FYP_G3 !")
                cursor.execute('''SELECT * FROM user_data''')
                data = cursor.fetchall()
                decoded_data = []
                for row in data:
                    decoded_row = list(row)
                    for i in range(7, 10):  # Adjust indices for Actual_skills, Recommended_skills, Recommended_courses
                        if isinstance(decoded_row[i], bytes):
                            decoded_row[i] = decoded_row[i].decode('utf-8')
                    decoded_data.append(decoded_row)
                
                df = pd.DataFrame(decoded_data, columns=[
                    'ID', 'Name', 'Email', 'Resume Score', 'Timestamp', 'Total Page',
                    'Predicted Field', 'User Level', 'Actual Skills', 'Recommended Skills',
                    'Recommended Course'
                ])
                
                st.header("**User's Data**")
                st.dataframe(df)
                st.markdown(get_table_download_link(df, 'User_Data.csv', 'Download Report'), unsafe_allow_html=True)

                st.subheader("**Pie-Chart for Predicted Field Recommendation**")
                labels = df['Predicted Field'].unique()
                values = df['Predicted Field'].value_counts()
                fig = px.pie(df, values=values, names=labels, title='Predicted Field according to the Skills')
                st.plotly_chart(fig)

                st.subheader("**Pie-Chart for User's Experienced Level**")
                labels = df['User Level'].unique()
                values = df['User Level'].value_counts()
                fig = px.pie(df, values=values, names=labels, title="Pie-Chart📈 for User's👨‍💻 Experienced Level")
                st.plotly_chart(fig)
            else:
                st.error("Wrong ID & Password Provided")

run()