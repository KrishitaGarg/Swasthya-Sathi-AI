import streamlit as st
from streamlit_option_menu import option_menu
import google.generativeai as genai
from PIL import Image as PILImage
from PIL import ImageFont
import io
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import feedparser
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from reportlab.lib.pagesizes import letter, inch
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
import base64
import numpy as np
import joblib
import re

# Load environment variables
load_dotenv()

# Set the font path
font_path = os.path.join("fonts", "ARIAL.TTF")

# Load the font
font = ImageFont.truetype(font_path, 24)

# Function to convert local image to Base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Convert the icon to base64
icon_base64 = get_base64_image("assets/icon.jpeg")

# Configure the page
st.set_page_config(
    page_title="Swasthya Sathi AI", 
    page_icon=f"data:image/jpeg;base64,{icon_base64}",
    layout="wide",
    initial_sidebar_state="expanded", 
)

# Custom CSS for better styling
st.markdown(
    """
    <style>
    body {
        background-color: #f4f1ea;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        padding: 20px;
        border-radius: 10px;
        background-color: #f4f1ea;
    }
    .stButton>button {
        background-color: #4a2618;
        color: white;
        border: none;
        padding: 6px 24px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 8px;
        transition: background 0.3s ease, transform 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #8b5e34;
        color: white;
        transform: scale(1.05);
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #8b5e34;
        padding: 12px;
        font-size: 16px;
        width: 100%;
        text-align: center;
    }
    .stTextInput>div {
        display: flex;
        justify-content: center;
        margin-top: 50px;
    }
    .stSidebar > div {
        background-color: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 8px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }
    .sidebar-emoji {
        text-align: center;
        margin-bottom: 15px;
    }
    .sidebar-emoji img {
        width: 80px;
        height: 80px;
    }
    .chat-message {
        font-size: 16px;
        font-weight: bold;
        color: white;
        background-color: #4a2618;
        padding: 10px;
        border-radius: 8px;
    }
    .sidebar-image {
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 80%;
    }
    .stFileUploader label {
        background-color: #a48173 !important;
        color: black !important;
        padding: 10px 20px;
        border-radius: 8px;
        font-size: 16px;
        font-weight: bold;
        text-align: center;
        transition: all 0.3s ease;
        display: inline-block;
        margin-bottom: 10px;
    }
    .stFileUploader label:hover {
        background-color: #8b5e34 !important;
        color: white !important;
        transform: scale(1.005);
    }
    .stTextArea label {
        background-color: #a48173 !important;
        color: black !important;
        font-size: 16px !important;
        padding: 8px 15px !important;
        border-radius: 8px !important;
        display: inline-block;
        transition: all 0.3s ease;
        margin-bottom: 10px;
        margin-top: 10px;
    }
    .stTextArea label:hover {
        background-color: #8b5e34 !important;
        transform: scale(1.05);
        color: white !important;
    }
    div.stForm div.stFormSubmitButton > button {
        background-color: #4a2618 !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 6px 24px !important;
        border-radius: 8px !important;
        border: none !important;
        cursor: pointer !important;
        transition: all 0.3s ease-in-out;
    }
    div.stForm div.stFormSubmitButton > button:hover {
        background-color: #8b5e34 !important;
        color: white !important;
        transform: scale(1.05);
    }    
    </style>
    """,
    unsafe_allow_html=True
)

# Display the logo in sidebar
st.sidebar.image("assets/icon.jpeg", use_container_width=True)

# Navigation menu
selected = option_menu(
    menu_title="Smart AI Medical Assistant", 
    options=["AI-Assisted Images Analysis", "Smart Medical Transcriber", "AI-Powered Lab Report Analyzer", "AI Medical Coding", "Health Risk & Insurance Evaluator", "Personalized Treatment & Diet Planner", "AI-Powered Medical Assistance"],
    icons=["activity", "file-text", "file-medical", "file-code", "shield-plus", "heart-pulse", "robot"], 
    orientation="horizontal",
    styles={
        "container": {"padding": "15px 0!important", "background-color": "#d8c3a5"},
        "icon": {"color": "#00000", "font-size": "20px"},
        "nav-link": {"font-size": "15px", "font-family": "serif", "text-align": "center", "margin":"0px", "--hover-color": "#fadfd4"},
        "nav-link-selected": {"background-color": "#4a2618"},}
)

@st.cache_resource # Cache the model to avoid loading it multiple times

# Function to load the Gemini Pro Vision model
def load_model():
    api_key = os.getenv("GOOGLE_API_KEY") # Load the Google API Key
    if not api_key:
        st.error("Google API Key not found in .env file.") # Display an error if the API key is not found
        st.stop()
    genai.configure(api_key=api_key) # Configure the API key
    return genai.GenerativeModel('gemini-1.5-flash') # Load the Gemini Pro Vision model

# Function to analyze image
def analyze_image(image, prompt):
    model = load_model() # Load the model
    response = model.generate_content([prompt, image]) # Generate content using the model
    return response.text # Return the generated content

# Function to search for research papers
def search_research_papers(query):
    search_url = f"https://scholar.google.com/scholar?q={query}" # Create the search URL
    response = requests.get(search_url) # Send a GET request
    soup = BeautifulSoup(response.content, 'html.parser') # Parse the HTML content
    papers = [{'title': item.select_one('.gs_rt').text, 'link': item.select_one('.gs_rt a')['href']} for item in soup.select('[data-lid]')] # Extract the paper title and link
    return papers

# Function to fetch and parse RSS feed
def fetch_rss_feed(feed_url):
    feed = feedparser.parse(feed_url) # Parse the RSS feed
    if feed.bozo: # Check if the feed is valid
        st.error("Failed to fetch RSS feed.")
        return []
    # Extract article information
    articles = [{'title': entry.title, 'link': entry.link, 'published': entry.get('published', 'No publication date')} for entry in feed.entries] 
    return articles

# Function to create a pathology report with matplotlib
def create_pathology_report(patient_info, service_info, specimens, theranostic_report):
    fig, ax = plt.subplots(figsize=(10, 12)) # Create a plot and axis

    # Function to add a rectangle with text inside
    def add_textbox(ax, x, y, width, height, header, text, wrap_text=True, fontsize=9, fontweight='normal', ha='left', va='top', line_height=0.02, color='white'):
        rect = patches.Rectangle((x, y), width, height, linewidth=1.5, edgecolor='black', facecolor=color) # Create a rectangle
        ax.add_patch(rect) # Add the rectangle to the plot
        plt.text(x + 0.01, y + height - 0.01, header, ha=ha, va=va, fontsize=fontsize, fontweight='bold', family='DejaVu Sans')
    
        # Function to wrap text and add it to the plot
        if wrap_text:
            words = text.split()
            lines = []
            current_line = ""

            for word in words:
                if len(current_line + word) * 0.01 > width: # Check if the line exceeds the width
                    lines.append(current_line) # Add the line to the list
                    current_line = word + " " # Start a new line
                else:
                    current_line += word + " " # Add the word to the current line

            if current_line:
                lines.append(current_line) # Add the last line

            for i, line in enumerate(lines):
                if i * line_height < height - line_height: # Check if the line exceeds the height
                    plt.text(x + 0.01, y + height - 0.03 - i * line_height, line, ha=ha, va=va, fontsize=fontsize, fontweight=fontweight, family='DejaVu Sans', clip_on=True) # Add the line to the plot
        else:
            plt.text(x + 0.01, y + height - 0.03, text, ha=ha, va=va, fontsize=fontsize, fontweight=fontweight, family='DejaVu Sans', clip_on=True) # Add the text to the plot

    # Add the main header
    plt.text(0.5, 0.96, 'LABORATORY MEDICINE PROGRAM', ha='center', va='center', fontsize=15, family='DejaVu Sans', fontweight='bold')

    # Add the subheader
    plt.text(0.5, 0.93, 'Surgical Pathology Consultation Report', ha='center', va='center', fontsize=13, family='DejaVu Sans', fontweight='bold')

    # Define the increased height for each section
    section_height = 0.8 / 4

    # Add Patient Information box without wrapping text
    add_textbox(ax, 0.05, 0.88 - section_height, 0.9, section_height, 'Patient Information', patient_info, wrap_text=False, fontsize=10, line_height=0.025, color='#E6F2FF')

    # Add Service Information box with wrapping text
    add_textbox(ax, 0.05, 0.88 - 2*section_height, 0.9, section_height, 'Observation', service_info, wrap_text=True, fontsize=10, line_height=0.025, color='#F5F5F5')

    # Add Specimen(s) Received box with wrapping text
    add_textbox(ax, 0.05, 0.88 - 3*section_height, 0.9, section_height, 'Inferences', specimens, wrap_text=True, fontsize=10, line_height=0.025, color='#E6F2FF')

    # Add Consolidated Theranostic Report section with wrapping text
    add_textbox(ax, 0.05, 0.88 - 4*section_height, 0.9, section_height, 'Conclusion', theranostic_report, wrap_text=True, fontsize=10, line_height=0.025, color='#F5F5F5')

    # Set the axis limits and hide the axes
    ax.set_xlim(0, 1) 
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Save the plot to a buffer
    buf = io.BytesIO() # Create a buffer to hold the image
    plt.savefig(buf, format='png') # Save the plot to the buffer
    buf.seek(0) # Reset the buffer position to the start
    plt.close(fig) # Close the plot to free up memory
    return buf # Return the buffer

# Function to create a PDF report
def convert_markdown_to_html(text):
    """Convert basic Markdown to HTML for better PDF rendering."""
    text = re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)  # Bold
    text = re.sub(r"\*(.*?)\*", r"<i>\1</i>", text)  # Italics
    text = re.sub(r"(?m)^- (.*?)$", r"• \1", text)  # Bullet points
    text = re.sub(r"\n", r"<br/>", text)  # Newlines
    return text

def create_pdf_report(patient_info, service_info, specimens, theranostic_report, diagnosis, detailed_diagnosis, image_buffer, report_format):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Register Arial font
    pdfmetrics.registerFont(TTFont('Arial', 'fonts/ARIAL.TTF'))

    # Define styles
    styles = getSampleStyleSheet()
    styleN = ParagraphStyle('Normal', fontName='Arial', fontSize=10, leading=14)
    styleH = ParagraphStyle('Heading1', fontName='Arial', fontSize=16, leading=20, alignment=1, spaceAfter=12, underline=True)
    styleH2 = ParagraphStyle('Heading2', fontName='Arial', fontSize=14, leading=16, spaceAfter=8, underline=True)

    # Header format
    format_details = {
        "Format 1": {"color": colors.black, "header": "Swasthya Sathi AI Medical Report"}
    }
    format_detail = format_details.get(report_format, format_details["Format 1"])

    # Logo and header layout using table
    try:
        logo_path = "assets/icon.jpeg"
        logo = Image(logo_path, width=1.2 * inch, height=1.2 * inch)
    except Exception:
        logo = Paragraph("<b>Logo Not Found</b>", styleN)

    header_table = Table(
        [[Paragraph(f"<b>{format_detail['header']}</b>", styleH), logo]],
        colWidths=[380, 120]  # Adjusted for better right alignment
    )
    header_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (0, 0), 'LEFT'),   # Aligns header text to the left
        ('ALIGN', (1, 0), (1, 0), 'RIGHT'),  # Aligns logo to the right
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 12))

    # Patient Info Table
    table_style = TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Arial'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 50),  # Shift table right
    ])

    patient_table = Table([
        [Paragraph("<b>Patient Information:</b>", styleN), patient_info],
        [Paragraph("<b>Observation:</b>", styleN), service_info],
        [Paragraph("<b>Inferences:</b>", styleN), specimens]
    ], colWidths=[160, 400])

    patient_table.setStyle(table_style)
    elements.append(patient_table)
    elements.append(Spacer(1, 12))

    # Diagnosis Section
    elements.append(Paragraph("<b>DIAGNOSIS</b>", styleH2))
    elements.append(Paragraph(convert_markdown_to_html(diagnosis) if diagnosis else "No significant findings.", styleN))
    elements.append(Spacer(1, 12))

    # Conclusion Section
    elements.append(Paragraph("<b>CONCLUSION</b>", styleH2))
    elements.append(Paragraph(convert_markdown_to_html(theranostic_report), styleN))
    elements.append(Spacer(1, 12))

    # X-Ray Image Section
    elements.append(Paragraph("<b>X-Ray Image:</b>", styleH2))
    elements.append(Image(image_buffer, width=5 * inch, height=3.5 * inch))
    elements.append(Spacer(1, 12))

    # Advice Section
    elements.append(Paragraph("<b>ADVICE</b>", styleH2))
    elements.append(Paragraph("<i>Clinical correlation recommended.</i>", styleN))

    # Custom Page Layout (White Background with Black Border)
    def add_background_and_border(canvas, doc):
        margin = 36
        canvas.saveState()
        canvas.setFillColor(colors.white)
        canvas.rect(margin, margin, doc.pagesize[0] - 2 * margin, doc.pagesize[1] - 2 * margin, fill=1)
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(2)
        canvas.rect(margin, margin, doc.pagesize[0] - 2 * margin, doc.pagesize[1] - 2 * margin)
        canvas.restoreState()

    # Build PDF
    doc.build(elements, onFirstPage=add_background_and_border, onLaterPages=add_background_and_border)

    buffer.seek(0)
    return buffer



# Function to display common instructions
def display_instructions(page):
    
    # Custom CSS for the sidebar header
    st.sidebar.markdown(
        """
        <style>
        .sidebar-header {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
        }
        </style>
        <div class="sidebar-header">Instructions</div>
        """,
        unsafe_allow_html=True
    )

    # Instructions for each page
    instructions = {
        "AI-Assisted Images Analysis": """
            1. Upload one or more medical images using the file uploader.
            2. Enter your prompt or use the default one provided.
            3. Click 'Analyze Image' to get the analysis.
            4. If not satisfied with the analysis, click 'Regenerate Analysis'.
            5. View related research papers based on the analysis for further details.
        """,
        "Smart Medical Transcriber": """
            1. Upload a medical prescription image using the file uploader.
            2. Enter your prompt or use the default one provided.
            3. Click 'Get Transcription' to see the analysis in tabular format.
        """,
        "AI-Powered Lab Report Analyzer": """
            1. Upload a medical report image using the file uploader.
            2. Enter your prompt or use the default one provided.
            3. Click 'Analyze Report' to get the analysis and generate the pathology report.
        """,
        "AI Medical Coding": """
            1. Upload a medical document image using the file uploader.
            2. Enter your prompt or use the default one provided.
            3. Click 'Get ICD Codes' to see the suggested ICD medical codes with descriptions.
        """,
        "Health Risk & Insurance Evaluator": """
            1. Upload an image containing relevant user data using the file uploader.
            2. Enter your prompt or use the default one provided.
            3. Click 'Analyze Risk' to get the percentage risk and detailed justification.
        """,
        "Personalized Treatment & Diet Planner": """
            1. Upload an image containing patient data using the file uploader.
            2. Enter your prompt or use the default one provided.
            3. Click 'Generate Plan' to get the treatment and diet plans.
        """,
        "AI-Powered Medical Assistance": """
            1. Describe your symptoms to know the possible disease or ask a medical question in the chat input.
            2. Get AI-powered medical insights based on your symptoms.
        """
    }

    # Display the instructions for the selected page
    st.sidebar.markdown(instructions.get(page, ""))

def display_medical_news():
    # Custom CSS for the sidebar header
    st.sidebar.markdown(
        """
        <style>
        .sidebar-header {
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            padding-bottom: 15px;
        }
        </style>
        <div class="sidebar-header">📰 Latest Medical News</div>
        """,
        unsafe_allow_html=True
    )

    # Toggle button
    show_news = st.sidebar.toggle("Show Medical News", value=False)

    # Display medical news if toggled on
    if show_news:
        feed_url = "https://health.economictimes.indiatimes.com/rss/topstories"
        articles = fetch_rss_feed(feed_url) # Fetch the RSS feed

        if articles:
            for article in articles:
                st.sidebar.markdown(
                    f"<div style='font-size: 0.9rem;'>"
                    f"<b>Title:</b> <a href='{article['link']}'>{article['title']}</a><br>"
                    f"<b>Published:</b> {article['published']}</div>"
                    "<hr>",
                    unsafe_allow_html=True
                )
        else:
            st.sidebar.info("No articles available at the moment.")

# Function to handle Medical Imaging Diagnostics section
def medical_imaging_diagnostics():
    st.header("AI-Assisted Images Analysis")
    #Upload Image
    uploaded_files = st.file_uploader("Upload medical image(s) to be diagnosed (JPG, JPEG, PNG).", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    # Analysis options
    default_prompt = "Analyze this medical image. Describe what you see, identify any abnormalities, and suggest potential diagnoses."
    prompt = st.text_area("Enter your prompt:", value=default_prompt, height=100)
    analyze_button = st.button("Analyze Image")

    regenerate_button = st.button("Regenerate Analysis") # Regenerate the analysis

    # Display uploaded image and generated analysis in two columns
    if uploaded_files:
        for uploaded_file in uploaded_files:
            col1, col2 = st.columns(2)

            with col1:
                st.header("Uploaded Image")
                image = PILImage.open(uploaded_file) # Open the uploaded image
                st.image(image, caption="Uploaded Medical Image", use_container_width=True) # Display the uploaded image

            with col2:
                st.header("Image Analysis")
                if analyze_button or regenerate_button:
                    with st.spinner("Analyzing the image..."):
                        try:
                            analysis = analyze_image(image, prompt) # Analyze the image using the AI model
                            st.markdown(analysis) # Display the analysis

                            # Extract the diagnosis from the analysis
                            detailed_diagnosis = analysis # Extract the detailed diagnosis
                            diagnosis = analysis.split('.')[0] # Extract the first sentence as the diagnosis

                            # Save the uploaded image to a buffer
                            img_buffer = io.BytesIO() # Create a buffer to hold the image
                            image.save(img_buffer, format='PNG') # Save the image to the buffer
                            img_buffer.seek(0) # Reset the buffer position to the start

                            # Generate PDF report
                            pdf_buffer = create_pdf_report("-", "-", "-", diagnosis, detailed_diagnosis, "", img_buffer, "Format 1")
                            st.download_button(
                                label="Download Report",
                                data=pdf_buffer,
                                file_name=f"medical_report_{uploaded_file.name}.pdf",
                                mime="application/pdf",
                                key=f"download_{uploaded_file.name}"
                            )

                            # Search for research papers
                            st.header("Related Research Papers")
                            papers = search_research_papers(diagnosis)
                            for paper in papers:
                                st.markdown(f"[{paper['title']}]({paper['link']})")

                        except Exception as e:
                            st.error(f"An error occurred: {str(e)}")
                else:
                    st.info("Click 'Analyze Image' to start the analysis.")

# Function to handle Medical Transcription section
def medical_transcription():
    st.header("Smart Medical Transcriber")

    # Upload prescription image
    uploaded_file = st.file_uploader("Upload the image of medical prescription (JPG, JPEG, PNG).", type=["jpg", "jpeg", "png"])

    # Analysis options
    default_prompt = "Analyze this medical prescription and transcribe it in tabular format."
    prompt = st.text_area("Enter your prompt:", value=default_prompt, height=100)
    analyze_button = st.button("Get Transcription")

    # Display uploaded image and generated transcription in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Prescription")
        if uploaded_file is not None:
            image = PILImage.open(uploaded_file) # Open the uploaded image
            st.image(image, caption="Uploaded Prescription", use_container_width=True) # Display the uploaded image
        else:
            st.info("Please upload an image using the uploader.")

    with col2:
        st.subheader("Transcription in Tabular Format")
        if uploaded_file is not None and analyze_button:
            with st.spinner("Analyzing the image..."):
                try:
                    image = PILImage.open(uploaded_file) # Open the uploaded image
                    analysis = analyze_image(image, prompt) # Analyze the image using the AI model
                    st.markdown(analysis) # Display the analysis in tabular format
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        elif uploaded_file is None:
            st.info("Upload an image and click 'Get Transcription' to see the results.")
        elif not analyze_button:
            st.info("Click 'Get Transcription' to start the analysis.")

# Function to extract patient info, service info, and specimens from the analysis
def extract_info_from_analysis(analysis):
    # Default values
    theranostic_report = """lorem ipsum
    lorem ipsum
    lorem ipsum"""

    patient_info = "Patient Name:         N.A.\n" \
                   "MRN:                          N.A.\n" \
                   "DOB:                          N.A. (Age: N.A.)\n" \
                   "Gender:                      N.A.\n" \
                   "HCN:                          N.A.\n" \
                   "Ordering MD:            N.A.\n" \
                   "Copy To:                   N.A.\n" \
                   "                                      N.A."

    service_info = """lorem ipsum
    lorem ipsum
    lorem ipsum"""

    specimens = """lorem ipsum
    lorem ipsum
    lorem ipsum"""

    # Extract patient info, service info, specimens, and theranostic report
    if "Patient Name:" in analysis:
        patient_info = analysis.split("Patient Name:")[1].split("Observation:")[0].strip()
    if "Observation:" in analysis:
        service_info = analysis.split("Observation:")[1].split("Inferences:")[0].strip()
    if "Inferences:" in analysis:
        specimens= analysis.split("Inferences:")[1].split("Conclusion:")[0].strip()    
    if "Conclusion:" in analysis:
        theranostic_report = analysis.split("Conclusion:")[1].strip()     

    return patient_info, service_info, specimens, theranostic_report

# Function to handle Medical Pathology Diagnostics section
def medical_pathology_diagnostics():
    st.header("AI-Powered Lab Report Analyzer")

    uploaded_file = st.file_uploader("Upload an image of a medical report to be analysed (JPG, JPEG, PNG).", type=["jpg", "jpeg", "png"])
    
    default_prompt = """You are a highly skilled medical professional specializing in pathology. Please analyze the uploaded medical pathology report and extract the following information accurately and concisely. Present the information in a structured format with clear labels:

1. **Patient Information:**
   - Patient Name
   - Medical Record Number (MRN)
   - Date of Birth (DOB) with Age
   - Gender
   - Health Card Number (HCN)
   - Ordering Physician
   - Copy To (if any)

2. **Observation:**
   - Summarize the key observations noted in the report in a short paragraph.

3. **Inferences:**
   - Summarize the main inferences derived from the observations in a short paragraph.

4. **Conclusion:**
   - Provide the final conclusion or diagnosis mentioned in the report in a short paragraph.

**Format for Output:**

- **Patient Information:**
  - Patient Name: [Extracted Name]
  - MRN: [Extracted MRN]
  - DOB: [Extracted DOB] (Age: [Extracted Age])
  - Gender: [Extracted Gender]
  - HCN: [Extracted HCN]
  - Ordering Physician: [Extracted Physician]
  - Copy To: [Extracted Copy To (if any)]

- **Observation:**
  - [Summarized Observations]

- **Inferences:**
  - [Summarized Inferences]

- **Conclusion:**
  - [Final Conclusion or Diagnosis]

Ensure that the extracted information is accurate and formatted correctly.


"""

    prompt = st.text_area("Enter your prompt:", value=default_prompt, height=100)
    
    analyze_button = st.button("Analyze Report")

    # Display uploaded image and generated pathology report in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Report")
        if uploaded_file is not None:
            image = PILImage.open(uploaded_file) # Open the uploaded image
            st.image(image, caption="Uploaded Medical Report", use_container_width=True) # Display the uploaded image
        else:
            st.info("Please upload an image using the uploader.")

    with col2:
        st.subheader("Report Analysis")
        if uploaded_file is not None and analyze_button:
            with st.spinner("Analyzing the image..."):
                try:
                    image = PILImage.open(uploaded_file) # Open the uploaded image
                    analysis = analyze_image(image, prompt) # Analyze the image using the AI model

                    # Extract relevant details for the report
                    patient_info, service_info, specimens, theranostic_report = extract_info_from_analysis(analysis)
                    
                    # Generate pathology report
                    report_buf = create_pathology_report(patient_info, service_info, specimens, theranostic_report) # Create the pathology report
                    st.image(report_buf, caption="Pathology Report", use_container_width=True) # Display the created pathology report

                    # Save the analysis as image
                    st.download_button(
                        label="Download Report Image",
                        data=report_buf,
                        file_name=f"pathology_report_{uploaded_file.name}.png",
                        mime="image/png",
                        key=f"download_report_{uploaded_file.name}"
                    )

                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        elif uploaded_file is None:
            st.info("Upload an image and click 'Analyze Report' to see the results.")
        elif not analyze_button:
            st.info("Click 'Analyze Report' to start the analysis.")

# Function to handle Medical Coding section
def medical_coding():
    st.header("AI Medical Coding")

    # Upload medical document image
    uploaded_file = st.file_uploader("Upload a medical document image(s) (JPG, JPEG, PNG).", type=["jpg", "jpeg", "png"])
    
    # Analysis options
    default_prompt = "Analyze the image and suggest the ICD medical codes with description. Make it simple and concise."
    prompt = st.text_area("Enter your prompt:", value=default_prompt, height=100)
    analyze_button = st.button("Get ICD Codes")

    # Display uploaded image and generated ICD codes in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Medical Document")
        if uploaded_file is not None:
            image = PILImage.open(uploaded_file) # Open the uploaded image
            st.image(image, caption="Uploaded Medical Document", use_container_width=True) # Display the uploaded image
        else:
            st.info("Please upload an image using the uploader.")

    with col2:
        st.subheader("ICD Codes and Descriptions")
        if uploaded_file is not None and analyze_button:
            with st.spinner("Analyzing the image..."):
                try:
                    image = PILImage.open(uploaded_file) # Open the uploaded image
                    analysis = analyze_image(image, prompt) # Analyze the image using the AI model
                    st.markdown(analysis) # Display the analysis
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        elif uploaded_file is None:
            st.info("Upload an image and click 'Get ICD Codes' to see the results.")
        elif not analyze_button:
            st.info("Click 'Get ICD Codes' to start the analysis.")

# Function to handle Insurance Risk Analysis section
def insurance_risk_analysis():
    st.header("Health Risk & Insurance Evaluator")

    uploaded_file = st.file_uploader("Upload an image containing user data (JPG, JPEG, PNG).", type=["jpg", "jpeg", "png"])
    
    default_prompt = """You are a highly skilled insurance analyst. Please analyze the uploaded image containing user data and calculate the insurance risk percentage. Provide a detailed justification for the calculated risk percentage based on the data.

**Format for Output:**

- **Risk Percentage:** [Calculated Percentage]%
- **Justification:** [Detailed Justification]

Ensure that the calculated risk and justification are accurate and well-explained."""

    prompt = st.text_area("Enter your prompt:", value=default_prompt, height=100)
    
    analyze_button = st.button("Analyze Risk")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Uploaded User Data Image")
        if uploaded_file is not None:
            image = PILImage.open(uploaded_file)
            st.image(image, caption="Uploaded User Data Image", use_container_width=True)
        else:
            st.info("Please upload an image using the uploader.")

    with col2:
        st.subheader("Risk Analysis")
        if uploaded_file is not None and analyze_button:
            with st.spinner("Analyzing the image..."):
                try:
                    image = PILImage.open(uploaded_file)
                    analysis = analyze_image(image, prompt)
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        elif uploaded_file is None:
            st.info("Upload an image and then click 'Analyze Risk' to see the results.")
        elif not analyze_button:
            st.info("Click 'Analyze Risk' to start the analysis.")

# Function to handle Treatment and Diet Plan Generator section
def treatment_diet_plan_generator():
    st.header("Personalized Treatment & Diet Planner")

    # Upload patient data image
    uploaded_file = st.file_uploader("Choose an image containing the patient's data (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    
    # Analysis options
    default_prompt = """You are an experienced medical professional. Carefully analyze the uploaded image containing patient data and generate a well-structured treatment and diet plan based on the provided information.

**Output Format:**

**Treatment Plan:**
- Provide a detailed treatment plan, including medications, therapies, and lifestyle recommendations.
- Mention any necessary precautions or follow-up procedures.

**Diet Plan:**
- Suggest a personalized diet plan based on the patient’s condition.
- Include recommended food groups, meal timings, and any dietary restrictions.

Ensure that the recommendations are **accurate, medically sound, and easy to follow**."""

    # Enter prompt
    prompt = st.text_area("Enter your prompt:", value=default_prompt, height=100)
    
    # Generate plan button
    generate_plan_button = st.button("Generate Plan")

    # Display uploaded image and generated plan in two columns
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Uploaded Patient Data Image")
        if uploaded_file is not None:
            image = PILImage.open(uploaded_file)
            st.image(image, caption="Uploaded Patient Data Image", use_container_width=True) # Display the uploaded image
        else:
            st.info("Please upload an image using the uploader.")

    with col2:
        st.subheader("Treatment and Diet Plan")
        if uploaded_file is not None and generate_plan_button:
            with st.spinner("Generating plans..."):
                try:
                    image = PILImage.open(uploaded_file)
                    analysis = analyze_image(image, prompt)
                    st.markdown(analysis) # Display the analysis/generated plan
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        elif uploaded_file is None:
            st.info("Upload an image and click 'Generate Plan' to see the results.")
        elif not generate_plan_button:
            st.info("Click 'Generate Plan' to start the analysis.")

# Load AI medical chatbot
def load_helpbot_model():

    # configure the model using API key from the environment variables
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Google API Key not found in .env file.")
        st.stop()
    genai.configure(api_key=api_key)

    # System instruction for the AI medical chatbot
    system_instruction = """
    You are a highly skilled medical assistant. Every response MUST follow this structured format:

    - **Possible Causes:**
      - [List of possible causes]

    - **Why It Happens:**
      - [Explanation of underlying reasons]

    - **Potential Diagnosis:**
      - [Cross-referenced diagnosis suggestions]

    - **Recommended Next Steps:**
      - [Guidance on seeking medical care]

    ---
    **Disclaimer:**  
    *I am an AI medical assistant. My responses are based on training data and should not replace professional medical advice. Always consult a healthcare provider.*
    """

    return genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        system_instruction=system_instruction
    )

def load_models():

    # Load the trained models
    models = {
        "Decision Tree": joblib.load("decision_tree_model.pkl"),
        "Random Forest": joblib.load("random_forest_model.pkl"),
        "Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    }

    # Load the label encoder and symptom list
    label_encoder = joblib.load("label_encoder.pkl")
    symptom_list = joblib.load("symptom_list.pkl")
    
    return models, label_encoder, symptom_list

# Predict disease based on symptoms
def predict_disease(symptoms, model_name="Decision Tree"):
    models, label_encoder, symptom_list = load_models()

    input_vector = np.zeros(len(symptom_list))  # Create an input vector with all zeros

    for symptom in symptoms:
        if symptom in symptom_list:
            index = symptom_list.index(symptom) # Get the index of the symptom
            input_vector[index] = 1  # Set the corresponding index to 1
        else:
            st.warning(f"⚠️ Symptom '{symptom}' not recognized.") # Warn if the symptom is not recognized

    input_vector = input_vector.reshape(1, -1)  # Reshape for model input

    model = models.get(model_name)
    if not model:
        st.error(f"❌ Model '{model_name}' not found.")
        return None

    prediction_index = model.predict(input_vector)[0] # Predict the disease index
    predicted_disease = label_encoder.inverse_transform([prediction_index])[0] # Inverse transform to get the disease name

    return predicted_disease

# Disease Prediction UI
def disease_prediction_ui():
    st.subheader("📋 AI-Powered Disease Prediction")
    models, label_encoder, symptom_list = load_models() # Load the models, label encoder, and symptom list

    if "show_form" not in st.session_state:
        st.session_state.show_form = True # Initialize the show_form state
    if "predicted_disease" not in st.session_state:
        st.session_state.predicted_disease = None # Initialize the predicted_disease state

    if st.session_state.show_form:
        with st.form("disease_prediction_form"):
            st.markdown("Please provide details for better diagnosis:")

            selected_symptoms = st.multiselect("Select your symptoms:", symptom_list) # Select the symptoms
            model_name = st.selectbox("Select Model:", ["Naive Bayes", "Decision Tree", "Random Forest"]) # Select the model

            submitted = st.form_submit_button("🔍 Predict Disease")

        if submitted:
            if selected_symptoms:
                try:
                    st.session_state.predicted_disease = predict_disease(selected_symptoms, model_name)
                except Exception as e:
                    st.session_state.predicted_disease = f"⚠️ Error in prediction: {str(e)}"
            else:
                st.warning("⚠️ Please select at least one symptom.")

    if st.session_state.predicted_disease:
        st.success(f"🔍 **Predicted Disease:** {st.session_state.predicted_disease}")

    if st.session_state.show_form:
        if st.button("❌ Close Form"):
            st.session_state.show_form = False
            st.session_state.predicted_disease = None
            st.rerun()
    else:
        if st.button("🩺 Start Disease Prediction"):
            st.session_state.show_form = True
            st.rerun()

# HelpBot UI
def helpBot():
    st.subheader("💬 Medical Query Assistant")

    if "helpbot_chat_session" not in st.session_state:
        model = load_helpbot_model() # Load the AI medical chatbot model
        st.session_state.helpbot_chat_session = model.start_chat(history=[]) # Start the chat session

    if "helpbot_messages" not in st.session_state:
        st.session_state.helpbot_messages = [] # Initialize the chat history

    st.container()
    for message in st.session_state.helpbot_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"]) # Display the chat history

    if user_input := st.chat_input("Describe your symptoms or ask a medical question..."):
        st.chat_message("user").markdown(user_input) # Display user input in the chat
        st.session_state.helpbot_messages.append({"role": "user", "content": user_input}) # Add user input to the chat history

        with st.chat_message("assistant"): 
            message_placeholder = st.empty() # Create a placeholder for the response
            full_response = ""

            with st.spinner("Processing..."): # Show a spinner while processing the response
                response_stream = st.session_state.helpbot_chat_session.send_message(user_input, stream=True) # Send user input to the chatbot
                for chunk in response_stream: # Get the response from the chatbot
                    full_response += chunk.text # Append the response to the full response
                    message_placeholder.markdown(full_response + "▌") # Display the response in the chat
                message_placeholder.markdown(full_response) # Display the full response

def medical_query_assisstant():
    st.header("🩺 AI-Powered Medical Assistance")

    col1, col2 = st.columns(2)
    with col1:
        st.success("📊 Get AI-powered medical insights based on your symptoms.")
        st.info("📈 Cross-referenced potential diagnoses for better decision-making.")
    with col2:
        st.warning("📄 Receive recommendations for further medical consultation.")
        st.error("💡 AI-assisted healthcare guidance tailored to your needs.")

    col1, col2 = st.columns([2, 2])
    with col1:
        helpBot()
    with col2:
        disease_prediction_ui()

# Main app
def main():
    st.sidebar.markdown("<h3 style='text-align: center; color: #11111; font-family: comic sans ms;'>Swasthya Sathi AI</h3>", unsafe_allow_html=True)
    display_instructions(selected) # Display instructions based on the selected option
    display_medical_news() # Display medical news in the sidebar

    if selected == "AI-Assisted Images Analysis":
        medical_imaging_diagnostics()
    elif selected == "Smart Medical Transcriber":
        medical_transcription()
    elif selected == "AI-Powered Lab Report Analyzer":
        medical_pathology_diagnostics()
    elif selected == "AI Medical Coding":
        medical_coding()
    elif selected == "Health Risk & Insurance Evaluator":
        insurance_risk_analysis()
    elif selected == "Personalized Treatment & Diet Planner":
        treatment_diet_plan_generator()
    elif selected == "AI-Powered Medical Assistance":
        medical_query_assisstant()

if __name__ == "__main__":
    main()
