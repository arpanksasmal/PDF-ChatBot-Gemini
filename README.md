# PDF-ChatBot-Gemini

This project provides an interface to interact with multiple PDF documents using Google Generative AI models. Users can upload PDF files, process them, and ask questions based on the content of these PDFs. The application uses a combination of Google Generative AI, LangChain, and FAISS for efficient text extraction and question-answering.

## Features

- **Upload PDF Files**: Users can upload multiple PDF files.
- **Text Extraction**: Extracts and processes text from uploaded PDFs.
- **Model Selection**: Choose from various Gemini models for question-answering.
- **Temperature Control**: Adjust the creativity of the AI responses.
- **Interactive QA**: Ask questions based on the content of the PDFs.

## Requirements

The project relies on the following Python packages:

- `streamlit`
- `google-generativeai`
- `python-dotenv`
- `langchain`
- `PyPDF2`
- `faiss-cpu`
- `langchain_google_genai`

These dependencies are listed in the `requirements.txt` file.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/arpanksasmal/PDF-ChatBot-Gemini.git
cd PDF-ChatBot-Gemini
```

### 2. Create and Activate a Virtual Environment

#### Using `venv`

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS and Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

#### Using `conda`

**On Windows, macOS, and Linux:**

```bash
conda create --name pdf-chatbot-gemini python=3.8
conda activate pdf-chatbot-gemini
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory of the project and add your Google API key:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Run the Application

```bash
streamlit run app.py
```

This command will start the Streamlit server, and you can access the application via your web browser.

## Usage

- **Upload PDF Files**: Use the file uploader in the sidebar to upload your PDF documents.
- **Process PDFs**: Click on "Submit & Process" to extract and store the text from the uploaded PDFs.
- **Select a Model**: Choose the desired Gemini model from the dropdown.
- **Set Temperature**: Adjust the slider to set the temperature for the AI responses.
- **Ask Questions**: Enter your question in the text input box to get answers based on the content of the PDFs.

## Error Handling

- **API Key**: Ensure your Google API key is set in the `.env` file.
- **File Uploads**: Check if the PDF files are correctly uploaded and processed.

## Contact

For any issues or questions, please contact Arpan Kumar Sasmal at [arpankumarsasmal@gmail.com](mailto:arpankumarsasmal@gmail.com).

## License

This project is licensed under the MIT License.