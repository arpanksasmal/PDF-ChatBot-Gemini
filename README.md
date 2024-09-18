# PDF-ChatBot-Gemini

This project provides an advanced interface for interacting with multiple PDF documents using Google Generative AI models. Users can upload PDF files, process them, and ask questions based on the content of these documents. The application leverages Google Generative AI, LangChain, and FAISS for efficient text extraction and question-answering.

## Live Demo

You can try out the live version of the application [here](https://pdf-chatbot-gemini-kg58jxmwx5gzwuzoznjirf.streamlit.app/).

## Features

- **Upload PDF Files**: Allows users to upload multiple PDF files simultaneously.
- **Text Extraction**: Efficiently extracts and processes text from the uploaded PDFs.
- **Model Selection**: Choose from various Gemini models for tailored question-answering.
- **Temperature Control**: Adjust the creativity of the AI responses with a temperature setting.
- **Interactive QA**: Ask questions and receive answers based on the content of the PDFs.

## Requirements

This project requires the following Python packages:

- `streamlit`
- `google-generativeai`
- `python-dotenv`
- `langchain`
- `PyPDF2`
- `faiss-cpu`
- `langchain_google_genai`

All dependencies are listed in the `requirements.txt` file.

## Setup

Follow these steps to set up and run the application:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/arpanksasmal/PDF-ChatBot-Gemini.git
   cd PDF-ChatBot-Gemini
   ```

2. **Create and Activate a Virtual Environment**

   - **Using venv**

     On Windows:
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

     On macOS and Linux:
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

   - **Using conda**

     On Windows, macOS, and Linux:
     ```bash
     conda create --name pdf-chatbot-gemini python=3.8
     conda activate pdf-chatbot-gemini
     ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Create a `.env` file in the root directory of the project and add your Google API key:

   ```env
   GOOGLE_API_KEY=your_google_api_key_here
   ```

5. **Run the Application**

   ```bash
   streamlit run app.py
   ```

   This command will start the Streamlit server. You can access the application via your web browser.

## Usage

- **Upload PDF Files**: Use the file uploader in the sidebar to upload your PDF documents.
- **Process PDFs**: Click on "Submit & Process" to extract and store the text from the uploaded PDFs.
- **Select a Model**: Choose the desired Gemini model from the dropdown menu.
- **Set Temperature**: Adjust the slider to set the temperature for the AI responses.
- **Ask Questions**: Enter your question in the text input box to get answers based on the content of the PDFs.

## Error Handling

- **API Key**: Ensure your Google API key is correctly set in the `.env` file.
- **File Uploads**: Verify that the PDF files are uploaded and processed correctly.

## Contact

For any issues or questions, please contact Arpan Kumar Sasmal at [arpankumarsasmal@gmail.com](mailto:arpankumarsasmal@gmail.com).

## License

This project is licensed under the [MIT License](LICENSE).