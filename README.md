PDFProcessor
PDFProcessor is a Python-based application designed to facilitate the extraction, processing, and interaction with PDF documents. Leveraging advanced natural language processing techniques, this tool enables users to query PDF content through a conversational interface.

Features
PDF Upload: Seamlessly upload PDF documents for processing.

Text Extraction: Extracts readable text from PDF files.

Text Chunking: Splits extracted text into manageable chunks for efficient processing.

Semantic Search: Utilizes embeddings to perform semantic searches within the document.

Conversational Interface: Engage in a chat-like interface to ask questions related to the PDF content.

Real-time Responses: Receive answers in real-time as the system processes your queries.

Installation
Prerequisites
Ensure you have the following installed:

Python 3.10 or higher

Ollama (for LLM interactions)

Clone the Repository
bash
Copy
Edit
git clone https://github.com/cebause01/PDFProcessor.git
cd PDFProcessor
Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Run the Application
bash
Copy
Edit
streamlit run main.py
Usage
Upload a PDF: Click on the "Upload a PDF" button in the sidebar to upload your document.

Ask Questions: Once the PDF is processed, type your questions in the input field and press Enter.

View Responses: The system will display answers based on the content of the PDF.

Technologies Used
Python: Programming language used for development.

Streamlit: Framework for building the web application.

PyPDF2: Library for PDF text extraction.

NumPy: Library for numerical operations.

Scikit-learn: Library for machine learning and cosine similarity calculations.

Ollama API: Interface for large language model interactions.

Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure your code adheres to the existing style and includes appropriate tests.

License
This project is licensed under the MIT License - see the LICENSE file for details.

