LLM Question Answering App
Overview
The LLM Question Answering App is a web application designed to facilitate the uploading and processing of text files (docx, pdf, txt) using Streamlit. It leverages advanced technologies such as LangChain, Large Language Models (LLM), ChromaDB, and Azure OpenAI to convert text into embeddings and store them in ChromaDB. This enables users to query documents for specific information, making it an invaluable tool for extracting insights from large volumes of text.
Features
•	Upload and Process Text Files: Users can upload text files in various formats (docx, pdf, txt) and process them within the app.
•	Text to Embeddings Conversion: The app converts uploaded text into embeddings, enhancing the searchability and accessibility of the content.
•	ChromaDB Storage: Embeddings are stored in ChromaDB, a vector store optimized for efficient querying.
•	Query Documents: Users can ask questions related to the document, and the app will provide relevant answers extracted directly from the document.
Getting Started
Prerequisites
Ensure you have Python version 3.7 or higher installed on your system.
Installation
1.	Clone the Repository: Clone this repository to your local machine.
git clone https://github.com/yourusername/llm-question-answering-app.git


2.	Install Dependencies: Navigate to the project directory and install the required Python libraries.
pip install streamlit langchain openai chromadb tiktoken


Running the App
1.	Start the Streamlit Server: Run the Streamlit app using the following command.
streamlit run app.py


2.	Access the App: Open your web browser and navigate to the URL provided by Streamlit (usually http://localhost:8501).
Deployment
For deploying your app to the Streamlit Community Cloud, follow these steps:
1.	Navigate to Streamlit Community Cloud: Go to the Streamlit Community Cloud and click the "New app" button.
2.	Choose Repository, Branch, and Application File: Select the appropriate repository, branch, and application file to deploy your app.
Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue if you find any bugs or have suggestions for improvements.
License
This project is licensed under the MIT License. See the LICENSE file for details.
________________________________________
This README provides a comprehensive overview of the LLM Question Answering App, including setup instructions, features, and deployment guidelines. It's designed to help users quickly understand how to use and contribute to the project.

