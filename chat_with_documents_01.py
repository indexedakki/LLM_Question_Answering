# Install all libraries by running in the terminal: pip install -q -r ./requirements.txt
import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb
# loading PDF, DOCX and TXT files as LangChain Documents
def load_document(file):
    import os
    name, extension = os.path.splitext(file)

    if extension == '.pdf':
        from langchain_community.document_loaders import PyPDFLoader
        print(f'Loading {file}')
        loader = PyPDFLoader(file)
    elif extension == '.docx':
        from langchain_community.document_loaders import Docx2txtLoader
        print(f'Loading {file}')
        loader = Docx2txtLoader(file)
    elif extension == '.txt':
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(file)
    else:
        print('Document format is not supported!')
        return None

    data = loader.load()
    return data


# splitting data in chunks
def chunk_data(data, chunk_size=256, chunk_overlap=20):
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(data)
    return chunks


# create embeddings using OpenAIEmbeddings() and save them in a Chroma vector store
def create_embeddings(chunks):
    # embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  # 512 works as well
    # vector_store = Chroma.from_documents(chunks, embeddings)

    # # if you want to use a specific directory for chromadb
    # # vector_store = Chroma.from_documents(chunks, embeddings, persist_directory='./mychroma_db')
    # return vector_store

    CHROMA_DATA_PATH = "chroma_data/"
    
   
    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
    
    # Define the name of the collection you want to check
    collection_name = "demo_docs"

    # Try to get the collection
    Flag =False
    try:
        collection = client.get_collection(name=collection_name)
        print(f"The collection '{collection_name}' exists.")
        Flag =True
        if Flag:
            # Assuming 'client' is your ChromaDB client instance
            client.delete_collection(name=collection_name)
            print(f"The collection '{collection_name}' deleted")
    except Exception as e:
        print(f"The collection '{collection_name}' does not exist or an error occurred: {e}")

    
    from chromadb.utils import embedding_functions

    CHROMA_DATA_PATH = "chroma_data/"
    EMBED_MODEL = "all-MiniLM-L6-v2"
    COLLECTION_NAME = "demo_docs"

    client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)

    embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

    documents=[doc.page_content for doc in chunks]
    collection.add(
            documents=[doc.page_content for doc in chunks],
            ids=[f"id{i}" for i in range(len(documents))],
            metadatas=[doc.metadata for doc in chunks]
            )
    return collection, client, embedding_func, COLLECTION_NAME


def ask_and_get_answer(vector_store, q, client, embedding_func, COLLECTION_NAME, k=3 ):
    # from langchain.chains import RetrievalQA
    # from langchain_openai import ChatOpenAI

    # llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)
    # retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})
    # chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # answer = chain.invoke(q)
    # return answer['result']

    vector_store = client.get_collection(
                            name=COLLECTION_NAME, 
                            embedding_function=embedding_func)

    context = """
    You are a well versed with knowledge about the file which is uploaded: {}
    """

    question = q

    query_results = vector_store.query(
        query_texts=[question],
        n_results=k
    )

    query_results.keys()

    # print(query_results["documents"])

    reviews_str = ",".join(query_results["documents"][0])

    # print(reviews_str)

    from langchain.chat_models import AzureChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage

    BASE_URL = "https://trail-outcome.openai.azure.com/"
    API_KEY = "b59f23e204b3426c9dbe1f6741b80acb"
    DEPLOYMENT_NAME = "LMM_OPENAI"

    model = AzureChatOpenAI(
        openai_api_base=BASE_URL,
        openai_api_version="2023-03-15-preview",
        deployment_name=DEPLOYMENT_NAME,
        openai_api_key=API_KEY,
        openai_api_type="azure",
    )

    # print(context.format(reviews_str))
    return (model(
        [
            SystemMessage(
                content= context.format(reviews_str)
            ),
            HumanMessage(
                content=q
            )
        ]
    ))


# calculate embedding cost using tiktoken
def calculate_embedding_cost(texts):
    import tiktoken
    enc = tiktoken.encoding_for_model('text-embedding-3-small')
    total_tokens = sum([len(enc.encode(page.page_content)) for page in texts])
    # check prices here: https://openai.com/pricing
    # print(f'Total Tokens: {total_tokens}')
    # print(f'Embedding Cost in USD: {total_tokens / 1000 * 0.00002:.6f}')
    return total_tokens, total_tokens / 1000 * 0.00002


# clear the chat history from streamlit session state
def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']
    


if __name__ == "__main__":
    import os

    # loading the OpenAI api key from .env
    # from dotenv import load_dotenv, find_dotenv
    # load_dotenv(find_dotenv(), override=True)

    # import shutil

    # # Define the path to the directory you want to delete
    # directory_path = 'chroma_data'

    # # Use shutil.rmtree to delete the directory and all its contents
    # if os.path.exists(directory_path):
    #     shutil.rmtree(directory_path)

    # st.image('img.png')
    st.subheader('LLM Question-Answering Application 🤖')
    with st.sidebar:
        # text_input for the OpenAI API key (alternative to python-dotenv and .env)
        # api_key = st.text_input('OpenAI API Key:', type='password')
        # if api_key:
        #     os.environ['OPENAI_API_KEY'] = api_key

        # file uploader widget
        uploaded_file = st.file_uploader('Upload a file:', type=['pdf', 'docx', 'txt'])

        # chunk size number widget
        chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=512, on_change=clear_history)

        # k number input widget
        k = st.number_input('k', min_value=1, max_value=20, value=3, on_change=clear_history)

        # add data button widget
        add_data = st.button('Add Data', on_click=clear_history)


        if uploaded_file and add_data: # if the user browsed a file
            with st.spinner('Reading, chunking and embedding file ...'):

                # writing the file from RAM to the current directory on disk
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)

                data = load_document(file_name)
                chunks = chunk_data(data, chunk_size=chunk_size)
                st.write(f'Chunk size: {chunk_size}, Chunks: {len(chunks)}')

                tokens, embedding_cost = calculate_embedding_cost(chunks)
                st.write(f'Embedding cost: ${embedding_cost:.4f}')

                # creating the embeddings and returning the Chroma vector store
                vector_store, client, embedding_func, COLLECTION_NAME = create_embeddings(chunks)

                # saving the vector store in the streamlit session state (to be persistent between reruns)
                st.session_state.vs = vector_store, client, embedding_func, COLLECTION_NAME
                st.success('File uploaded, chunked and embedded successfully.')

    # user's question text input widget
    q = st.text_input('Ask a question about the content of your file:')
    if q: # if the user entered a question and hit enter
        standard_answer = "Answer only based on the text you received as input. Don't search external sources. " \
                          "If you can't answer then return `I DONT KNOW`."
        q = f"{q} {standard_answer}"
        if 'vs' in st.session_state: # if there's the vector store (user uploaded, split and embedded a file)
            vector_store, client, embedding_func, COLLECTION_NAME = st.session_state.vs
            st.write(f'k: {k}')
            answer = ask_and_get_answer(vector_store, q, client, embedding_func, COLLECTION_NAME, k )

            # text area widget for the LLM answer
            st.text_area('LLM Answer: ', value=answer)

            st.divider()

            # if there's no chat history in the session state, create it
            if 'history' not in st.session_state:
                st.session_state.history = ''

            # the current question and answer
            value = f'Q: {q} \nA: {answer}'

            st.session_state.history = f'{value} \n {"-" * 100} \n {st.session_state.history}'
            h = st.session_state.history

            # text area widget for the chat history
            st.text_area(label='Chat History', value=h, key='history', height=400)

# run the app: streamlit run ./chat_with_documents.py

