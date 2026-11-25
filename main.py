import os
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"), override=True)

PDF_PATH = os.path.join(BASE_DIR, "LTI_Convolution_5_Page_Report-2.pdf")

if not os.path.isfile(PDF_PATH):
    raise FileNotFoundError(f"Uploaded PDF not found: {PDF_PATH}")

api_key = os.getenv("NVIDIA_API_KEY")
if not api_key:
    raise RuntimeError(
        "NVIDIA_API_KEY is not set. Add it to the .env file as NVIDIA_API_KEY=your_key"
    )

print(f"Loading document from: {PDF_PATH}")
docs = PyPDFLoader(PDF_PATH).load()
print(f"Loaded {len(docs)} document pages.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100,
)
chunks = text_splitter.split_documents(docs)
print(f"Created {len(chunks)} text chunks.")

print("Creating embeddings and indexing into Qdrant...")
embeddings = FastEmbedEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    cache_dir=os.path.join(BASE_DIR, ".fastembed"),
)

client = QdrantClient(location=":memory:")
client.create_collection(
    collection_name="lti_convolution_report",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
)

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="lti_convolution_report",
    embedding=embeddings,
)
vectorstore.add_documents(chunks)
print("Indexing complete!")

llm = ChatNVIDIA(
    model="nvidia/nemotron-3.5-lightning-30b-a3b",
    nvidia_api_key=api_key,
    temperature=0.2,
    max_completion_tokens=2500,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert tutor for the uploaded report on convolution in "
        "Linear Time-Invariant (LTI) systems. Answer ONLY from the provided context. "
        "Use the report's values exactly: x[n] = [1, 2, 1, 0], "
        "h[n] = [1, 1, 1], and y[n] = [1, 3, 4, 3, 1, 0]. "
        "Explain equations and results clearly. If the context does not contain "
        "the answer, say that you don't know.\n\nContext:\n{context}",
    ),
    ("human", "{question}"),
])

rag_chain = prompt | llm | StrOutputParser()


def query_rag(question: str):
    print("\n" + "=" * 60)
    print(f"Query: {question}")
    print("=" * 60)

    retrieved_docs = retriever.invoke(question)
    print(f"\nRetrieved {len(retrieved_docs)} relevant context chunk(s):")
    for i, doc in enumerate(retrieved_docs, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        preview = doc.page_content.strip().replace("\n", " ")[:150]
        print(f" [{i}] (PDF page {page_num + 1 if isinstance(page_num, int) else page_num}): {preview}...")

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    print("\nGenerating response from NVIDIA LLM...")
    response = rag_chain.invoke({"context": context, "question": question})
    print("\nAnswer:")
    print(response)
    print("=" * 60)
    return response


if __name__ == "__main__":
    test_question = (
        "Calculate the convolution of x[n] = [1, 2, 1, 0] and "
        "h[n] = [1, 1, 1]. Explain the result and its length."
    )
    query_rag(test_question)
