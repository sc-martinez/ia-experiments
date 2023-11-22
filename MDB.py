import bs4
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
from sys import stdin
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def loadContent(content_url):
    loader = WebBaseLoader(
        web_paths=(content_url,),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    print(splits[0])
    print(splits[1])

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever()

    prompt = hub.pull("rlm/rag-prompt")
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return rag_chain


print("Please type the url of content that you want to process and make questions about Ej 'https://lilianweng.github.io/posts/2023-06-23-agent/':")

contentUrl = stdin.readline().strip()

print("Loading content into RAG model .......")

rag_chain = loadContent(contentUrl)

print("Please type your question around the loaded content  Ej: 'What is Task Decomposition?'")
print("=============================================================")
print("Question : ")
line = stdin.readline().strip()
while line:
    question = line
    response = rag_chain.invoke(line)
    print("Response:")
    print(response)
    print("=============================================================")
    print("Question : ")
    line = stdin.readline().strip()
