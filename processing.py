from abc import ABC,abstractmethod
import os
import dotenv
from langchain_community.document_loaders.pdf import PyPDFLoader#type:ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter#type:ignore
from langchain_community.vectorstores.chroma import Chroma#type:ignore
from langchain_community.embeddings import HuggingFaceEmbeddings#type:ignore
from langchain_openai.llms import OpenAI#type:ignore
from langchain.retrievers.self_query.base import SelfQueryRetriever#type:ignore
from langchain.chains.query_constructor.schema import AttributeInfo #type:ignore
from langchain_community.document_loaders import TextLoader

dotenv.load_dotenv()


class FileLoader(ABC):

    @abstractmethod
    def processor_file(self):
        pass
    
    @abstractmethod
    def splitting_text(self):
        pass
    
    @abstractmethod
    def embedding_vector_store(self):
        pass

    @abstractmethod
    def call_ai(self):
        pass



class PdfLoader(FileLoader):
    def __init__(self,arquivos):
        self.file = arquivos

    def processor_file(self):

        loader = PyPDFLoader(self.file)
        arquivo = loader.load()

        return arquivo
    
    def splitting_text(self):

        split = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50,
            separators=["\n\n","\n","","."," "]

        )

        docs = split.split_documents(self.processor_file())
        return docs
    
    def embedding_vector_store(self):

        diretorio = "pdf_directory"

        embedding = HuggingFaceEmbeddings()

        vector_db = Chroma.from_documents(documents=self.splitting_text(),embedding=embedding,persist_directory=diretorio)

        return vector_db

    

    def call_ai(self):

        document_description = "apostilas de informações"

        metadata_info = [AttributeInfo(name='source',description='Nome da apostila de onde o texto original foi retirado. Pode ser "apostila.pdf" ou "LLM.pdf".',type='string'),AttributeInfo(name='page',description='A página da apostila de onde o texto foi extraído. Número da página.',type='integer')]

        llm = OpenAI(api_key=os.getenv("openaiKey"))
        retriever = SelfQueryRetriever.from_llm(
            llm,
            self.embedding_vector_store(),
            document_description,
            metadata_info,
            verbose=True
            
        )

        return retriever
        

class TxtLoader(FileLoader):
    def __init__(self,arquivos):
        self.file = arquivos

    def processor_file(self):

        loader = TextLoader(self.file)
        arquivo = loader.load()

        return arquivo
    
    def splitting_text(self):

        split = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50,
            separators=["\n\n","\n","","."," "]

        )

        docs = split.split_documents(self.processor_file())
        return docs
    
    def embedding_vector_store(self):

        diretorio = "txt_directory"

        embedding = HuggingFaceEmbeddings()

        vector_db = Chroma.from_documents(documents=self.splitting_text(),embedding=embedding,persist_directory=diretorio)

        return vector_db

    

    def call_ai(self):

        document_description = "apostilas de informações"

        metadata_info = [AttributeInfo(name='source',description='Nome da arquivo de onde o texto original foi retirado. Pode ser "apostila.txt" ou "LLM.txt"',type="string")]

        llm = OpenAI(api_key=os.getenv("openaiKey"))
        retriever = SelfQueryRetriever.from_llm(
            llm,
            self.embedding_vector_store(),
            document_description,
            metadata_info,
            verbose=True
            
        )

        return retriever
    


class FactoryLoader:
    def __init__(self):
        ...

    def get_loader(self,type,arquivo):
        if type == "application/pdf":
            pdf_path = arquivo.read()
            with open("ui_pdf.pdf","wb")as f:
                f.write(pdf_path)
            pdf_docs = "ui_pdf.pdf"
            pdf = PdfLoader(pdf_docs)
            final_pdf = pdf.call_ai()
            return final_pdf
        else:

            if arquivo.type == "text/plain":
                text = arquivo.read()
                with open("ui_txt.txt","wb") as f:
                    f.write(text)
                docs_txt = "ui_txt.txt"
                txt = TxtLoader(docs_txt)
                final_txt = txt.call_ai()
                return final_txt
            