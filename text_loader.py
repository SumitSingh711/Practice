from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='write a summary of following poem - \n {poem}',
    input_variables=['poem']
)

txt_path = r'cricket.txt'

loader = TextLoader(txt_path, encoding='utf-8')

docs = loader.load() 

chain = prompt | model | parser
print(chain.invoke({'poem': docs[0].page_content}))

