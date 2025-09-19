from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

prompt = PromptTemplate(
    template='Write 5 interesting facts about {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke(({'topic': 'Black Hole'}))

chain.get_graph().print_ascii()