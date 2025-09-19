from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

template1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='Write a 5 line summary on the following text. /n {text}',
    input_variables=['text']
)


'''Manual method of prompting'''
# prompt1 = template1.invoke({'topic': 'Black Hole'})
# result1 = model.invoke(prompt1)

# prompt2 = template2.invoke({'text': result1.content})
# result2 = model.invoke(prompt2)

# print(result2.content)

'''Using str parser'''

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model

result = chain.invoke({'topic': 'Black Hole'})
print(result.content)