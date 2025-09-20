from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()

model = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Give the Sentiment of the feedback")

parser1 = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=Feedback)

prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into a postitive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Write an appropriate resposnse to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Write an appropriate resposnse to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

positive_chain = prompt2 | model | parser1
negative_chain = prompt3 | model | parser1

branch_chain = RunnableBranch(
    (lambda x: x.sentiment == 'positive', positive_chain),
    (lambda x: x.sentiment == 'negative', negative_chain),
    RunnableLambda(lambda x: "could not find sentiment"), 
)

conditional_chain = classifier_chain | branch_chain

print(conditional_chain.invoke({
    'feedback': 'This is a beautiful phone'
}))

conditional_chain.get_graph().print_ascii()