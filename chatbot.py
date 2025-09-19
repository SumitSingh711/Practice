from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

client = ChatGroq(
    model_name="llama-3.1-8b-instant"
)

chat_history = []

while True:
    user = input('You: ')
    chat_history.append({'role': 'user', 'content': user})
    if user == 'exit':
        break
    response = client.invoke(chat_history)
    chat_history.append({'role': 'assistant', 'content': response.content})
    print(f'Tokens: {response.response_metadata["token_usage"]["total_tokens"]}, Bot: {response.content}')

print(chat_history)