from langchain_community.retrievers import WikipediaRetriever

retriver = WikipediaRetriever(top_k_results=2, lang='en')

query = 'Who is the best all format batsman in the world'

docs = retriver.invoke(query)

print(docs[1].metadata)

for i, doc in enumerate(docs):
    print(i, doc.metadata)