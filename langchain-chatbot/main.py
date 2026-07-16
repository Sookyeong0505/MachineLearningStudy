from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.vectorstores import DeepLake

from langchain.text_splitter import CharacterTextSplitter

from langchain import OpenAI

from langchain.document_loaders import SeleniumURLLoader

from langchain import PromptTemplate


articles = ['https://www.digitaltrends.com/computing/claude-sonnet-vs-gpt-4o-comparison/',
           'https://www.digitaltrends.com/computing/apple-intelligence-proves-that-macbooks-need-something-more/',
           'https://www.digitaltrends.com/computing/how-to-use-openai-chatgpt-text-generation-chatbot/',
           'https://www.digitaltrends.com/computing/character-ai-how-to-use/',
           'https://www.digitaltrends.com/computing/how-to-upload-pdf-to-chatgpt/']

# Use the selenium to load the documents
loader = SeleniumURLLoader(urls=articles)
docs_not_splitted = loader.load()

# 크기 1000으로 청크를 나눈다.
# Split the documents into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(docs_not_splitted)

# TODO: 수정하기
my_activeloop_org_id = "didogrigorov"
my_activeloop_dataset_name = "jetbrains_article_dataset"
dataset_path = f"hub://{my_activeloop_org_id}/{my_activeloop_dataset_name}"
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)


# add documents to our Deep Lake dataset
db.add_documents(docs)

# Check the top relevant documents to a specific query
query = "how to check disk usage in linux?"
docs = db.similarity_search(query)
print(docs[0].page_content)

# 4. 챗봇 기능 만들기
# user question
query = "How to check disk usage in linux?"

# retrieve relevant chunks
docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]

# format the prompt
chunks_formatted = "nn".join(retrieved_chunks)
prompt_formatted = prompt.format(chunks_formatted=chunks_formatted, query=query)

# generate answer
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)
answer = llm(prompt_formatted)
print(answer)

# 5. 대화기록 만들기
# Create conversational memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")

# Define a prompt template that includes memory
template = """You are an exceptional customer support chatbot that gently answers questions.

{chat_history}

You know the following context information.

{chunks_formatted}

Answer the following question from a customer. Use only information from the previous context information. Do not invent stuff.

Question: {input}

Answer:"""

prompt = PromptTemplate(
    input_variables=["chat_history", "chunks_formatted", "input"],
    template=template,
)

# Initialize the OpenAI model
llm = OpenAI(openai_api_key="YOUR API KEY", model="gpt-3.5-turbo-instruct", temperature=0)

# Create the LLMChain with memory
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# User query
query = "What was the 5th point about on the question how to remove spotify account?"

# Retrieve relevant chunks
docs = db.similarity_search(query)
retrieved_chunks = [doc.page_content for doc in docs]

# Format the chunks for the prompt
chunks_formatted = "nn".join(retrieved_chunks)

# Prepare the input for the chain
input_data = {
    "input": query,
    "chunks_formatted": chunks_formatted,
    "chat_history": memory.buffer
}

# Simulate a conversation
response = chain.predict(**input_data)

print(response)