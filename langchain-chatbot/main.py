from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DeepLake
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import SeleniumURLLoader
from langchain import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# 0. 설정
# ============================================================
# DeepLake 저장 위치를 클라우드/로컬 중 선택
#   False → hub://{ORG_ID}/{DATASET_NAME}  (ACTIVELOOP_TOKEN 필요)
#   True  → 로컬 폴더 (토큰 불필요)
USE_LOCAL_DB = True

# Activeloop 네임스페이스. 기본값은 가입 시 사용자명이며,
# 실제 계정의 네임스페이스와 '정확히' 일치해야 push가 됩니다.
ORG_ID = "sookyeong0505s-organization"
DATASET_NAME = "jetbrains_article_dataset"
LOCAL_DB_PATH = r"/db"

dataset_path = LOCAL_DB_PATH if USE_LOCAL_DB else f"hub://{ORG_ID}/{DATASET_NAME}"

# 데이터 소스 기사 링크
ARTICLES = [
    "https://www.digitaltrends.com/computing/claude-sonnet-vs-gpt-4o-comparison/",
    "https://www.digitaltrends.com/computing/apple-intelligence-proves-that-macbooks-need-something-more/",
    "https://www.digitaltrends.com/computing/how-to-use-openai-chatgpt-text-generation-chatbot/",
    "https://www.digitaltrends.com/computing/character-ai-how-to-use/",
    "https://www.digitaltrends.com/computing/how-to-upload-pdf-to-chatgpt/",
]

# ============================================================
# 1. 문서 로드 & 분할
# ============================================================
# SeleniumURLLoader는 Chrome + 드라이버가 필요합니다.
# selenium 4.6+ 는 Selenium Manager가 드라이버를 자동으로 받아오므로,
# Chrome 브라우저만 설치돼 있으면 대체로 별도 설정이 필요 없습니다.

# 1.1 셀레니움으로 문서를 로드
loader = SeleniumURLLoader(urls=ARTICLES)
docs_not_splitted = loader.load()
print("문서 로드 완료")

# 1.2 문서를 작게 나눈다. (청크 크기=1000)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(docs_not_splitted)
print("문서 split 완료")


# ============================================================
# 2. 임베딩 & 벡터 DB 구축
# ============================================================
embeddings = OpenAIEmbeddings()
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
print("벡터 DB 구축")

# 주의: 스크립트를 다시 실행할 때마다 같은 문서가 '중복' 저장됩니다.
#       한 번 넣은 뒤에는 아래 줄을 주석 처리하고 검색만 반복하세요.
# db.add_documents(docs)

# ============================================================
# 3. 프롬프트 · 메모리 · 체인 구성
# ============================================================
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

# memory가 chat_history를 자동으로 주입/저장하므로,
# 아래 chain.predict 호출 시 chat_history를 직접 넘기지 않습니다.
memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")

# API 키는 환경변수(OPENAI_API_KEY)에서 자동으로 읽습니다. 코드에 하드코딩 금지.
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0)

# 메모리를 포함하는 LLM 체인 생성
chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory
)

# ============================================================
# 4. 질의응답 함수
# ============================================================
def ask(query: str) -> str:
    """질문과 유사한 청크를 검색해 컨텍스트로 넣고 답변을 생성한다."""
    found = db.similarity_search(query)
    print(f"similarity_search 결과 : \n {docs}")
    chunks_formatted = "\n\n".join(doc.page_content for doc in found)

    # input / chunks_formatted 만 전달 → chat_history는 memory가 처리
    return chain.predict(input=query, chunks_formatted=chunks_formatted)


query = "how to check disk usage in linux?"
response = ask(query)
print(response)


# ============================================================
# 5. 실행 예시
# ============================================================
if __name__ == "__main__":
    q1 = "How to check disk usage in linux?"
    print("Q:", q1)
    print("A:", ask(q1))
    print("-" * 60)

    # 두 번째 질문 — memory 덕분에 이전 대화 맥락이 이어집니다.
    q2 = "What was the 5th point about on the question how to remove spotify account?"
    print("Q:", q2)
    print("A:", ask(q2))