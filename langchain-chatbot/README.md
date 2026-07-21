# LangChain Chatbot

LangChain과 OpenAI API를 활용한 문맥 기반 대화형 챗봇입니다. 벡터 데이터베이스를 사용하여 외부 데이터를 검색하고 이를 기반으로 답변을 생성합니다.

## 📋 프로젝트 개요

이 프로젝트는 다음의 핵심 기능을 구현합니다:

- **문서 로드**: Selenium을 사용하여 웹 URL에서 동적으로 문서를 로드
- **문서 분할**: 대용량 문서를 청크 단위로 분할 (청크 크기: 1000, 중복: 200)
- **임베딩 & 벡터화**: OpenAI 임베딩 모델을 사용하여 문서를 벡터로 변환
- **벡터 데이터베이스**: DeepLake를 사용하여 벡터 데이터를 저장 및 검색
- **대화형 챗봇**: LLMChain과 메모리를 활용하여 대화 맥락을 유지하는 챗봇
- **유사도 검색**: 사용자의 질문과 유사한 문서 청크를 자동으로 검색하여 답변 생성

## 🛠️ 기술 스택

- **LLM**: OpenAI GPT-3.5-Turbo-Instruct
- **프레임워크**: LangChain
- **벡터 DB**: DeepLake
- **웹 스크래핑**: Selenium
- **임베딩**: OpenAI Embeddings
- **텍스트 처리**: LangChain CharacterTextSplitter
- **메모리 관리**: ConversationBufferMemory

## 📦 주요 의존성

```
langchain==0.1.20          # LLM 체인 프레임워크
langchain-community==0.0.38 # 커뮤니티 패키지
openai>=2.46.0             # OpenAI API
deeplake<4                 # 벡터 데이터베이스
selenium>=4.46.0           # 웹 스크래핑
python-dotenv>=1.2.2       # 환경 변수 관리
tiktoken>=0.13.0           # 토큰 카운팅
unstructured>=0.21.5       # 비정형 데이터 처리
```

## 🚀 설치 및 실행

### 1. 환경 설정

```bash
# Python 버전: 3.12 이상 필요
python --version

# 프로젝트 의존성 설치
pip install -r requirements.txt
# 또는 uv를 사용하는 경우:
uv sync
```

### 2. 환경 변수 설정

`.env` 파일을 프로젝트 루트에 생성하고 다음 정보를 추가합니다:

```env
OPENAI_API_KEY=your_openai_api_key_here
ACTIVELOOP_TOKEN=your_activeloop_token_here  # 클라우드 DB를 사용할 경우만 필요
```

### 3. 스크립트 실행

```bash
python main.py
```

## 📁 프로젝트 구조

```
langchain-chatbot/
├── main.py                 # 메인 애플리케이션 코드
├── README.md              # 프로젝트 문서
├── pyproject.toml         # 프로젝트 설정 (Python 3.12+)
├── uv.lock                # 의존성 잠금 파일 (uv 사용)
├── .env                   # 환경 변수 (실제 사용 시 필수)
├── .python-version        # Python 버전 지정 파일
└── .venv/                 # 가상 환경 (자동 생성)
```

## 🔄 작동 원리

### 1️⃣ 문서 로드 및 분할

```python
# Selenium을 사용하여 웹페이지에서 문서 로드
loader = SeleniumURLLoader(urls=ARTICLES)
docs_not_splitted = loader.load()

# 문서를 청크 단위로 분할 (1000 토큰, 200 토큰 중복)
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(docs_not_splitted)
```

### 2️⃣ 임베딩 및 벡터 DB 구축

```python
# OpenAI 임베딩 모델 사용
embeddings = OpenAIEmbeddings()

# DeepLake에 벡터 저장
db = DeepLake(dataset_path=dataset_path, embedding_function=embeddings)
db.add_documents(docs)  # 문서를 벡터로 변환하여 저장
```

### 3️⃣ 프롬프트 및 메모리 구성

```python
# 시스템 프롬프트 정의
template = """You are an exceptional customer support chatbot...
{chat_history}
{chunks_formatted}
Question: {input}
Answer:"""

# 대화 기록을 자동으로 관리
memory = ConversationBufferMemory(memory_key="chat_history", input_key="input")

# LLM 체인 생성
chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
```

### 4️⃣ 질의응답 처리

```python
def ask(query: str) -> str:
    # 사용자 질문과 유사한 문서 청크 검색 (유사도 검색)
    found = db.similarity_search(query)
    
    # 검색된 청크를 컨텍스트로 포맷
    chunks_formatted = "\n\n".join(doc.page_content for doc in found)
    
    # LLM에 프롬프트 입력 (chat_history는 메모리가 자동 처리)
    return chain.predict(input=query, chunks_formatted=chunks_formatted)
```

## ⚙️ 설정 옵션

`main.py`의 설정 섹션에서 다음을 조정할 수 있습니다:

| 설정 | 설명 | 기본값                  |
|------|------|----------------------|
| `USE_LOCAL_DB` | 로컬/클라우드 벡터 DB 사용 여부 | `True` (로컬)          |
| `ORG_ID` | Activeloop 네임스페이스 | `"example-org"`      |
| `DATASET_NAME` | 데이터셋 이름 | `"article_dataset"`  |
| `LOCAL_DB_PATH` | 로컬 DB 저장 경로 | `/db`                |
| `ARTICLES` | 데이터 소스 URL 목록 | 기술 기사 5개             |
| `chunk_size` | 텍스트 청크 크기 | `1000`               |
| `chunk_overlap` | 청크 사이의 중복 부분 | `200`                |
| `temperature` | LLM 창의성 (0=정확, 1=창의) | `0`                  |

## 💡 사용 예시

```python
# 첫 번째 질문
q1 = "How to check disk usage in linux?"
print("Q:", q1)
print("A:", ask(q1))

# 두 번째 질문 - 메모리가 대화 맥락을 유지합니다
q2 = "What was the 5th point about on the question how to remove spotify account?"
print("Q:", q2)
print("A:", ask(q2))
```

## ⚠️ 중요 주의사항

1. **문서 중복 저장 방지**: `db.add_documents(docs)` 는 매 실행마다 같은 문서를 중복으로 저장합니다. 
   - 한 번 실행 후 주석 처리하고, 이후에는 검색만 반복하세요.

2. **Chrome 브라우저 필요**: Selenium 기반 문서 로드를 위해 Chrome이 설치되어 있어야 합니다.

3. **API 키 보안**: `.env` 파일은 `.gitignore`에 추가하여 커밋되지 않도록 주의하세요.

4. **환경 변수**: 
   - `OPENAI_API_KEY`는 필수입니다.
   - `ACTIVELOOP_TOKEN`은 클라우드 DB(`USE_LOCAL_DB=False`)를 사용할 때만 필요합니다.

## 📊 데이터 소스

현재 프로젝트는 다음의 Digital Trends 기사를 학습 데이터로 사용합니다:

- Claude Sonnet vs GPT-4o 비교
- Apple Intelligence와 MacBooks
- OpenAI ChatGPT 사용 방법
- Character AI 사용 가이드
- PDF를 ChatGPT에 업로드하는 방법

## 🔧 문제 해결

### Selenium Chrome 드라이버 오류
- Chrome 브라우저가 설치되어 있는지 확인하세요.
- Selenium 4.6+는 Selenium Manager가 드라이버를 자동으로 다운로드합니다.

### OpenAI API 오류
- API 키가 올바르게 설정되었는지 확인하세요.
- API 사용량 및 가격책정을 확인하세요.

### 벡터 DB 오류
- `USE_LOCAL_DB=True`인 경우, 폴더 권한을 확인하세요.
- `USE_LOCAL_DB=False`인 경우, Activeloop 토큰과 Organization ID를 확인하세요.
