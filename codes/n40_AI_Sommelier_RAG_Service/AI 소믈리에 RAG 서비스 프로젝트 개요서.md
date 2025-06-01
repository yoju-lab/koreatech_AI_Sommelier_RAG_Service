# 프로젝트 개요서: AI 소믈리에 RAG 서비스

---

## 1. 프로젝트 개요

* **프로젝트명:** AI 소믈리에 RAG 서비스
* **목적:**
  음식 사진(이미지)와 한 줄 설명을 입력받아,
  AI가 음식의 특징을 분석하고, 대규모 와인 리뷰 데이터베이스(벡터DB, Pinecone)에서 음식과 어울리는 와인 정보를 검색(RAG, Retrieval-Augmented Generation)한 뒤
  LLM(GPT-4o-mini)로 자연어 추천 결과를 생성해 사용자에게 제공하는 웹 서비스 구축
* **기대효과:**

  * 초개인화 와인 추천 경험 제공
  * LLM+RAG 기술(이미지→설명→벡터검색→생성) 전체 파이프라인 실습
  * 최신 생성형 AI/검색 증강/멀티모달 AI 실전 체험

---

## 2. 서비스 시나리오

1. 사용자는 음식 사진을 업로드하고 한 줄 설명을 입력
2. AI가 이미지를 분석(vision LLM: GPT-4o-mini)해 음식의 맛과 특징을 한 문장으로 요약
3. 음식 설명을 임베딩으로 변환해 Pinecone DB에서 유사한 와인 리뷰를 top-K로 검색
4. 검색된 리뷰와 음식 정보를 LLM(GPT-4o-mini)에 전달
5. LLM이 최적의 와인을 한글로 추천하고 그 이유를 자연어로 설명
6. UI에 추천 와인, 이유, 검색된 리뷰와 유사도 등 표시

---

## 3. 시스템 아키텍처

* **Frontend/UI:**

  * Streamlit 기반 웹 UI
  * 이미지 업로드, 설명 입력, 결과 표시(리뷰, 유사도, 상세 추천)
* **Backend 핵심 모듈:**

  * sommelier.py: 이미지 분석, 벡터DB 검색, LLM 프롬프트 처리
* **외부 서비스:**

  * OpenAI API: GPT-4o-mini (LLM), text-embedding-3-small (임베딩)
  * Pinecone: 벡터 데이터베이스(와인 리뷰 임베딩 저장/검색)
* **환경설정:**

  * 환경 변수는 .env 파일로 관리

---

## 4. 주요 사용 기술 및 환경

* **프로그래밍 언어:** Python 3.12+
* **주요 라이브러리:**

  * openai, langchain\_openai, langchain\_pinecone, pinecone, streamlit, python-dotenv, base64 등
* **모델:**

  * 텍스트 임베딩: text-embedding-3-small
  * LLM: gpt-4o-mini
* **벡터DB:**

  * Pinecone (us-east1-aws, cosine metric, dimension=1536)

---

## 5. 환경변수 및 설정 파일 예시

**`.env` 파일 예시**

```
OPENAI_API_KEY=sk-...         # OpenAI 개인키
OPENAI_LLM_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
PINECONE_API_KEY=pcsk_...
PINECONE_ENVIRONMENT=us-east1-aws
PINECONE_INDEX_REGION=us-east1
PINECONE_INDEX_CLOUD=aws
PINECONE_INDEX_NAME=wine-reviews
PINECONE_INDEX_DIMENSION=1536
PINECONE_INDEX_METRIC=cosine
```

※ 실제 환경에서는 API 키를 안전하게 보관하고, .env 파일은 절대 공개 저장소에 올리지 않습니다.

---

## 6. 핵심 코드 구조

### (1) Streamlit app.py

* 두 개 컬럼(입력/출력, 이미지 미리보기) 구성
* 1단계: 이미지/설명 → 맛 분석(LLM Vision)
* 2단계: 벡터DB에서 유사한 와인 리뷰(top-K, 유사도 점수 포함) 검색
* 3단계: LLM을 통한 와인 상세 추천 결과 생성 및 출력

### (2) sommelier.py

* 환경 변수 로드 및 LLM/임베딩/벡터스토어 초기화
* **describe\_dish\_flavor:**

  * 입력 이미지(base64)와 프롬프트로 음식 맛 설명(LLM Vision)
* **search\_wine, search\_wine\_with\_score:**

  * 음식 설명으로 Pinecone에서 유사한 와인 리뷰 및 유사도 점수 검색
* **recommand\_wine:**

  * 음식 설명과 검색 리뷰를 context로 하여 GPT-4o-mini에 페어링 추천 프롬프트 작성 및 호출

---

## 7. 서비스 흐름 예시

1. 사용자가 음식 이미지를 업로드하고 “이 요리에 어울리는 와인을 추천해주세요” 입력
2. 시스템이 이미지를 Vision LLM 프롬프트에 포함시켜 맛/풍미 한 문장 생성
3. 이 설명으로 Pinecone에 유사도 검색 → top-2 리뷰와 유사도 표시
4. 이 결과와 함께 LLM에 “한글로 추천, 이유 설명” 프롬프트로 최종 추천 생성
5. Streamlit UI에 분석 결과, 리뷰 목록, 상세 추천 결과 단계별 표시

---

## 8. 한계 및 확장 가능성

* 실제 리뷰DB, LLM 모델, 임베딩 품질에 따라 추천 품질이 좌우됨
* **확장 가능:**

  * 추천 결과에 와인 라벨 이미지 추가
  * 사용자의 피드백/선호 학습 반영
  * 다양한 음식/와인 DB 추가 등

---

## 9. 기타 운영/보안

* API Key, 데이터는 외부 노출 금지
* 모델 사용량/비용 및 Pinecone 쿼리 비용 관리 필요

---

