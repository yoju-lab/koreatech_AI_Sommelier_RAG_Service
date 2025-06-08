# Self-Consistency 프롬프트 엔지니어링 실습


난이도별(Easy/Medium/Hard) Self-Consistency 프롬프트와 LCEL 체이닝을 익히는 실습입니다.

이 노트북에서는 **Self-Consistency** 프롬프트 기법과 간단한 **검색-증강 생성**(Retrieval-Augmented Generation, RAG) 기법을 결합하여, 난이도가 다른 질의들(Easy/Medium/Hard)에 답하는 실습을 합니다. Self-Consistency 기법이란 한 질문에 대해 LLM이 여러 개의 답변을 생성하도록 한 다음, 그 중 가장 일관된 답을 최종적으로 선택하는 방법입니다. 이를 통해 답변의 신뢰성과 정확도를 높일 수 있습니다. 또한 RAG 기법으로서 미리 주어진 문서들에서 관련 정보를 검색하여 프롬프트에 포함함으로써, LLM이 보다 사실에 기반한 답변을 생성하도록 돕습니다. (여기서는 간단히 3개의 예시 문서를 이용해 검색을 흉내냅니다.)

이 실습에서는 3단계의 질의를 다룹니다. 우선 **공통 설정** 단계에서 환경 설정과 모델 초기화, 그리고 검색 함수와 Self-Consistency 파서를 정의합니다. 이후 **Easy, Medium, Hard** 난이도의 질의에 대해 각각:

* 관련 문서를 조회하여 프롬프트를 구성하고,
* LLM으로부터 여러 개의 초안 답변을 생성받은 뒤,
* 앞서 정의한 파서를 이용해 가장 일관된 최종 답변을 도출합니다.

마지막으로, 노트북에서는 **LangChain Expression Language (LCEL)** 체이닝 개념도 소개합니다. LCEL은 LangChain에서 체인을 선언적으로 구성하는 방법이지만, 이 노트북의 코드는 개별 함수를 순차적으로 호출하는 방식으로 체인을 구현하여 개념을 보여줍니다.

## 1. 공통 설정

이 코드 셀에서는 이후 단계에서 공통으로 사용할 환경 및 도구들을 설정합니다:

* **필요 라이브러리 임포트**: `os`, `dotenv`, LangChain의 `ChatOpenAI` 및 `PromptTemplate`, `numpy`, `pydantic`, `langchain.schema`의 `BaseOutputParser`, `langchain.embeddings`의 `OpenAIEmbeddings`, 그리고 `re`(정규표현식)을 임포트합니다. LangChain 라이브러리를 통해 LLM(대형 언어 모델)을 쉽게 사용할 수 있습니다.
* **환경 변수 로드**: `.env` 파일에서 OpenAI API 키와 모델명을 불러옵니다 (`OPENAI_API_KEY`, `OPENAI_LLM_MODEL`). 이렇게 하면 API 키 등을 코드에 직접 노출하지 않고도 설정할 수 있습니다.
* **LLM 초기화**: `ChatOpenAI` 클래스를 이용해 OpenAI 챗 모델을 초기화합니다. `model_name`에는 불러온 모델명을, `openai_api_key`에는 API 키를 전달합니다. `temperature=0.7`로 설정하여 생성의 무작위성을 약간 부여합니다(0에 가까울수록 determinisitc, 1에 가까울수록 더 창의적인 출력). 또한 `n=5`로 지정하여 한 번의 요청으로 답변 5개를 생성하도록 합니다. 이렇게 여러 개의 응답을 받는 것이 Self-Consistency 기법의 토대입니다.
* **간이 문서 데이터베이스 구성**: `docs` 딕셔너리에 파리에 관한 3개의 짧은 문서를 미리 저장합니다. 예를 들어 "에펠탑이 파리의 상징", "파리는 세느강을 따라 발달했고 루브르 박물관이 유명", "파리는 연간 2천만 관광객 방문" 등의 정보가 들어 있습니다. 이들은 LLM에 지식을 제공하기 위한 **외부 지식소스** 역할을 합니다.
* **Fake Retriever 함수 정의**: `fake_retriever(query, top_k)` 함수는 간단한 검색기 역할을 합니다. 실제로는 `docs` 딕셔너리에서 상위 `top_k`개의 문서를 반환하도록 구현되어 있습니다. (데모이므로 query를 활용한 실제 검색을 하진 않고, 그냥 사전의 처음 몇 문서를 준다고 볼 수 있습니다.) 이후 단계에서 사용자가 묻는 질문에 대해 관련된 문서를 찾는 용도로 이 함수를 사용할 것입니다. 이는 RAG의 "검색" 단계에 해당합니다.
* **RobustSelfConsistencyParser 클래스 정의**: 여러 개의 LLM 응답을 받아 **가장 일관된 최종 답변**을 선택하는 사용자 정의 파서입니다. LangChain의 `BaseOutputParser`를 상속하며, 내부에 임베딩 모델(`OpenAIEmbeddings`)을 사용합니다. 이 파서의 주요 동작은 다음과 같습니다:

  1. **정답 패턴 추출**: 각 생성된 응답에서 "최종 답변:"이라는 표현 뒤에 오는 문장을 추출합니다. 정규표현식을 사용해 대소문자나 형식(`**최종 답변**:`처럼 굵게 표시된 경우도 포함)과 무관하게 "최종 답변:" 키워드를 찾아 그 뒤의 텍스트를 가져옵니다. 이렇게 하면 모델 응답 중에서 실제 최종 답변 부분만 뽑아낼 수 있습니다. 만약 어떤 응답에도 이 패턴이 없다면(예: 모델이 형식을 지키지 않은 경우) 각 응답 전체를 후보 답변으로 사용합니다.
  2. **임베딩 벡터 계산**: 추출된 각 후보 답변 문장에 대해 사전 학습된 문장 임베딩(벡터 표현)을 계산합니다. `OpenAIEmbeddings.embed_documents` 메소드를 이용해 모든 답변 문장들을 벡터화합니다.
  3. **유사한 답변 군집화**: 답변 벡터들끼리 코사인 유사도를 비교하여 비슷한 답변들을 동일한 그룹으로 묶습니다. `threshold` 값(기본 0.9) 이상의 유사도를 보이면 같은 클러스터로 간주합니다. 구현상 먼저 첫 답변을 첫 클러스터 대표로 삼고, 다음 답변들이 기존 클러스터 대표들과 얼마나 유사한지 점검하여 충분히 유사하면 해당 클러스터에 추가하고, 아니면 새로운 클러스터를 만듭니다. 이렇게 하면 모델 응답들이 서로 내용상 대체로 일치하는지에 따라 몇 개의 그룹으로 나뉩니다.
  4. **최빈값(clustering) 답변 선택**: 가장 많은 응답이 모여있는 클러스터를 찾습니다. 즉 여러 생성 결과 중 가장 다수가 동의하는 내용의 답변 그룹을 선택하는 것입니다. 그 클러스터에 속한 답변들 중 대표로 첫 번째 것을 최종 답으로 반환합니다. (여러 답변이 비슷한 내용이면 그 중 아무거나 골라도 내용은 같을 것이므로 첫 번째를 선택합니다.) 이 방식으로 Self-Consistency, 즉 **여러 번 생성된 답변들 중 다수결로 가장 그럴듯한 답변을 얻는 효과**를 냅니다.

이러한 파서를 통해 만약 하나의 답만 빼고 나머지 네 개의 답이 서로 비슷하면, 그 네 개가 같은 클러스터로 묶여 최종 답변으로 선택되고, 혼자 동떨어진 한 개의 답변은 무시됩니다. 이를 통해 모델의 일회성 오류나 일관성 없는 답을 걸러낼 수 있습니다.

```python
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import numpy as np
from pydantic import PrivateAttr
from langchain.schema import BaseOutputParser
from langchain.embeddings import OpenAIEmbeddings
import numpy as np
import re

# .env 로드
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL")  # ex: "gpt-4o-mini"

# LLM 초기화: n개의 샘플 생성
llm = ChatOpenAI(
    model_name=OPENAI_LLM_MODEL,
    openai_api_key=OPENAI_API_KEY,
    temperature=0.7,
    n=5
)

# Fake Retriever: Top-3 문서 반환
docs = {
    "doc1": "파리의 상징은 에펠탑이며, 1889년에 세워졌습니다.",
    "doc2": "파리는 세느강을 따라 발달한 도시로, 루브르 박물관이 유명합니다.",
    "doc3": "파리는 연간 약 2천만 명의 관광객이 방문하는 세계적 관광 도시입니다."
}
def fake_retriever(query: str, top_k: int=3):
    return [docs[f"doc{i}"] for i in range(1, top_k+1)]

class RobustSelfConsistencyParser(BaseOutputParser):
    threshold: float = 0.9
    _embeddings: OpenAIEmbeddings = PrivateAttr()

    def __init__(self, *, threshold: float = 0.9):
        super().__init__(threshold=threshold)
        self._embeddings = OpenAIEmbeddings()

    def parse(self, generations: list[str]) -> str:
        # 1) 우선 “최종 답변” 패턴으로 뽑아보기
        answers = []
        pattern = re.compile(r"(?:\d+\.\s*)?\**최종\s*답변\**[:\s]*(.+)", re.IGNORECASE|re.DOTALL)
        for text in generations:
            m = pattern.search(text)
            if m:
                answers.append(m.group(1).strip())
        # 2) 패턴 매칭이 하나도 안 되면, raw generation 전체를 답변 후보로 사용
        if not answers:
            answers = [text.strip() for text in generations]

        # 3) 의미 기반 클러스터링
        embs = np.array(self._embeddings.embed_documents(answers))
        rep_embs, counts, clusters = [], [], []
        for ans, emb in zip(answers, embs):
            for idx, rep in enumerate(rep_embs):
                sim = np.dot(emb, rep)/(np.linalg.norm(emb)*np.linalg.norm(rep))
                if sim >= self.threshold:
                    counts[idx] += 1
                    clusters[idx].append(ans)
                    break
            else:
                rep_embs.append(emb); counts.append(1); clusters.append([ans])

        # 4) 최빈 클러스터의 대표 답변 반환
        best = int(np.argmax(counts))
        return clusters[best][0]
```

## 2. 난이도 하 (Easy)

* 질의 난이도: 단순 사실 묻기
* 예시: “파리의 상징은 무엇인가요?”

이제 쉬운(Easy) 난이도의 질의를 풀어봅니다. 질문은 "파리의 상징은 무엇인가요?"로, 단순한 사실 확인 유형입니다. 이를 해결하는 과정은 다음과 같습니다:

1. **질문 및 문서 검색**: `query` 변수에 질문 문장을 저장한 뒤, `fake_retriever(query, top_k=2)`를 호출하여 관련 문서 2개를 가져옵니다. 여기서는 `doc1`과 `doc2` 두 개의 문서가 반환되며, 내용은 각각 에펠탑(파리의 상징)과 세느강/루브르 박물관에 대한 정보입니다.
2. **프롬프트 생성**: `PromptTemplate`을 사용하여 LLM에 보낼 프롬프트를 만듭니다. 프롬프트 양식은 여러 줄의 문자열로 작성되었고, `{doc1}`, `{doc2}`, `{question}` 플레이스홀더를 포함합니다. 실제 내용이 채워진 프롬프트는 예를 들면 다음과 같습니다:

   ```text
   === 문서1 ===
   파리의 상징은 에펠탑이며, 1889년에 세워졌습니다.

   === 문서2 ===
   파리는 세느강을 따라 발달한 도시로, 루브르 박물관이 유명합니다.

   질문: 파리의 상징은 무엇인가요?

   **최종 답변**:
   ```

   문서 내용을 보여주고 질문을 제시한 뒤, "**최종 답변**:"이라고 명시함으로써 모델이 그 뒤에 최종 답을 작성하도록 유도합니다.
3. **프롬프트 포맷팅 및 메시지 변환**: `prompt_easy.format_prompt(...)`를 통해 위 템플릿의 플레이스홀더에 실제 `doc1`, `doc2` 텍스트와 질문을 채워 넣습니다. 그리고 `.to_messages()`를 호출하여 이 프롬프트를 대화 형식 메시지로 변환합니다. LangChain에서는 프롬프트를 바로 문자열로 보내기보다 내부적으로 `SystemMessage`나 `HumanMessage` 등의 형태로 취급하기 때문에 이런 변환을 거칩니다.
4. **LLM 응답 생성**: 준비된 `messages`를 `llm.generate([messages])`에 전달하여 OpenAI 모델로부터 답변을 생성합니다. 앞서 `n=5`로 모델을 초기화했으므로, 이 호출은 하나의 질문에 대해 서로 다른 5개의 응답을 생성합니다. `response.generations[0]`에는 이렇게 생성된 5개 답변 객체의 목록이 들어있습니다.
5. **응답 출력 및 확인**: 리스트 내의 각 Generation 객체에서 `.text` 속성을 추출하여 순수 텍스트 응답들 `raw_texts`를 얻습니다. `print`를 통해 `raw_texts`를 출력해보면 모델이 생성한 다섯 가지 답변을 확인할 수 있습니다. 이 경우 질문이 단순하고 문서에 답이 명확히 있었기 때문에 다섯 응답 모두 "파리의 상징은 에펠탑입니다."처럼 거의 동일한 답변이 나올 것으로 예상됩니다.
6. **Self-Consistency 파서 적용**: `RobustSelfConsistencyParser(threshold=0.85)` 인스턴스를 생성하여 앞서 얻은 `raw_texts`를 파싱합니다. 약간 낮춘 임계값 0.85를 준 것은 혹시 표현상의 사소한 차이(예: 마침표 여부 등)를 같다고 볼 수 있게 하기 위함입니다. 파서는 다섯 개의 답변이 모두 같은 의미임을 인지하고 그 중 하나를 최종 답으로 선택할 것입니다.
7. **최종 답변 출력**: 최종 선택된 답변을 출력합니다. Easy 단계의 결과는 아마도 "파리의 상징은 에펠탑입니다."로서, 모델의 모든 응답이 일치하여 그대로 반환된 경우입니다.

```python
from langchain import PromptTemplate

query = "파리의 상징은 무엇인가요?"
retrieved = fake_retriever(query, top_k=2)

# 1) Prompt → messages
prompt_easy = PromptTemplate(
    input_variables=["doc1","doc2","question"],
    template="""
=== 문서1 ===
{doc1}

=== 문서2 ===
{doc2}

질문: {question}

**최종 답변**:
"""
)

prompt_value = prompt_easy.format_prompt(
    doc1=retrieved[0],
    doc2=retrieved[1],
    question=query
)

messages = prompt_value.to_messages()  # ChatMessage 리스트

# 2) LLM으로 여러 답변 생성
response = llm.generate([messages])  
generations = response.generations[0]        # 5개의 Generation 객체

# 3) 파서에 raw text 리스트 전달
raw_texts = [gen.text for gen in generations]
print("Easy raw texts:", raw_texts)
parser = RobustSelfConsistencyParser(threshold=0.85)
final_answer = parser.parse(raw_texts)

print("Easy 최종 답변:", final_answer)
```

## 3. 난이도 중 (Medium)

* 질의 난이도: 비교·정리
* 예시: “파리의 주요 관광지 두 곳과 방문 시기를 알려주세요.”

다음은 중간(Medium) 난이도의 질문입니다. "파리의 주요 관광지 두 곳과 방문 시기를 알려주세요."처럼 두 가지 정보를 요구하는 비교·정리 유형 질문을 다룹니다. 처리 흐름은 비슷하지만 프롬프트 구성과 파서 동작에 몇 가지 변화가 있습니다:

1. **질문 설정 및 문서 검색**: `query`에 질문을 저장하고, `fake_retriever(query, top_k=3)`을 호출하여 관련된 문서 3개를 얻습니다. Easy 단계와 달리 이번에는 `doc1`, `doc2`, `doc3` 모두 사용됩니다. 이 문서들은 파리의 상징(에펠탑), 유명한 장소(루브르 박물관), 관광객 정보(연간 방문객 수) 등을 담고 있어, 질문에 필요한 단서를 제공합니다.
2. **프롬프트 템플릿 작성**: `PromptTemplate`을 이용해 프롬프트를 준비합니다. 이번 프롬프트는 답변 형식을 좀 더 엄격히 지정하기 위해 **지시사항**을 포함하고 있습니다. 템플릿 내용을 살펴보면:

   * "아래 문서를 참고하여 **한 번에 하나의 샘플**을 생성하세요."라는 문구로 시작합니다. 이는 모델에게 한 번에 하나의 답변만 생성하라고 지시하는 부분입니다. (만약 명시하지 않으면 모델이 여러 답을 나열하려고 할 수도 있으므로, Self-Consistency 기법에서는 이런 지시를 줘서 한 응답에는 하나의 해결책만 제시하도록 유도합니다.)
   * 문서1, 문서2, 문서3의 내용과 질문이 순서대로 나열됩니다.
   * `[지시사항]` 섹션에서 출력 형식 규칙을 제시합니다:

     * "**최종 답변**: <답변 문장>" 형태로 한 문장의 답을 작성할 것.
     * 답변에 질문 문구를 반복하지 마세요.
     * 부가 설명, 번호 없이 딱 한 줄로만 작성하세요.
       이러한 지침들은 모델이 일관된 형식의 정답만 내놓도록 돕습니다.
3. **프롬프트 채우기 및 메시지 변환**: `prompt_med.format_prompt(...)`로 템플릿에 실제 문서 내용과 질문을 채워 넣고, `.to_messages()`로 LangChain 메시지 객체로 변환합니다. 이때까지의 과정은 Easy와 동일합니다.
4. **LLM 다중 응답 생성**: `llm.generate([messages])`를 호출하여 5개의 답변 초안을 생성합니다. 질문이 조금 더 복잡하므로 모델은 문서 정보를 조합하거나 상식을 활용해 다양한 답변을 낼 수 있습니다. 예를 들어 어떤 답은 "에펠탑과 루브르 박물관 - 연중 언제든지 방문 가능"이라고 할 수 있고, 다른 답은 "에펠탑과 루브르 박물관 - 가장 좋은 시기는 봄과 가을"이라고 할 수도 있습니다. 이렇게 약간씩 다른 5개의 문장이 생성됩니다.
5. **원시 응답 확인**: `raw_texts_med` 리스트로 5개 응답 텍스트를 모아 출력해봅니다. 여러 응답을 비교해보면 대다수 답변들이 공통적으로 에펠탑과 루브르 박물관을 언급하더라도, 방문 시기에 대해 표현이 다를 수 있습니다 (일부는 특정 계절을 추천하고, 일부는 "항상 방문 가능"처럼 언급할 수 있습니다).
6. **Self-Consistency 파싱**: `RobustSelfConsistencyParser(threshold=0.9)`를 이용해 이 답변들을 분석합니다. 임계값을 0.9로 높게 준 것은 답변 문장이 거의 동일한 경우에만 같은 클러스터로 인정하도록 하기 위함입니다. 파서는 답변들의 의미적 유사도를 계산하고, 내용이 거의 동일한 답변들끼리 묶습니다. 예를 들어 5개 중 4개의 답변이 "연중 내내 방문할 수 있다"는 취지로 비슷하고 1개만 "봄가을이 좋다"고 다르면, 전자의 4개가 한 그룹이 되어 다수파를 이룹니다.
7. **최종 답 선택 및 출력**: 가장 많은 답변이 모인 그룹의 대표 답변을 최종 결과로 선정합니다. 위 예시의 경우 "에펠탑과 루브르 박물관은 파리의 주요 관광지로, 연중 언제든지 방문할 수 있습니다."와 같은 문장이 최종 답으로 선택될 것입니다. 즉, 모델 응답들 중 다수가 동의하는 내용을 답변으로 확정한 것입니다.

```python
query = "파리의 주요 관광지 두 곳과 방문 시기를 알려주세요."
retrieved = fake_retriever(query, top_k=3)

from langchain import PromptTemplate

# 1) Prompt → messages
prompt_med = PromptTemplate(
    input_variables=["doc1","doc2","doc3","question"],
    template="""
아래 문서를 참고하여 **한 번에 하나의 샘플**을 생성하세요.

=== 문서1 ===
{doc1}

=== 문서2 ===
{doc2}

=== 문서3 ===
{doc3}

질문: {question}

[지시사항]
- **최종 답변**: <답변 문장>
- 질문 문구를 반복하지 마세요.
- 부가 설명, 번호 없이 딱 한 줄로만 작성하세요.
"""
)

prompt_value = prompt_med.format_prompt(
    doc1=retrieved[0],
    doc2=retrieved[1],
    doc3=retrieved[2],
    question="파리의 주요 관광지 두 곳과 방문 시기를 알려주세요."
)

messages = prompt_value.to_messages()

# 2) LLM으로 여러 답변 생성
response = llm.generate([messages])
generations = response.generations[0]  # 5개의 Generation 객체

# 3) raw text 리스트 준비
raw_texts_med = [gen.text for gen in generations]

print(f"raw_texts_med: {raw_texts_med}")

# 4) 파싱
parser = RobustSelfConsistencyParser(threshold=0.9)
final_med = parser.parse(raw_texts_med)
print("Medium 최종 답변:", final_med)
```

## 4. 난이도 상 (Hard)

* 질의 난이도: 복합 추론·추천
* 예시: “파리의 역사, 관광지, 방문 시기를 종합해 3일 여행 일정을 추천해주세요.”

마지막으로 어려운(Hard) 난이도의 질문입니다. "파리의 역사, 관광지, 방문 시기를 종합해 3일 여행 일정을 추천해주세요."와 같이 복합적인 요구사항(역사적인 맥락 + 관광지 + 최적 방문시기)을 모두 반영하여 창의적인 결과를 내야 하는 질문입니다. 코드는 다음과 같이 진행됩니다:

1. **질문 및 문서 준비**: `query`에 질문을 설정하고 `fake_retriever(query, top_k=3)`으로 3개의 문서를 모두 가져옵니다. 이전과 동일하게 파리 관련 문서들이 맥락으로 제공됩니다만, 이번 질문은 단순 사실 조회를 넘어 여러 정보를 종합해야 하기 때문에, 문서 내용(에펠탑, 루브르 박물관, 관광객 정보 등)을 토대로 모델이 새로운 계획을 만들어내야 합니다.
2. **프롬프트 생성 (3일 일정 추천)**: `PromptTemplate`으로 프롬프트를 작성합니다. 이번에도 "**한 번에 하나의** 3일 일정 추천 샘플을 생성하세요."라고 하여 한 번 호출에 하나의 일정만 만들도록 지시하고 있습니다. 이어서 문서1, 문서2, 문서3와 질문이 주어지고, `[지시사항]`에서 답안 형식을 엄격히 규정합니다:

   * "**최종 추천 일정**: <3일 일정 요약 한 줄>" 형식으로 답안을 작성해야 합니다.
   * 출력에 불필요한 제목이나 형식을 넣지 말고 ("### 일정 추천 샘플" 등의 표제 없이) 한 줄 요약만 제시하라고 강조합니다.
     이러한 지침은 모델이 장황하게 일정을 나열하지 않고, 3일치 계획을 한 문장으로 깔끔하게 요약하도록 유도합니다.
3. **프롬프트 채우기 및 메시지 변환**: 문서들과 질문을 템플릿에 채워 넣고 `.to_messages()`로 변환합니다 (이 과정은 이전과 동일합니다).
4. **LLM 응답 생성**: LLM에 프롬프트를 보내 5개의 여행 일정 초안을 만들어냅니다. 이 질문은 답이 정해져 있지 않고 창의적이어야 하므로, 모델이 내놓는 5가지 일정안이 서로 세부 내용이나 표현 면에서 꽤 다를 수 있습니다. 예를 들어 어떤 답은 "3일 동안 에펠탑과 루브르 박물관을 방문하고 세느강을 따라 산책하며 파리의 역사와 문화를 체험하세요."처럼 서술형으로 제안할 수 있고, 다른 답은 "1일차: 에펠탑, 2일차: 루브르, 3일차: 세느강"처럼 요약할 수도 있습니다. 각 답변은 파리의 주요 명소들을 포함하지만 표현이나 강조점에서 다양성이 있을 것입니다.
5. **생성된 일정 확인**: `raw_texts_hard` 리스트로 5개 답변 텍스트를 모아서 출력해보고, 서로 어떻게 다른지 살펴볼 수 있습니다. 대부분의 답변이 에펠탑과 루브르 박물관 방문을 포함하고 있을 것이고, 아마 세느강 산책이나 유람 같은 공통 요소도 있을 것입니다. 하지만 문장의 구조나 추천 방식(서술형 문장 vs. 목록 나열 등)은 응답마다 다를 수 있습니다.
6. **Self-Consistency로 일관된 일정 선택**: 앞 단계와 마찬가지로 `parser.parse(raw_texts_hard)`를 호출하여 가장 일관된 일정을 선택합니다. 임베딩 유사도 임계값 0.9를 적용하여 매우 비슷한 표현의 답변들만 같은 그룹으로 묶습니다. 5개의 답변이 있다면 두세 개 정도는 서로 흡사한 어조로 핵심 명소들을 언급했을 수 있고, 나머지는 형태가 달라 개별 그룹이 될 수 있습니다. 이때 가장 큰 그룹(예: 두 개 이상 답변이 매우 유사한 그룹)이 찾는다면 그 그룹의 답을 선택할 것입니다. 만약 모든 답변이 제각각이면 다수가 동의하는 답이 없으므로 그 중 첫 번째 것을 반환할 수도 있습니다 (하지만 보통 중요한 요소들은 겹치기 마련입니다).
7. **최종 추천 일정 출력**: 파서를 거쳐 확정된 최종 3일 일정 추천을 출력합니다. 예를 들어 여러 답변에 공통으로 등장한 "에펠탑과 루브르 박물관 방문, 세느강 산책" 등이 포함된 문장이 선택될 가능성이 높습니다. Self-Consistency 기법을 통해 모델의 다양한 아이디어 중 핵심이 되는 공통 요소를 가진 답이 최종 결과로 제공됩니다.

```python
query = "파리의 역사, 관광지, 방문 시기를 종합해 3일 여행 일정을 추천해주세요."
retrieved = fake_retriever(query, top_k=3)

# 1) Prompt → messages

prompt_hard_single = PromptTemplate(
    input_variables=["doc1","doc2","doc3","question"],
    template="""
아래 문서를 참고하여 **한 번에 하나의** 3일 일정 추천 샘플을 생성하세요.

=== 문서1 ===
{doc1}

=== 문서2 ===
{doc2}

=== 문서3 ===
{doc3}

질문: {question}

[지시사항]
- **최종 추천 일정**: <3일 일정 요약 한 줄>
- “### 일정 추천 샘플” 제목 없이, 한 줄짜리 요약만 작성하세요.
"""
)

prompt_value = prompt_hard_single.format_prompt(
    doc1=retrieved[0],
    doc2=retrieved[1],
    doc3=retrieved[2],
    question="파리의 역사, 관광지, 방문 시기를 종합해 3일 여행 일정을 추천해주세요."
)

messages = prompt_value.to_messages()

# 2) LLM으로 여러 답변 생성
response = llm.generate([messages])
generations = response.generations[0]

# 3) raw text 리스트 준비
raw_texts_hard = [gen.text for gen in generations]
print("Hard raw texts:", raw_texts_hard)

# 4) 파싱
final_hard = parser.parse(raw_texts_hard)
print("Hard 최종 답변:", final_hard)
```
