# RAG와 Tree of Thought (ToT) 기법 활용 실습


이 코드 셀에서는 RAG와 Tree of Thought (ToT) 기법 활용을 위한 **LangChain** 라이브러리의 주요 모듈들을 불러옵니다. 이후 코드에서 사용할 핵심 도구들을 임포트하며, 각 도구의 역할은 다음과 같습니다:

* **ChatOpenAI** – OpenAI의 대규모 언어 모델(LLM)을 사용해 챗봇 형태로 답변을 생성할 수 있게 해주는 LangChain 래퍼입니다. (예: GPT-4 모델을 챗 API로 호출)
* **PromptTemplate** – LLM에 전달할 프롬프트(질의 및 컨텍스트)를 쉽게 구성하기 위한 템플릿 클래스입니다. `{}` 플레이스홀더에 값을 채워 자동으로 프롬프트 문자열을 만들어 줍니다.
* **RegexParser** – LLM의 출력에서 **정규표현식**을 통해 원하는 패턴의 문자열을 추출할 때 사용하는 파서입니다. (예를 들어, 응답에서 숫자만 추출하거나 특정 형식만 가져올 때 유용)
* **JsonOutputParser** – LLM의 출력이 JSON 형식일 때 이를 Python 딕셔너리나 리스트 등의 구조화된 데이터로 변환해주는 파서입니다. 모델의 응답을 손쉽게 다룰 수 있도록 도와줍니다.

마지막으로 `print('✅ Imports OK')`를 실행하여 필요한 라이브러리들이 문제없이 임포트되었는지 확인합니다. 이 줄을 실행하면 \*\*"✅ Imports OK"\*\*라는 메시지가 출력되어 준비가 완료되었음을 알려줍니다.

```python
from langchain_openai import ChatOpenAI  # OpenAI LLM wrapper
from langchain.prompts import PromptTemplate  # PromptTemplate
from langchain.output_parsers.regex import RegexParser  # RegexParser
from langchain_core.output_parsers.json import JsonOutputParser  # JsonOutputParser

print('✅ Imports OK')
```

이 코드 셀에서는 **예시 문서 데이터**를 준비합니다. `docs_str` 변수에 여러 줄로 이루어진 문자열을 저장하여, RAG와 ToT 개념 및 LangChain 사용법에 대한 세 개의 \*\*문서(Document)\*\*를 정의합니다. 이후 단계에서 LLM에 정보를 제공하기 위한 컨텍스트로 활용될 것입니다 (실제 RAG 시스템에서는 벡터 DB에서 문서를 검색하지만, 여기서는 예시로 미리 문서를 넣습니다).

`docs_str`에 포함된 문서들은 다음과 같습니다:

1. **Doc1: RAG 파이프라인 개요 및 구성 요소** – **검색 증강 생성**(RAG, *Retrieval-Augmented Generation*) 파이프라인에 대한 설명입니다. RAG는 LLM이 외부 지식과 함께 답변을 생성하도록 하는 기법으로, 문서에는 RAG의 **5가지 구성 요소**가 정리되어 있습니다:

   * *Retriever* : 사용자 질문과 유사한 내용을 벡터 DB 등에서 검색하여 **관련 문서**들을 상위 k개 찾는 단계
   * *Document Combiner* : 검색된 문서들을 하나의 컨텍스트로 **병합하거나 요약**하는 단계
   * *Prompt Constructor* : 결합된 컨텍스트와 사용자 질문을 합쳐 LLM에 보낼 **최종 프롬프트**를 만드는 단계
   * *Generator (LLM)* : 최종 프롬프트를 입력으로 받아 **답변을 생성**하는 단계 (실제로 LLM이 답을 만드는 부분)
   * *Post-processing* : LLM이 생성한 답변을 검토하여 **불필요한 정보 제거**, **포맷 정리** 또는 **응답의 신뢰성 검증** 등을 수행하는 단계

2. **Doc2: Tree of Thought(ToT) 기법 개념과 응용** – **Tree of Thought (사고의 나무)** 기법에 대한 설명입니다. ToT는 한 번에 하나의 최종 답만 생성하는 대신, 모델이 \*\*여러 갈래의 사고 경로(branch)\*\*를 탐색하도록 유도하는 **프롬프트 엔지니어링 기법**입니다. 문서에는 ToT의 핵심 개념이 포함되어 있습니다:

   * *사고 노드(Node)* : 문제를 풀기 위한 중간 생각 또는 가설을 나타내는 단계들
   * *분기(Branching)* : 서로 다른 접근 방법이나 가설로 **생각을 분岐**하여 다양한 경로를 탐색하는 것
   * *선택(Selection)* : 각 경로의 타당성을 평가하여 **가장 유망한 사고 경로를 선택**하는 과정

   ToT 기법은 **복잡한 문제 해결**에 활용될 수 있으며, 예를 들어 수학 문제를 풀 때 여러 풀이 과정을 시도하거나, RAG 시스템에서 **다중 문서 간 일관성**을 검토할 때 유용합니다. 여러 경로의 중간 결과(예: 문서 이해 → 핵심 문장 추출 → 문장 조합)를 비교한 뒤 **가장 신뢰도 높은 답변**을 얻을 수 있다는 장점이 소개되어 있습니다.

3. **Doc3: LangChain 0.3 LCEL 스타일 Runnable 체이닝** – LangChain 라이브러리 **v0.3**에서 새롭게 도입된 **LCEL(Language Chain Execution Layer)** 개념과 **Runnable** 체이닝에 대한 설명입니다. 여기서는 LangChain에서 제공하는 실행 단위들이 소개됩니다:

   * *PromptRunnable*: `PromptTemplate`을 사용해 프롬프트를 생성하는 실행 단계
   * *LLMRunnable*: LLM을 호출하여 응답을 얻는 실행 단계
   * *ParserRunnable*: LLM의 텍스트 출력을 JSON이나 정규표현식 등을 이용해 **파싱하여 구조화된 데이터로 변환**하는 실행 단계

   이러한 Runnable들은 파이프(`|`) 연산자를 통해 **체인처럼 연결**할 수 있습니다. 문서의 예시 코드에서는 `chain = PromptTemplate(...) | ChatOpenAI(...) | RegexParser(...)` 형태로, 프롬프트 생성 → LLM 호출 → 출력 파싱 단계가 **한 줄의 체인으로 구성**되는 모습을 보여줍니다. 이처럼 LCEL을 활용하면 필요한 경우 중간에 `JsonOutputParser`나 커스텀 툴을 삽입하는 등, **유연하고 선언적인 체인 구성**이 가능해집니다.

위와 같이 정의된 `docs_str` 문자열은 이후 실습 단계에서 **LLM에 전달될 컨텍스트**로 사용됩니다. 요약하면, 검색 엔진을 통해 가져온 문서 대신 미리 준비된 예시 문서를 활용하여 RAG + ToT의 동작을 실습하려는 것입니다.

````python
docs_str = '''
Doc1: RAG 파이프라인 개요 및 구성 요소
Retrieval-Augmented Generation(RAG) 파이프라인은 대규모 언어 모델(LLM)에 외부 지식을 실시간으로 결합하여 응답의 정확도와 신뢰성을 높이는 방법론입니다.
1. Retriever: 벡터 DB(예: Qdrant, Pinecone)에서 사용자 질의와 유사도가 높은 문서(top-k)를 검색
2. Document Combiner: 검색된 문서를 하나의 컨텍스트 블록으로 병합하거나 요약
3. Prompt Constructor: 합쳐진 컨텍스트와 사용자 질의를 결합해 LLM에 보낼 프롬프트 생성
4. Generator (LLM): 최종 프롬프트를 입력으로 받아 답변 생성
5. Post-processing: 불필요 정보 제거, 포맷팅, 생산된 답변 검증

Doc2: Tree of Thought(ToT) 기법 개념과 응용
Tree of Thought(ToT)는 한 번에 단일 답변을 생성하는 대신, 모델이 여러 개의 중간 추론 경로(branch)를 탐색해 최적의 사고 흐름을 선택하도록 유도하는 프롬프트 엔지니어링 기법입니다.
- 사고 노드(Node): 중간 추론 단계별로 핵심 개념이나 가설을 생성
- 분기(Branching): 서로 다른 가설 또는 접근 방식을 분리하여 평가
- 선택(Selection): 각 분기의 타당성을 비교한 뒤, 가장 적합한 사고 경로 선택
응용 사례로는 복잡한 수학 문제 풀이, 다단계 논증 생성, RAG 시스템에서 다중 문서 간 일관성 검사 등이 있으며, 특히 RAG와 결합 시 “문서 이해 → 핵심 문장 추출 → 문장 조합 → 최종 답변”의 여러 경로를 실험해 가장 신뢰도 높은 결과를 얻을 수 있습니다.

Doc3: LangChain 0.3 LCEL 스타일 Runnable 체이닝
LangChain 0.3 버전에서는 LCEL(Language Chain Execution Layer)라는 새로운 Runnable 추상화를 도입했습니다. 주요 특징은 다음과 같습니다.
1. PromptRunnable: `PromptTemplate` 객체에 따라 프롬프트를 생성하는 단계
2. LLMRunnable: 생성된 프롬프트를 LLM에 전달하여 응답을 받아오는 단계
3. ParserRunnable: LLM 출력물을 파싱(예: JSON, Regex)해 구조화된 데이터로 변환
이들 Runnable은 파이프라인 연산자(`|`)로 체이닝할 수 있어, 예를 들어:
```python
chain = PromptTemplate(...) | ChatOpenAI(...) | RegexParser(...)
````

와 같이 선언만으로도 “프롬프트 생성 → LLM 호출 → 파싱” 흐름이 자동 구성됩니다. 필요 시 JsonOutputParser나 CustomTool을 중간에 삽입해 확장할 수 있습니다.
'''

````

## 난이도 하 (기초 단계)  
- PromptTemplate와 LLM 호출 결과를 직접 출력합니다.

이 코드 셀에서는 앞서 정의한 문서들을 **컨텍스트로 포함한 프롬프트**를 생성하고, 이를 LLM에 보내어 **사용자 질문에 대한 답변을 얻은 뒤 출력**합니다. 이는 RAG 파이프라인의 가장 기본적인 형태로서, **검색된 문서를 컨텍스트로 주고 한 번의 LLM 호출로 답변을 생성**하는 단계입니다. 구체적인 동작은 다음과 같습니다:

- **프롬프트 구성**: `PromptTemplate` 객체 `template_low`에 `docs`와 `query` 변수를 채워 넣어 최종 프롬프트 문자열을 만듭니다. 프롬프트에는 미리 준비된 문서(`{docs}`)와 사용자 질문(`{query}`)이 포함되며, 마지막에 **"간단히 ToT 적용 이점을 설명하세요."**라는 지시가 붙어 있습니다.  
- **LLM 응답 생성**: `ChatOpenAI` 객체 `llm_low`를 통해 위 프롬프트를 **GPT 모델**에 전달하고 응답을 생성합니다. 여기서는 temperature=0으로 설정하여 **결과의 일관성**을 높였습니다. (`gpt-4o-mini` 모델은 예시로 사용된 작은 GPT-4 모델입니다.)  
- **결과 출력**: LLM이 반환한 응답 객체인 `result_low`를 `print`로 출력하여, 모델이 생성한 답변 내용을 확인합니다. 이 답변에는 **"RAG에서 ToT를 적용하면 어떤 이점이 있는가?"**에 대한 설명이 담겨 있으며, 예를 들어 *다양한 추론 경로 탐색*, *문서 간 일관성 향상*, *복잡한 문제 해결 능력 제고* 등 **ToT 적용의 이점**을 나열하는 형식으로 나올 것입니다.

```python
template_low = PromptTemplate(
    input_variables=['docs','query'],
    template="""
문서:
{docs}

질문: {query}

간단히 ToT 적용 이점을 설명하세요.
"""
)
llm_low = ChatOpenAI(model='gpt-4o-mini', temperature=0)
result_low = (template_low | llm_low).invoke({'docs': docs_str, 'query': 'RAG에서 ToT 적용 이점은 무엇인가요?'})
print(result_low)
````

## 난이도 중 (응용 단계)

* Few-shot 예시와 JsonOutputParser를 사용해 JSON 파싱을 수행합니다.

이 코드 셀에서는 **Few-shot 예시**를 프롬프트에 포함하고 **JSON 형식으로 출력**하도록 유도하여, 모델의 응답을 구조화된 형태로 받아봅니다. 이전 단계보다 응용된 프롬프트 기법과 출력 처리가 사용되며, 흐름은 다음과 같습니다:

* **예시 제공(Few-shot)**: 프롬프트 맨 앞에 `few_shot` 변수에 저장된 **Q\&A 예시**를 넣습니다. 이 예시는 \*"질문: RAG에 ToT 적용 시 이점?"\*에 대한 모범 답안 형태로, **중간 사고**와 **최종 답변**의 예를 보여줍니다. 이를 통해 모델이 **원하는 답변 형식**을 학습하도록 합니다. (Few-shot 학습: 몇 가지 예시를 프롬프트에 제시하여 모델이 유사한 형식을 따르도록 만드는 기법입니다.)
* **JSON 형태 답변 유도**: `PromptTemplate`인 `template_med`에서 사용자 질문과 문서를 제공한 후, **JSON 배열 형식의 출력**만 생성하라는 지시를 합니다. 프롬프트 예시에 아래와 같은 목표 형식을 명시합니다:

  ```json
  [
    {"step": "중간 사고", "content": "..."},
    {"step": "최종 답변", "content": "..."}
  ]
  ```

  이처럼 출력 형식을 못박아 둠으로써, LLM이 답변을 **구조화된 JSON 형태**로 내놓도록 유도합니다.
* **LLM 호출 및 파서 적용**: `ChatOpenAI` 객체 `llm_med`를 temperature=0.2로 설정해 LLM을 호출합니다 (조금의 무작위성을 허용). 모델은 주어진 예시와 지시에 따라 **JSON 형태의 응답**을 생성할 것입니다. 그 다음 `JsonOutputParser`인 `parser_med`가 동작하여, LLM의 텍스트 출력을 바로 **Python 리스트** 등 **구조화된 데이터로 파싱**합니다. LangChain의 `|` 체이닝 연산자를 활용하여 프롬프트 → LLM → JSON 파서를 **연속 실행**한 점에 유의하세요. (`(template_med | llm_med | parser_med).invoke(...)` 형태로 한 번에 처리합니다.)
* **구조화된 결과 출력**: 파서가 반환한 `result_med`를 출력하여 **구조화된 응답 결과**를 확인합니다. 출력 결과는 파이썬의 리스트 형태로, 각 원소가 딕셔너리 `{"step": ..., "content": ...}`로 표현됩니다. 예를 들어:

  ```python
  [ {'step': '중간 사고', 'content': '문서 이해 → 핵심 문장 추출 → 문장 조합'},
    {'step': '최종 답변', 'content': '여러 경로를 실험해 가장 신뢰도 높은 결과를 얻습니다.'} ]
  ```

  와 같은 형태로, **첫 번째 항목**은 문서를 이용한 중간 사고 과정에 대한 내용, **두 번째 항목**은 최종 답변 내용을 담고 있습니다. 이렇게 JSON으로 구조화함으로써, 응답 내용을 후처리하거나 다른 코드에서 활용하기가 훨씬 수월해집니다.

```python
few_shot = '''
예시)
질문: RAG에 ToT 적용 시 이점?
- 중간 사고: 문서 요약 → 핵심 개념 추출
- 최종 답변: 중간 단계를 통해 정확도를 높입니다.
'''  

template_med = PromptTemplate(
    input_variables=['few_shot','docs','query'],
    template="""
{few_shot}

문서:
{docs}

질문: {query}

아래 JSON 배열만 순수하게 출력하세요:
[
  {{"step":"중간 사고","content":"..."}},
  {{"step":"최종 답변","content":"..."}}
]
"""
)
llm_med = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
parser_med = JsonOutputParser()
result_med = (template_med | llm_med | parser_med).invoke(
    {'few_shot': few_shot, 'docs': docs_str, 'query': 'RAG ToT 적용 방법은?'}
)
print(result_med)
```

## 난이도 상 (심화 단계)

* 중간 사고 생성 → 요약 → 최종 답변 흐름을 체이닝합니다.

이 코드 셀에서는 **복잡한 체이닝 기법**을 활용하여, 하나의 질문에 대해 LLM이 곧바로 답변하도록 하는 대신 **여러 단계의 추론 과정을 거쳐 최종 답변을 생성**합니다. 구체적으로 세 번의 `PromptTemplate`와 LLM 호출을 \*\*연쇄(Chaining)\*\*하여 \*\*“중간 사고 생성 → 요약 → 최종 답변”\*\*의 3단계 파이프라인을 구현합니다. 각 단계의 역할은 다음과 같습니다:

1. **중간 사고 생성 단계** – 첫 번째 프롬프트 `tmpl1`은 `{docs}`와 `{query}`를 입력으로 받아, ToT 기법을 적용한 \*\*중간 사고(process)\*\*를 요청합니다. 프롬프트 내용에는:

   * **1) 중간 사고: 핵심 개념 요약**,
   * **2) 중간 사고: 비교 분석**,
   * **3) 선택 이유 설명**
     이라는 세 가지 항목이 명시되어 있습니다. 이는 모델에게 **질문에 답하기 전에 필요한 사고 과정**을 세 부분으로 나누어 작성하도록 지시하는 것입니다. 이어서 `ChatOpenAI (llm_hi)`가 이 프롬프트를 받아 **문서를 검토하고 질문에 대한 중간 추론 결과**를 생성합니다. 그 결과, 예를 들어 *“ToT 적용을 위한 핵심 개념 정리, 여러 접근 방식 비교, 그리고 최종 선택 이유”* 등에 대한 서술이 만들어집니다.

2. **요약 단계** – 두 번째 프롬프트 `tmpl2`는 앞 단계의 \*\*중간 사고 결과(`{intermediate}`)\*\*를 받아들여, **해당 내용을 200자 이내로 요약**하도록 모델에게 지시합니다. 이 단계는 ToT의 **가지치기(branch pruning)** 개념과 유사하게, 앞서 생성된 장문의 사고 과정을 **핵심만 간추리는 역할**을 합니다. `llm_hi`는 앞 단계의 출력을 요약하여 **핵심 내용만 담은 요약문**을 만들어냅니다. (예: 앞서 여러 관점으로 나눠 서술된 내용을 한두 문장으로 압축)

3. **최종 답변 생성 단계** – 세 번째 프롬프트 `tmpl3`는 요약된 내용 `{summary}`를 기반으로 **최종 사용자 답변**을 작성하도록 요청합니다. 이제 LLM은 요약문을 참고하여 질문에 대한 **최종 답변을 생성**하며, 이 답변은 앞의 요약된 사고 과정을 바탕으로 한 **완결된 응답**이 됩니다. `llm_hi`가 생성한 최종 답변에는 질문에 대한 해결책이나 설명이 담기며, 체이닝을 거치지 않고 바로 답변했을 때보다 **더 체계적이고 균형 잡힌 내용**이 들어갈 가능성이 높습니다.

위 세 단계는 \*\*파이프 연산자 `|`\*\*를 이용해 하나의 체인 `chain_hi`으로 연결되었습니다. `chain_hi.invoke(...)`를 호출하면 각 단계가 순차적으로 실행되며, **1단계 출력 → 2단계 입력, 2단계 출력 → 3단계 입력**으로 자동 전달됩니다. 최종적으로 `res_hi`에는 LLM이 생성한 **최종 답변 텍스트**가 저장되고, `print(res_hi)`를 통해 그 내용을 출력합니다.

이 **심화 체인 기법**은 Tree of Thought의 아이디어를 반영한 것으로, 모델이 한 번에 모든 것을 답하기보다는 **생각을 나누어 단계적으로 진행**하도록 만든 점이 특징입니다. 이렇게 하면 복잡한 질문에 대해 **더 깊이 있고 신뢰성 높은 답변**을 얻을 수 있지만, 그만큼 여러 번의 LLM 호출이 필요하므로 **시간과 자원 소모가 증가**하는 트레이드오프도 있습니다. 실제로 이 코드 셀의 최종 출력 내용을 보면, **ToT 기법을 RAG에 적용했을 때의 장점**(여러 관점 탐색, 신뢰성 향상 등)과 함께 **단점**(시간 지연과 시스템 복잡도 증가)까지 언급되어 있습니다. 이처럼 체이닝을 활용하면 모델의 다양한 사고 과정을 거쳐 **균형 잡힌 답변**을 도출할 수 있음을 보여줍니다.

```python
# 1) 중간 사고 생성
tmpl1 = PromptTemplate(
    input_variables=['docs','query'],
    template="""
문서:
{docs}

질문: {query}

1) 중간 사고: ToT 적용을 위한 핵심 개념 요약
2) 중간 사고: 비교 분석
3) 선택 이유 설명
"""
)
# 2) 중간 사고 요약
tmpl2 = PromptTemplate(
    input_variables=['intermediate'],
    template="""
{intermediate}

위 중간 사고를 200자 이내로 요약하세요.
"""
)
# 3) 최종 답변 생성
tmpl3 = PromptTemplate(
    input_variables=['summary'],
    template="""
요약:
{summary}
최종 답변을 작성하세요.
"""
)
llm_hi = ChatOpenAI(model='gpt-4o-mini', temperature=0)
chain_hi = tmpl1 | llm_hi | tmpl2 | llm_hi | tmpl3 | llm_hi
res_hi = chain_hi.invoke({'docs': docs_str, 'query': 'ToT 분기 로직으로 RAG 성능 개선 방안은?'})
print(res_hi)
```
