# RAG 프롬프트 압축 실습


High 단계에서 JSON 응답을 엄격히 제어하고 후처리하는 코드 포함.

```python
# 세 개의 긴 문서 정의
doc1 = '''현대 교육 분야에서는 **인공지능(AI) 기반 플랫폼**이 빠르게 확산되고 있다. 초기에는 단순히 학습 진도 관리나 자동 채점에 머물렀으나, 최근에는 학습자 개인의 성향과 학습 패턴을 분석해 최적의 학습 경로를 제시하는 **맞춤형 학습 추천 시스템**으로 진화하고 있다. 예를 들어, 학습자의 풀이 시간을 실시간으로 모니터링해 난이도를 조절하고, 오답 유형을 분석해 유사 문제를 자동으로 출제해 주는 기능이 이미 상용화된 상태이다.

또한, **자연어 처리(NLP)** 기술의 발달로 강의 콘텐츠를 자동으로 요약·생성하거나, 학습자의 질문에 실시간으로 답변하는 챗봇 튜터가 등장했다. 이들 챗봇은 단순 키워드 검색을 넘어 의미 기반 검색과 대화형 학습을 지원하여, 교실 수업이나 온라인 강의의 한계를 극복하는 보조 교사 역할을 수행한다.

미래에는 강화 학습 기반 에이전트가 학습자와의 상호작용을 통해 학습 효과를 스스로 최적화하고, **메타러닝** 기법으로 새로운 과목·주제에도 빠르게 적응하는 플랫폼이 나올 것으로 기대된다. 특히, VR·AR 환경과 결합된 몰입형 학습(Immersive Learning)에서는 가상 교실에서 AI 튜터와 1:1 과외를 받는 듯한 경험이 가능해질 전망이다.

이처럼 AI 교육 플랫폼은 교육의 **접근성**, **효율성**, **개인화**를 획기적으로 개선하며**,** 전통적 교실 수업의 패러다임을 재정의하고 있다.'''

doc2 = '''**Retrieval-Augmented Generation(RAG)**은 대규모 언어 모델(LLM)의 생성 능력과 외부 지식 저장소(벡터 DB)를 결합해, 더 정확하고 최신의 정보를 활용한 답변을 생성하는 기술이다. 전통적인 LLM은 학습 시점 이후의 정보나 매우 구체적인 사안에 대해 오류를 범하기 쉽지만, RAG는 벡터 검색을 통해 연관 문서를 찾아와 이를 컨텍스트로 제공함으로써 이러한 한계를 보완한다.

RAG 파이프라인은 주로 세 단계로 이루어진다. 첫째, 사용자의 질의(Query)를 임베딩하여 벡터 DB에서 연관도가 높은 문서를 **Top-K** 방식으로 검색한다. 둘째, 검색된 문서들을 프롬프트에 포함시키고, 셋째, LLM이 이를 종합해 최종 답변을 생성한다. 이 과정에서 문서 정제(cleaning), 메타데이터 필터링, 요약(compression) 등의 전처리 기법이 함께 적용되면 효율성과 정확도가 더욱 향상된다.

실제 활용 사례로는 기업 내부 위키 및 문서 DB를 활용한 **사내 지식 검색 챗봇**, 법률·의학 분야에서 대규모 논문·판례·임상 데이터를 기반으로 한 **전문 상담 시스템**, 그리고 전자상거래 분야에서 고객 리뷰와 상품 설명을 결합해 **정교한 상품 추천**을 제공하는 시스템 등이 있다.

특히, RAG에 **프롬프트 압축(prompt compression)** 기법을 도입하면, LLM 입력 토큰 수를 줄여 응답 비용(cost)을 절감하면서도 핵심 정보를 유지해 답변 품질을 높일 수 있다.'''

doc3 = '''**프롬프트 압축(Prompt Compression)**은 LLM에 입력하는 컨텍스트(문서·대화 로그·검색 결과 등)를 최대한 효율적으로 압축해, 모델 토큰 한도 내에서 핵심 정보만 전달하는 기법이다. 일반적으로 LLM 비용은 입력·출력 토큰 수에 비례하므로, 불필요한 문장이나 중복 정보가 많을수록 비용이 증가하고 응답 속도가 느려진다.

압축 방법은 크게 두 가지로 나뉜다. 첫째, **정적 압축**으로, Python 슬라이싱으로 필요 문구를 잘라내는 방식이다. 둘째, **동적 압축**으로, 요약용 LLM을 별도로 두고 `docs → summarizer → compressed_docs` 순으로 체인을 구성해 의미 기반으로 압축하는 방법이다. 후자는 중요한 세부 사항을 놓치지 않으면서도 더 극적인 토큰 절감 효과를 기대할 수 있다.'''

retrieved_docs = [doc1, doc2, doc3]
```

이 코드 셀에서는 예제로 사용할 세 개의 긴 문서를 정의하고 있습니다:

- **`doc1`**: 인공지능 기반 교육 플랫폼의 발전과 개인 맞춤형 학습 시스템에 대한 내용. (현대 교육에서 AI 플랫폼이 어떻게 활용되고 발전하고 있는지 소개합니다.)
- **`doc2`**: RAG(Retrieval-Augmented Generation) 기술과 이를 활용한 정확하고 최신 정보 제공 방법에 대한 설명. (대규모 언어 모델과 외부 지식 베이스를 결합하여 답변의 정확성을 높이는 기법을 다룹니다.)
- **`doc3`**: 프롬프트 압축(Prompt Compression)의 개념과 정적/동적 압축 기법에 대한 소개. (LLM에 입력되는 문맥을 효율적으로 줄여 토큰 비용을 줄이는 방법을 설명합니다.)

각 문서는 여러 줄로 이루어진 긴 텍스트를 담고 있으며, Python의 삼중 따옴표(`'''`) 문법을 사용하여 이러한 여러 줄의 문자열을 그대로 변수에 저장합니다. 문자열 내부에 `**굵은 텍스트**`와 같은 마크다운 문법이 보이지만, 이는 문자열의 일부일 뿐 실제로 포맷이 적용되지는 않습니다. 마지막으로 `retrieved_docs` 리스트에 `doc1`, `doc2`, `doc3`를 순서대로 담아 이후 단계에서 세 문서를 한꺼번에 사용할 수 있도록 준비합니다.

## 난이도 하: 기본 요약 압축 (슬라이싱)

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

docs = '\n\n'.join(retrieved_docs)
trunc = ' '.join(docs.split()[:200])
template = PromptTemplate.from_template(
    '다음 문서를 읽고 요약하세요. 문서:\n{docs}\n질문: {question}'
)
runnable = template | ChatOpenAI(model='gpt-4o-mini', temperature=0)
print(runnable.invoke({'docs': trunc, 'question': '공통 주제는?'}))
```

이 코드 셀에서는 앞에서 정의한 문서들을 이용하여 간단한 요약 작업을 수행합니다. 주요 동작은 다음과 같습니다:

1. `PromptTemplate`와 `ChatOpenAI` 클래스를 임포트합니다. `PromptTemplate`은 프롬프트(질의)의 형식을 템플릿으로 정의할 때 사용하며, `ChatOpenAI`는 OpenAI의 챗 모델(여기서는 `gpt-4o-mini`)을 호출하는 객체입니다.
2. 앞서 준비한 `retrieved_docs`의 세 문서를 하나의 문자열로 합칩니다. `'\n\n'.join(retrieved_docs)`는 각 문서 사이에 빈 줄 두 개(`\n\n`)를 넣어 이어붙이는 것으로, 이렇게 하면 서로 다른 문서의 내용이 두 줄 공백으로 구분됩니다.
3. 합쳐진 문서 문자열을 단어 단위로 쪼갠 후 처음 200개 단어까지만 취하여(`docs.split()[:200]`), 다시 `' '.join(...)`으로 합쳐 `trunc` 변수에 저장합니다. 즉, 세 문서의 내용 중 앞부분 200단어만 발췌하여 요약 입력용 텍스트로 사용합니다. 이러한 고정 길이 자르기 방식이 바로 앞서 문서에서 언급된 **정적 압축**의 한 예입니다 (내용을 이해하기 쉽게 일부만 잘라내는 단순한 압축 기법).
4. 요약 작업에 사용할 프롬프트 템플릿을 생성합니다. `PromptTemplate.from_template('다음 문서를 읽고 요약하세요. 문서:\n{docs}\n질문: {question}')` 코드는 주어진 문자열에서 `{docs}`와 `{question}` 자리를 나중에 채워넣을 수 있는 템플릿 객체를 만들어냅니다. 템플릿 내용은 한국어로 *"다음 문서를 읽고 요약하세요. 문서:\n[문서내용]\n질문: [질문]"* 형식인데, 실제 실행 시 `[문서내용]` 부분에 `docs` 텍스트가, `[질문]` 부분에 사용자의 질문이 삽입됩니다.
5. `template | ChatOpenAI(...)` 구문은 앞의 템플릿 출력이 바로 뒤의 `ChatOpenAI` 모델의 입력으로 연결되는 **파이프라인**을 구성합니다. 즉, 템플릿에 문서와 질문을 채워 완성된 프롬프트를 생성하면, 그 프롬프트를 `gpt-4o-mini` 모델에 전달하여 답을 얻도록 한 줄로 연결한 것입니다. 여기서 모델 생성자의 `temperature=0`은 답변 생성의 무작위성을 0으로 줄여 항상 같은 입력에 대해 일관된 출력(결정론적 출력)을 얻기 위함입니다.
6. `runnable.invoke({...})`를 호출하여 체인을 실행합니다. 인자로 `{'docs': trunc, 'question': '공통 주제는?'}` 딕셔너리를 넣는데, 이는 템플릿의 `{docs}` 자리에 `trunc` 문자열을, `{question}` 자리에 `"공통 주제는?"`이라는 질문을 채워 넣으라는 의미입니다. 최종적으로 `print(...)`를 통해 모델의 응답을 출력합니다. 이 응답에는 세 문서를 통해 얻은 **공통 주제**가 요약되어 포함되어 있을 것입니다. (예를 들어, 출력 스트림에는 모델이 작성한 요약문이 표시되며, 내부적으로 콘텐츠와 메타데이터를 담은 객체로 나타날 수 있지만, 여기서는 요약된 핵심 내용 자체가 중요합니다.)

## 난이도 중: 요약 기반 동적 압축

```python
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

docs = '\n\n'.join(retrieved_docs)
trunc = ' '.join(docs.split()[:300])
sum_t = PromptTemplate.from_template('100토큰 이내 요약: {docs}')
summ = sum_t | ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
msg = summ.invoke({'docs': trunc}); summary = msg.content
qa_t = PromptTemplate.from_template('요약:{summary}\n질문:{question}')
qa = qa_t | ChatOpenAI(model='gpt-4o-mini', temperature=0)
print(qa.invoke({'summary': summary, 'question': '공통 주제는?'}))
```

이 코드 셀은 '난이도 중' 단계로, 요약을 활용한 **동적 압축** 방법을 보여줍니다. 앞선 정적 슬라이싱과 달리 여기서는 LLM을 사용하여 먼저 문서를 요약하고 그 요약본으로 질문에 답하는 두 단계 방식을 사용합니다. 동작 과정은 다음과 같습니다:

1. 필요한 라이브러리를 다시 임포트합니다 (`PromptTemplate`, `ChatOpenAI`). 이전 셀에서 이미 임포트했더라도, 이 셀을 단독으로 실행할 때를 대비하여 다시 불러오고 있습니다.
2. 세 문서를 합쳐 `docs` 문자열로 만들고, 이번에는 300개의 단어까지만 잘라 `trunc`에 저장합니다. 이는 앞 단계보다 더 많은 내용(300단어)을 남겨두어 약간 덜 압축된 버전의 문서를 준비한 것입니다.
3. 요약 작업을 위한 프롬프트 템플릿 `sum_t`를 정의합니다. 템플릿 문자열 `'100토큰 이내 요약: {docs}'`는 주어진 문서(`{docs}` 자리)에 대해 **100토큰 이내**로 요약하라는 명령을 담고 있습니다. 여기서 "토큰"은 모델이 처리하는 텍스트의 단위이며, 100토큰은 대략 짧은 단락 정도의 분량을 의미합니다. 이처럼 요약 길이에 제한을 두어 핵심만 간추리도록 지시합니다.
4. `sum_t | ChatOpenAI(model='gpt-4o-mini', temperature=0.3)`로 요약용 체인 `summ`을 생성합니다. 앞의 템플릿을 사용해 요약 프롬프트를 만들고, 이를 `ChatOpenAI` 모델에 전달해 요약 결과를 얻는 파이프라인입니다. 여기서 `temperature=0.3`으로 약간의 무작위성을 부여했는데, 이는 요약 결과가 항상 동일하지 않고 다양하게 표현될 수 있도록 한 것입니다 (값이 0보다 크면 출력에 창의성이 가미됩니다).
5. `msg = summ.invoke({'docs': trunc})`를 호출하여 준비된 문서(`trunc`)에 대한 요약을 생성합니다. `msg`는 모델의 응답 객체이고, `msg.content`를 통해 요약된 텍스트만 추출하여 `summary` 변수에 저장합니다. 이제 `summary`에는 합쳐진 문서들의 핵심 내용을 담은 요약문이 들어 있습니다.
6. 질문을 위해 새로운 프롬프트 템플릿 `qa_t`를 정의합니다. 템플릿 `'요약:{summary}\n질문:{question}'`은 방금 얻은 요약문과 질문을 하나로 합쳐 줍니다. 즉, 프롬프트의 형태가 "요약: [요약문]\n질문: [질문]"이 되며, 모델은 요약문을 참고하여 질문에 답하게 됩니다.
7. `qa_t | ChatOpenAI(model='gpt-4o-mini', temperature=0)`로 최종 질의응답 체인 `qa`를 생성합니다. 앞서 만든 템플릿에 질문과 요약을 채우면, 그 프롬프트를 `ChatOpenAI` 모델에 전달해 답변을 얻는 흐름입니다. 이번에는 `temperature=0`으로 설정하여 모델이 가능한 한 결정론적으로, 즉 요약문에 기반한 가장 신뢰도 높은 답변을 내놓도록 합니다.
8. `print(qa.invoke({'summary': summary, 'question': '공통 주제는?'}))`를 호출하여 요약된 정보로부터 **공통 주제**에 대한 답변을 얻습니다. 최종 답변이 출력되며, 이 접근법에서는 모델이 전체 문서 대신 **요약된 핵심 정보**를 보고 답을 생성하기 때문에 토큰 사용량을 줄이면서도 중요한 내용에 집중할 수 있다는 장점이 있습니다.

## 난이도 상: 조건부 압축 및 엄격 JSON 응답

```python
import re, json
from typing import Dict
from langchain_core.runnables.base import Runnable
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

class ConditionalCompressor(Runnable):
    def __init__(self, threshold=300):
        self.threshold = threshold
        self.summarizer = (
            PromptTemplate.from_template('80토큰 요약: {docs}')
            | ChatOpenAI(model='gpt-4o-mini', temperature=0.3)
        )
    def invoke(self, inputs: Dict[str, str]) -> str:
        docs = inputs['docs']; words = docs.split()
        if len(words) > self.threshold:
            sliced = ' '.join(words[:self.threshold])
            return self.summarizer.invoke({'docs': sliced}).content
        return docs

# 1) 압축 수행
docs = '\n\n'.join(retrieved_docs)
comp_text = ConditionalCompressor().invoke({'docs': docs})

# 2) QA with strict JSON only
qa_template = PromptTemplate.from_template(
    '문서:\n{docs}\n질문: {question}\n'
    '응답은 오직 JSON 객체 하나({{"answer": "..."}})만 출력하세요.'
)
qa = qa_template | ChatOpenAI(model='gpt-4o-mini', temperature=0)
qa_msg = qa.invoke({'docs': comp_text, 'question': '공통 주제는 무엇인가요?'})

# 3) 후처리 및 파싱
m = re.search(r"\{.*\}", qa_msg.content, re.S)
if not m:
    raise ValueError('JSON 출력 실패: ' + qa_msg.content)
parsed = json.loads(m.group())
print(parsed)
```

이 코드 셀은 '난이도 상' 단계로, **조건부 압축**과 **엄격한 JSON 형식 응답**을 구현합니다. 앞서 동적 요약을 활용했지만, 여기서는 텍스트 길이에 따라 요약을 수행할지 말지 결정하고, 모델에게 답변을 JSON 형태로 출력하도록 요구하는 고급 기법을 보여줍니다. 단계별 내용은 다음과 같습니다:

1. Python의 기본 모듈 `re`(정규표현식)와 `json`을 임포트하고, 타입 힌트를 위해 `typing`의 `Dict`를 불러옵니다. 또한 `langchain_core.runnables.base`에서 `Runnable` 베이스 클래스를, 그리고 이전에 사용했던 `PromptTemplate`과 `ChatOpenAI`를 다시 임포트합니다. `Runnable`은 Langchain에서 사용자 정의 연산을 체인에 통합하기 위한 기본 클래스입니다.
2. `ConditionalCompressor` 라는 커스텀 클래스를 정의합니다. 이 클래스는 `Runnable`을 상속하여 체인에 넣을 수 있는 **조건부 압축기** 역할을 합니다:
    - **생성자** (`__init__`): `threshold=300`이라는 단어 수 임계치를 설정하고, `self.summarizer`라는 요약 체인을 준비합니다. `self.summarizer`는 `'80토큰 요약: {docs}'`라는 요약용 프롬프트 템플릿과 `ChatOpenAI(model='gpt-4o-mini', temperature=0.3)` 모델을 파이프로 연결한 것으로, 길이가 긴 문서를 최대 80토큰으로 요약하는 역할을 합니다.
    - **`invoke` 메서드**: 입력으로 받은 `docs` 문자열을 단어별로 나눈 리스트(`words`)를 만들어 그 길이를 확인합니다. 만약 단어 수가 설정된 임계치(300 단어)보다 많으면, `words[:300]`으로 앞부분 300단어만 잘라낸 후 이를 `self.summarizer.invoke({'docs': sliced})`에 전달하여 요약 결과를 얻습니다. 이렇게 요약된 텍스트(`.content`로 추출)를 반환하고, 반대로 300 단어 이하면 요약을 하지 않고 원문 그대로를 반환합니다.
    - 이 클래스는 문서가 아주 길 때만 요약을 수행하고 짧으면 그대로 두는 **조건부 압축** 로직을 캡슐화하고 있습니다. 이를 통해 불필요한 정보 손실을 줄이면서도 입력 길이를 관리할 수 있습니다.
3. 정의한 `ConditionalCompressor`를 즉시 활용하여 문서 압축을 수행합니다. 먼저 `docs = '\n\n'.join(retrieved_docs)`로 세 문서를 다시 하나의 문자열로 결합합니다. 그리고 `comp_text = ConditionalCompressor().invoke({'docs': docs})`를 호출하여 결합된 문서를 조건부 압축기에 넣습니다. 결과로 얻어진 `comp_text`에는 문서 길이에 따라 **압축된 텍스트**가 들어갑니다. (현재 예시의 세 문서는 비교적 길기 때문에, 아마도 앞부분 300단어를 요약한 80토큰 분량의 텍스트로 압축되었을 것입니다.)
4. 압축된 텍스트를 사용하여 질문에 답하는 프롬프트를 구성하고, 모델에게 **엄격한 JSON 형식**으로 답변을 요구합니다. `qa_template` 변수를 통해 프롬프트 템플릿을 정의하는데, 여기에는 문서 내용(`{docs}`)과 질문(`{question}`) 뿐만 아니라 *"응답은 오직 JSON 객체 하나({`"answer": "..."`})만 출력하세요."* 라는 문장이 포함되어 있습니다. 이 지시는 모델에게 답변을 오로지 `{"answer": "..."}` 형태의 JSON 한 객체로만 출력하라는 요구사항입니다. 그런 다음 `qa_template | ChatOpenAI(model='gpt-4o-mini', temperature=0)`로 템플릿과 모델을 파이프로 연결하여 QA 체인을 만들고, `qa_msg = qa.invoke({'docs': comp_text, 'question': '공통 주제는 무엇인가요?'})`로 실제 질문을 실행합니다. 여기서 질문을 "공통 주제는 무엇인가요?"라고 표현하여 모델이 보다 자연스러운 한국어로 된 질문에 답하도록 했습니다.
5. 모델의 응답을 받아 **후처리 및 파싱**을 수행합니다. `re.search(r"\{.*\}", qa_msg.content, re.S)`를 이용해 모델 출력 문자열에서 `{`와 `}`로 둘러싸인 JSON 객체 부분만 찾아냅니다 (`re.S` 플래그는 줄바꿈이 있어도 전체 문자열에서 패턴을 찾을 수 있도록 합니다). 만약 이러한 JSON 형태를 찾지 못했다면 모델이 지시에 따르지 않고 엉뚱한 형식으로 답변한 것이므로 `ValueError`를 발생시켜 오류를 표시합니다. JSON 문자열을 올바르게 찾았다면 `json.loads(m.group())`로 그 문자열을 실제 파이썬 사전(dict) 객체로 변환합니다. 마지막으로 `print(parsed)`를 통해 파싱된 결과를 출력합니다. 이렇게 하면 모델의 답변이 `{'answer': 'AI 기반 교육 기술의 발전과 맞춤형 학습 시스템'}`처럼 Python 딕셔너리 형태로 나타나며, `'answer'` 키 아래에 세 문서의 공통 주제가 텍스트로 담겨 나오게 됩니다.