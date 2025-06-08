# LangChain RAG Few-Shot 프롬프트 엔지니어링 실습


LangChain 0.3 및 LCEL 스타일 체이닝 예제입니다.
각 섹션마다 예시 Q-A 페어를 포함하여 진정한 Few-Shot 프롬프트를 구현했습니다.

**첫 번째 코드 셀 설명:** 이 코드 셀에서는 RAG(Retrieval-Augmented Generation)를 위한 준비 작업을 수행합니다.

* 먼저 `json` 모듈과 LangChain 라이브러리의 여러 구성 요소들을 임포트합니다. 여기에는 프롬프트 템플릿을 구성하는 `PromptTemplate`, OpenAI 기반 모델을 활용하는 `ChatOpenAI`, 문서 데이터를 담는 `Document`, 그리고 출력에서 원하는 정보를 추출하는 `RegexParser`가 포함됩니다.
* 다음으로, 이미 \*\*리트리버(Retriever)\*\*를 통해 확보한 것으로 가정한 3개의 문서를 `Document` 객체 형태로 리스트 `docs`에 준비합니다. 각 문서는 파리, 런던, 교토에 대한 간략한 정보(수도 및 연간 관광객 수)를 담고 있습니다. 이 과정은 **데이터 준비 단계**로, 일반적인 RAG 파이프라인에서 검색된 문서를 활용하는 부분입니다 (여기서는 편의를 위해 하드코딩되어 있습니다).
* 마지막으로, 오픈AI의 언어 모델을 사용하기 위해 `ChatOpenAI` 객체를 생성하여 `llm` 변수에 할당합니다. `model="gpt-4o-mini"`는 사용할 모델의 이름을 나타내며, `temperature=0` 설정을 통해 응답 생성 시 **무작위성을 배제**하여 항상 일관되고 deterministically 동일한 답변을 얻도록 합니다 (즉, **재현 가능하고 안정적인 출력**을 위해 temperature를 0으로 설정).

```python
import json
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document
from langchain.output_parsers import RegexParser

# 가정: 이미 Retriever가 Top-3 문서를 가져옴
docs = [
    Document(page_content="파리는 프랑스의 수도로, 연간 관광객 수 약 3000만 명입니다."),
    Document(page_content="런던은 영국의 수도로, 연간 관광객 수 약 2000만 명입니다."),
    Document(page_content="교토는 일본의 옛 수도로, 연간 관광객 수 약 1500만 명입니다.")
]

# 모델 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
```

## 1. 난이도 하 (Easy) – Low Complexity

### Few-Shot 예시 포함

**코드 셀 1 설명 (쉬운 난이도):** 이 코드 셀에서는 **Few-Shot 학습** 기법을 사용하여 간단한 질의 응답을 수행합니다. 두 개의 예시 Q\&A를 프롬프트에 포함시켜 모델이 답변 형식을 학습하도록 합니다.

* 먼저 `PromptTemplate.from_template` 메소드를 이용해 프롬프트 템플릿을 생성합니다. 이 템플릿에는 문서 목록 `{docs}`를 표시하는 섹션과 **예시 Q\&A 페어**가 포함됩니다. 예시는 "Q: 파리의 국가는? A: 프랑스"와 "Q: 런던의 국가는? A: 영국"처럼, 실제 질의와 그에 대한 정답 형식으로 두 가지를 제공합니다. 이러한 **Few-Shot 예시**는 모델에게 질문과 답변의 형식을 미리 보여주어, 이후에 제시할 실제 질문에 대해 올바른 형식과 내용을 가진 답변을 생성하도록 도와줍니다. 템플릿의 마지막 부분에는 실제 질문 자리표시자 `{query}`가 있으며, 모델은 이 부분을 채워 답변하게 됩니다.
* `RegexParser`를 설정하여 모델의 출력에서 원하는 부분만 추출합니다. 여기서는 정규표현식 `(?s)(.*)`를 사용하여 출력 전체를 하나의 그룹으로 캡처하며, 이를 `'answer'`라는 키로 저장합니다. 이 파서는 \*\*모델의 응답에서 우리가 관심 있는 텍스트(전체 답변)\*\*만 가져오는 역할을 합니다. (이번 예시에서는 응답 전체가 곧 정답이므로 그대로 캡처합니다.)
* 그 다음, `template | llm | parser` 구문을 통해 프롬프트 템플릿 → LLM(언어 모델) → 파서로 이어지는 \*\*체인(chain)\*\*을 구성합니다. 이 체인은 LangChain Expression Language(LCEL)의 문법을 활용한 것으로, `|` 연산자를 통해 각 단계를 순차적으로 연결합니다. 이렇게 하면 입력이 템플릿에 채워지고, 그 결과가 LLM으로 전달되어 응답이 생성된 뒤, 마지막으로 파서를 통해 필요한 부분이 추출되는 **일련의 처리 흐름**이 만들어집니다.
* `chain.invoke({...})`를 호출하여 체인을 실행합니다. 이때 `{docs}` 자리에는 앞서 준비한 `docs` 리스트의 내용이 들어가고, `{query}` 자리에는 실제 질문인 `'교토의 국가는?'`이 전달됩니다. 코드에서는 `'\n\n'.join(d.page_content for d in docs)`를 통해 세 문서의 내용을 두 줄씩 띄워 하나의 문자열로 합쳐 `docs` 자리에 채웁니다. 모델은 주어진 문서 내용과 예시를 참고하여 질문에 대한 답변을 생성하게 됩니다.
* 마지막으로 `print(res['answer'])`를 통해 모델의 답변을 출력합니다. 이 예시에서는 교토에 대한 국가를 묻고 있으므로, 모델은 제공된 문서의 정보에 따라 \*\*"일본"\*\*이라고 답합니다. (교토는 일본에 있기 때문에, 예시 Q\&A와 문서 내용으로부터 정답을 유추할 수 있습니다.)

```python
# 예시 두 개를 포함하는 Few-Shot
template = PromptTemplate.from_template(
    '''문서:
{docs}

예시:
Q: 파리의 국가는?
A: 프랑스

Q: 런던의 국가는?
A: 영국

Q: {query}
A:'''  
)
parser = RegexParser(regex=r'(?s)(.*)', output_keys=['answer'])
chain = template | llm | parser
res = chain.invoke({
    'docs': '\n\n'.join(d.page_content for d in docs),
    'query': '교토의 국가는?'
})
print(res['answer'])
```

(실행 결과)

```
일본
```

## 2. 난이도 중 (Medium) – Low JSON Complexity

### JSON Few-Shot 예시

**코드 셀 2 설명 (중간 난이도):** 이 코드 셀에서는 **JSON 형식의 답변**을 생성하는 예시를 보여줍니다. 모델에게 출력 형식을 JSON으로 제한하기 위해, 프롬프트에 JSON 형태의 Few-Shot 예시와 출력 요건을 명시합니다.

* `PromptTemplate.from_template`으로 새로운 프롬프트 템플릿을 정의합니다. 이번 템플릿에도 문서 섹션과 예시 Q\&A가 포함되는데, 답변 형식은 **JSON 배열**로 구성됩니다. 예를 들어 첫 번째 예시 질문은 "Q: 파리의 관광객 수는?"이고, 이에 대한 답을 `A: [{"city":"파리","visitors":"3000만"}]`처럼 **리스트 내부에 하나의 JSON 객체**로 제공합니다. 두 번째 예시도 런던에 대한 유사한 형태로 제공합니다. 이러한 예시를 통해 모델은 질문에 대한 **응답을 JSON으로 표현하는 방식**을 학습합니다. 템플릿 끝부분에 실제 질문 `{query}`가 들어가며, 특별히 `"응답은 JSON 배열만 출력하세요."`라는 지시문을 추가하여 모델이 **여분의 설명 없이 오직 JSON 결과만 생성**하도록 유도합니다.
* `RegexParser`를 사용하여 모델 출력에서 JSON 배열 부분만을 추출하도록 설정합니다. 정규표현식 `(\[.*\])`은 대괄호 `[...]`로 둘러싸인 전체 내용을 매치하며, 이것을 `'json_str'` 키로 저장합니다. 이처럼 파서를 사용하면 모델의 출력에 불필요한 텍스트가 섞여 있더라도 **우리가 원하는 JSON 부분만 깔끔하게 추출**할 수 있습니다.
* 준비된 프롬프트 템플릿, LLM, 파서를 이전과 마찬가지로 체인으로 연결합니다 (`chain = template | llm | parser`). 이 체인을 실행하면, 프롬프트가 완성되고 → 모델이 JSON 형식의 답변을 생성하며 → 파서가 그 JSON 텍스트만 결과로 추출하는 단계가 차례로 이루어집니다.
* `chain.invoke({...})`로 체인을 실행할 때, `{docs}`에는 세 문서의 내용이 합쳐져 전달되고 `{query}`에는 실제 질문인 `'교토의 관광객 수는?'`이 입력됩니다. 모델은 제공된 문서에서 교토에 관한 정보를 찾고, Few-Shot 예시를 참고하여 답을 JSON 형식으로 만들어 냅니다.
* `res['json_str']`로 추출된 JSON 문자열을 `json.loads(...)` 함수를 통해 \*\*Python 객체(리스트 형태)\*\*로 변환합니다. 이 과정은 문자열로 된 JSON을 실제 데이터 구조로 파싱하여 이후 프로그램에서 활용할 수 있게 해줍니다.
* 마지막으로 `print(data)`를 통해 파싱된 데이터를 출력합니다. 교토에 대한 관광객 수 정보가 문서에 있으므로, 모델은 이를 JSON 형식으로 반환하며, 파싱 결과는 파이썬의 리스트 안에 딕셔너리 객체 `{'city': '교토', 'visitors': '1500만'}`가 들어있는 형태가 됩니다. 실제 출력 결과를 보면 모델이 교토의 관광객 수를 1500만으로 표시하여 JSON 응답을 성공적으로 생성했음을 알 수 있습니다.

```python
# 두 개의 JSON Q-A 예시 포함
template = PromptTemplate.from_template(
    '''문서:
{docs}

예시:
Q: 파리의 관광객 수는?
A: [{{"city":"파리","visitors":"3000만"}}]

Q: 런던의 관광객 수는?
A: [{{"city":"런던","visitors":"2000만"}}]

Q: {query}
응답은 JSON 배열만 출력하세요.
A:'''  
)
parser = RegexParser(regex=r'(\[.*\])', output_keys=['json_str'])
chain = template | llm | parser
res = chain.invoke({
    'docs': '\n\n'.join(d.page_content for d in docs),
    'query': '교토의 관광객 수는?'
})
data = json.loads(res['json_str'])
print(data)
```

(실행 결과)

```
[{'city': '교토', 'visitors': '1500만'}]
```

## 3. 난이도 상 (Hard) – Medium JSON Complexity

### 조건부 JSON Few-Shot 예시

**코드 셀 3 설명 (높은 난이도):** 이 코드 셀에서는 **조건부 논리가 포함된 JSON 형식 답변**을 생성하는 예시를 다룹니다. 특정 질문에 대한 데이터가 없는 경우, 모델이 빈 배열 `[]`을 반환하도록 프롬프트에 지시하고 있습니다.

* 새로운 프롬프트 템플릿을 `PromptTemplate.from_template`으로 정의합니다. 이번 예시 질문들은 도시의 "연간 성장률"에 관한 것입니다. 문서에는 이러한 정보가 없지만, 우리는 Few-Shot 예시로 두 도시(파리와 런던)의 성장률 데이터를 제공하여 모델이 답변 형식을 학습하도록 합니다. 예를 들어 "Q: 파리 연간 성장률은?"에 대해 `A: [{"year":"2023","growth":"2.5%"}]`와 같은 JSON 답을 예시로 넣습니다. 런던에 대해서도 비슷한 형식의 예시를 제공합니다. 이때 템플릿의 마지막 부분에 실제 질문 `{query}`와 함께 `"응답은 JSON 배열만 출력하세요. 데이터 없으면 빈 배열."`이라는 문구를 포함시킵니다. 이를 통해 **모델에게 해당 정보가 문서에 없을 경우 빈 JSON 배열을 답변으로 내놓으라**고 명시적으로 안내합니다.
* `RegexParser`는 이전과 마찬가지로 정규표현식 `(\[.*\])`를 사용하여 출력 중 **대괄호로 감싸진 JSON 배열 부분만 추출**하도록 설정합니다. 이렇게 하면 모델이 답변과 함께 다른 설명을 덧붙이더라도, 우리가 원하는 JSON만 깨끗이 얻을 수 있습니다.
* 그런 다음 템플릿, LLM, 파서를 체인으로 연결합니다 (`chain = template | llm | parser`). 이제 체인을 실행하면, 문서와 예시로 채워진 프롬프트가 생성되고 → 모델이 조건에 맞는 답변(JSON 또는 빈 배열)을 만들어내며 → 파서가 그 JSON 문자열 부분만 반환하게 됩니다.
* `chain.invoke({...})`를 호출하여 실제 질의를 처리합니다. `{docs}`에는 변함없이 세 문서의 내용이 들어가고, `{query}`에는 `'교토 연간 성장률은?'`이라는 질문이 주어집니다. 여기서 교토에 대한 연간 성장률 정보는 문서에도 예시에도 없으므로, 모델은 앞서 프롬프트에 명시된 지시에 따라 해당 데이터를 찾지 못했을 경우의 대응을 하게 됩니다.
* 모델은 교토의 연간 성장률 데이터가 없음을 판단하면, 프롬프트 조건대로 `"[]"` (빈 JSON 배열) 형태의 답변을 생성합니다. `res['json_str']`를 통해 이 결과 문자열을 얻은 뒤, `json.loads`를 사용해 **Python의 리스트 객체**로 변환합니다.
* 마지막으로 `print(data)`를 실행하여 결과를 출력합니다. 출력된 결과가 빈 리스트 `[]`인 것을 확인할 수 있는데, 이는 **모델이 요구한 조건에 맞게 데이터가 없을 때 빈 JSON 배열을 반환**했음을 보여줍니다. 이러한 방식으로 프롬프트를 구성하면, 모델의 응답 형식을 세밀하게 제어하거나 특정 조건을 달성할 수 있습니다.

```python
# 세 개의 JSON 예시 포함 (조건부)
template = PromptTemplate.from_template(
    '''문서:
{docs}

예시:
Q: 파리 연간 성장률은?
A: [{{"year":"2023","growth":"2.5%"}}]

Q: 런던 연간 성장률은?
A: [{{"year":"2023","growth":"1.8%"}}]

Q: {query}
응답은 JSON 배열만 출력하세요. 데이터 없으면 빈 배열.
A:'''  
)
parser = RegexParser(regex=r'(\[.*\])', output_keys=['json_str'])
chain = template | llm | parser
res = chain.invoke({
    'docs': '\n\n'.join(d.page_content for d in docs),
    'query': '교토 연간 성장률은?'
})
data = json.loads(res['json_str'])
print(data)
```

(실행 결과)

```
[]
```
