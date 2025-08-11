[Try It On] 추천시스템 레포지토리입니다.

## 추천 시스템 개요

이 프로젝트는 유저의 선호도를 반영하여 의류 상품을 추천하는 등의 다양한 추천 알고리즘을 제공하는 추천 시스템입니다.

### 📊 추천 알고리즘 종류

1. **콘텐츠 기반 추천** (`/recommend/for-you`)

   - 사용자의 선호 태그를 기반으로 유사한 상품 추천
   - 개인화된 추천 제공

2. **인기 상품 추천** (`/recommend/trending`)

   - 전체 사용자의 행동 데이터를 기반으로 인기 상품 추천
   - 위시리스트, 주문, 아바타 착용 등 다양한 행동 고려

3. **연령대별 추천** (`/recommend/age-group`)

   - 특정 연령대와 성별의 사용자들이 선호하는 상품 추천
   - 연령대별 맞춤 추천

4. **벡터 기반 하이브리드 추천** (`/recommend/hybrid`)

   - 벡터 임베딩을 활용한 상품 유사도 기반 추천
   - 카테고리 다양성을 고려한 추천

5. **협업필터링 추천**
   - **기본 협업필터링** (`/recommend/collaborative`): 사용자 간 유사성을 기반으로 추천
   - **다양성 기반 협업필터링** (`/recommend/collaborative-diverse`): 사용자가 접해보지 않은 카테고리 우선 추천
   - **하이브리드 협업필터링** (`/recommend/hybrid-collaborative`): 협업필터링과 콘텐츠 기반 추천 결합


---

## 🎯 Commit Convention

| 태그       | 설명                                           |
| ---------- | ---------------------------------------------- |
| `feat`     | 새로운 기능 추가                               |
| `fix`      | 버그 수정                                      |
| `docs`     | 문서 수정 (README 등)                          |
| `style`    | 코드 포맷팅, 세미콜론 누락 등 (기능 변경 없음) |
| `refactor` | 코드 리팩토링 (기능 변경 없음)                 |
| `test`     | 테스트 코드 추가 및 리팩토링                   |
| `chore`    | 빌드 설정, 패키지 매니저 등 기타 변경          |

---

## 🚀 프로젝트 실행 방법

### 1. 레포지토리 클론

```bash
git clone https://github.com/TryItOn-TIO/TryItOn-analytics.git
cd TryItOn-analytics
```

### 2. 가상 환경 설정

- mac os

```bash
python -m venv .venv
source .venv/bin/activate
```

- windows

```bash
python -m venv .venv
source .venv\Scripts\activate
```

### 3. 의존성 설치

```bash
pip install -r requirements.txt
```

### 4. 서버 실행

```bash
uvicorn main:app --reload
```

### 5. 브라우저에서 확인

1. http://127.0.0.1:8000 <br />
   접속하여 서버 실행 확인

2. http://127.0.0.1:8000/docs <br />
   대화형 API 문서로 확인 (Swagger UI로 API 호출과 테스트 가능합니다.)

3. http://127.0.0.1:8000/redoc <br />
   문서 중심으로 확인

### 6. API 엔드포인트

#### 협업필터링 API

- `GET /recommend/collaborative` - 기본 협업필터링 추천
- `GET /recommend/collaborative-diverse` - 다양성 기반 협업필터링 추천
- `GET /recommend/hybrid-collaborative` - 하이브리드 협업필터링 추천
- `POST /recommend/collaborative/clear-cache` - 협업필터링 캐시 초기화

#### 태그 기반 API

- `GET /recommend/for-you` - 개인화 추천 (태그 기반)

#### 쿼리 기반 API

- `GET /recommend/trending` - 인기 상품 추천
- `GET /recommend/age-group` - 연령대별 추천
- `GET /recommend/similar-to/{product_id}` - 유사 상품 추천
- `GET /recommend/tryon-based` - try-on 기반 추천
- `GET /recommend/hybrid` - 벡터 기반 하이브리드 추천
