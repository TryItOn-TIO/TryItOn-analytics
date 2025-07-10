[Try It On] 추천시스템 레포지토리입니다.

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
python main.py
```
