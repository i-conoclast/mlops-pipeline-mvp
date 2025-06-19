# Contributing to MLOps Pipeline MVP

## 브랜치 전략

### 브랜치 구조
- `master`: 프로덕션 준비된 안정적인 코드
- `dev`: 개발 중인 기능들의 통합 브랜치
- `feature/*`: 새로운 기능 개발 브랜치

### 워크플로우

1. **새 기능 개발**
   ```bash
   git checkout dev
   git pull origin dev
   git checkout -b feature/your-feature-name
   ```

2. **개발 완료 후 dev에 머지**
   ```bash
   git checkout dev
   git merge feature/your-feature-name
   git push origin dev
   ```

3. **dev에서 master로 승격**
   ```bash
   git checkout master
   git merge dev
   git push origin master
   ```

## 커밋 규칙

### 커밋 메시지 형식
```
<type>: <description>

[optional body]
```

### 타입 종류
- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `docs`: 문서 수정
- `test`: 테스트 코드 추가/수정
- `refactor`: 코드 리팩토링
- `ci`: CI/CD 설정 변경
- `chore`: 기타 작업

### 예시
```bash
git commit -m "feat: add model validation pipeline"
git commit -m "fix: resolve MLflow tracking URI configuration"
git commit -m "docs: update README with deployment instructions"
```

## Pull Request 가이드라인

### PR 생성 전 체크리스트
- [ ] 모든 테스트 통과
- [ ] 코드 린팅 통과 (black, flake8, isort)
- [ ] 새로운 기능에 대한 테스트 작성
- [ ] 문서 업데이트 (필요한 경우)

### PR 제목 형식
```
[타입] 간단한 설명
```

예시:
- `[FEAT] Add automated model deployment pipeline`
- `[FIX] Resolve Airflow DAG import error`
- `[DOCS] Update logging system documentation`

### PR 설명 템플릿
```markdown
## 변경 내용
- 주요 변경사항 요약

## 테스트
- [ ] 유닛 테스트 통과
- [ ] 통합 테스트 통과
- [ ] 수동 테스트 완료

## 체크리스트
- [ ] 코드 리뷰 완료
- [ ] 문서 업데이트
- [ ] 로그 기록 업데이트
```

## 개발 환경 설정

### 로컬 개발
```bash
# 프로젝트 클론
git clone git@github.com:i-conoclast/mlops-pipeline-mvp.git
cd mlops-pipeline-mvp

# Poetry 환경 설정
poetry install
poetry shell

# 테스트 실행
poetry run pytest tests/ -v

# 코드 품질 검사
poetry run black src/ tests/
poetry run isort src/ tests/
poetry run flake8 src/ tests/
```

### Docker 환경
```bash
# 전체 서비스 실행
docker-compose up -d

# 서비스 확인
# MLflow UI: http://localhost:5000
# Airflow UI: http://localhost:8080 (admin/admin)
```

## 코드 품질 기준

### 테스트 커버리지
- 최소 80% 커버리지 유지
- 새로운 기능은 반드시 테스트 코드 포함

### 코드 스타일
- Black을 사용한 자동 포매팅
- isort를 사용한 import 정리
- flake8을 사용한 린팅 규칙 준수

### 문서화
- 모든 함수와 클래스에 docstring 작성
- README 및 관련 문서 최신 상태 유지
- Claude Code 세션 로그 작성

## 보안 가이드라인

- 코드에 하드코딩된 비밀번호, API 키 등 포함 금지
- 환경 변수 또는 별도 설정 파일 사용
- 민감한 정보는 .gitignore에 추가

## 이슈 리포팅

### 버그 리포트
1. GitHub Issues에 버그 리포트 작성
2. 재현 단계, 예상 결과, 실제 결과 명시
3. 관련 로그 및 스크린샷 첨부

### 기능 제안
1. GitHub Issues에 기능 제안 작성
2. 필요성과 예상 효과 설명
3. 구현 방안 제시 (선택사항)