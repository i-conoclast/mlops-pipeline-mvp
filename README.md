# MLOps Pipeline MVP

대규모 실험·모니터링 및 MLOps 파이프라인 경험 축적을 위한 4주 완성 MVP 프로젝트

## 프로젝트 개요

**목표**: MLOps 워크플로 구축 (4주 완성 MVP 버전)  
**기간**: 2025년 6월 16일 ~ 2025년 7월 13일

## 주차별 목표

### 1주차: 핵심 기반 구축 (6/16 ~ 6/22)
- [x] 프로젝트 설정 및 CI 구축
- [ ] 실험 관리 PoC (MLflow)
- [ ] Orchestration PoC (Airflow)

### 2주차: 파이프라인 통합 (6/23 ~ 6/29)
- [ ] Airflow DAG 고도화
- [ ] GitHub Actions와 Airflow 연동

### 3주차: E2E 자동화 및 배포 (6/30 ~ 7/6)
- [ ] 배포(CD) 파이프라인 구축
- [ ] 모니터링 및 알림 추가

### 4주차: 최종 테스트 및 회고 (7/7 ~ 7/13)
- [ ] 최종 파이프라인 운영 및 기록
- [ ] 회고 및 블로그 작성

## 기술 스택

- **CI/CD**: GitHub Actions
- **실험 관리**: MLflow
- **워크플로우 오케스트레이션**: Apache Airflow
- **컨테이너화**: Docker
- **모니터링**: 기본 알림 (Slack/Email)

## 프로젝트 구조

```
mlops-pipeline-mvp/
├── src/                    # 소스 코드
│   ├── models/            # 모델 관련 코드
│   ├── data/              # 데이터 처리
│   └── utils/             # 유틸리티
├── airflow/               # Airflow DAGs
├── .github/workflows/     # GitHub Actions
├── docker/                # Docker 설정
├── requirements.txt       # Python 의존성
└── docker-compose.yml     # 로컬 개발 환경
```