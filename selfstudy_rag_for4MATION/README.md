# RAG Fundamentals — KDIC 프로젝트 기반 개념 정리

> LikeLion AI/NLP 5기 · 4MATION 예금보험공사 RAG 챗봇 PoC에서 실제로 굴린 개념들을 역순으로 되짚는 학습 트랙.
> 각 토픽은 독립 `.ipynb` 한 개로 대응하며, 이 문서는 그 인덱스이자 메타데이터 원장이다.

---

## 학습 원칙

실무에서 이미 손으로 만진 것들이라 "몰라서 배우는" 게 아니라 **"왜 그게 맞았는지 언어화하는"** 작업이다. 따라서 모든 노트북은 아래 4단 구조를 지킨다.

| 섹션 | 내용 |
|---|---|
| **1. 개념** | 표준 정의 · 수식 · 다른 개념과의 경계 (예: RRF ≠ score fusion) |
| **2. 최소 구현** | 라이브러리 없이 numpy/순수 파이썬으로 20줄 이내 재현 |
| **3. 프로젝트 앵커** | 4MATION 레포의 어느 파일/어느 결정에 이 개념이 박혀 있는지 |
| **4. 숫자 검증** | 실제 코퍼스·평가셋 수치로 확인. 숫자 없으면 그 노트북은 미완성 |

> 격언 이식: **"숫자 없는 머지는 없다" → "숫자 없는 TIL은 없다"**

---

## 디렉토리 구조

```text
rag-fundamentals/
├── README.md                      # 이 문서 (인덱스 · 진도 · 레퍼런스)
├── assets/                        # 공용 샘플 데이터 (미니 코퍼스·미니 평가셋)
│   ├── mini_corpus.jsonl          #   프로젝트 코퍼스에서 20청크 샘플링
│   └── mini_testset.jsonl         #   평가셋에서 30문항 샘플링
├── A_retrieval/
│   ├── A1_embedding-biencoder.ipynb
│   ├── A2_bm25-sparse.ipynb
│   ├── A3_korean-tokenization.ipynb
│   ├── A4_ann-index-faiss.ipynb
│   ├── A5_rank-fusion-rrf.ipynb
│   ├── A6_metadata-boosting.ipynb
│   └── A7_reranking.ipynb
├── B_pipeline/
│   ├── B1_chunking-strategy.ipynb
│   ├── B2_hierarchical-metadata.ipynb
│   ├── B3_crawling-ethics-robots.ipynb
│   └── B4_source-integrity-stale-data.ipynb
├── C_evaluation/
│   ├── C1_ir-metrics.ipynb
│   ├── C2_micro-vs-macro.ipynb
│   ├── C3_sample-size-confidence.ipynb
│   ├── C4_data-leakage.ipynb
│   ├── C5_labeling-methodology.ipynb
│   └── C6_rag-evaluation-ragas.ipynb
├── D_generation/
│   ├── D1_rag-architecture.ipynb
│   ├── D2_query-routing.ipynb
│   ├── D3_grounding-citation.ipynb
│   ├── D4_decoding-params.ipynb
│   └── D5_guardrail-rejection.ipynb
└── E_engineering/
    ├── E1_interface-contract.ipynb
    ├── E2_reproducibility.ipynb
    └── E3_cost-latency-budget.ipynb
```

**네이밍 규칙**: `{트랙}{번호}_{kebab-case-영문}.ipynb` — 정렬 순서 = 학습 순서.

---

## 진도표

### 트랙 A — 검색 (Retrieval)

- [ ] **A1** 임베딩과 bi-encoder
- [ ] **A2** Sparse retrieval — TF-IDF에서 BM25까지
- [ ] **A3** 한국어 토크나이징
- [ ] **A4** ANN 인덱스와 FAISS
- [ ] **A5** Rank fusion과 RRF
- [ ] **A6** 메타데이터 부스팅·하드필터·per-doc cap
- [ ] **A7** Reranking

### 트랙 B — 데이터 파이프라인

- [ ] **B1** 청킹 전략
- [ ] **B2** 계층 메타데이터 설계
- [ ] **B3** 크롤링 윤리와 robots.txt
- [ ] **B4** 소스 무결성과 stale data 처리

### 트랙 C — 평가

- [ ] **C1** IR 메트릭
- [ ] **C2** micro vs macro 평균
- [ ] **C3** 표본 크기와 신뢰구간
- [ ] **C4** 데이터 누수 (self-referential leakage)
- [ ] **C5** 평가셋 라벨링 방법론
- [ ] **C6** RAG 전용 평가

### 트랙 D — 생성 / LLM

- [ ] **D1** RAG 아키텍처 전반
- [ ] **D2** 쿼리 라우팅
- [ ] **D3** 그라운딩과 출처 표시
- [ ] **D4** 디코딩 파라미터
- [ ] **D5** 가드레일과 거절

### 트랙 E — 엔지니어링

- [ ] **E1** 인터페이스 계약 동결
- [ ] **E2** 재현성
- [ ] **E3** 비용·레이턴시 예산

---

## 토픽 상세

### 트랙 A — 검색 (Retrieval)

#### A1. 임베딩과 bi-encoder

| 항목 | 내용 |
|---|---|
| **학습 목표** | 문장이 왜 벡터가 되는지, bi-encoder 구조, 코사인 유사도와 내적의 관계, 정규화의 역할 |
| **프로젝트 앵커** | `jhgan/ko-sroberta-multitask` (768d) · UI "관련 문서 유사도" = dense cosine top-1 |
| **핵심 질문** | · 왜 `IndexFlatIP`(내적)인데 코사인이라 부를 수 있나?<br>· 문장 길이가 짧으면 왜 임베딩 품질이 떨어지나? (명사형 질의 이슈)<br>· mean pooling vs CLS 토큰 |
| **검증 과제** | 미니 코퍼스에 대해 같은 질의를 명사형/자연어형으로 넣고 top-1 코사인 값 비교 |
| **레퍼런스** | [Sentence-Transformers 문서](https://sbert.net/) · [Pinecone Learn — Embeddings](https://www.pinecone.io/learn/) · [HF NLP Course Ch.1](https://huggingface.co/learn/nlp-course) |

#### A2. Sparse retrieval — TF-IDF에서 BM25까지

| 항목 | 내용 |
|---|---|
| **학습 목표** | 역색인 구조, TF-IDF의 한계, BM25의 `k1`(포화)·`b`(길이 정규화)가 실제로 하는 일 |
| **프로젝트 앵커** | `rank_bm25` + kiwipiepy 토크나이저 조합 · 자연어화 시 BM25 단독 0.90→0.73 붕괴 |
| **핵심 질문** | · `k1`을 키우면/줄이면 랭킹이 어떻게 변하나?<br>· 청크 길이가 들쭉날쭉하면 `b`가 왜 중요해지나?<br>· BM25F(필드 가중)는 언제 필요한가? (보호한도 id 16 케이스) |
| **검증 과제** | `k1`·`b`를 격자로 바꿔가며 미니 평가셋 hit@3 표 만들기 |
| **레퍼런스** | Elastic 블로그 "Practical BM25" 3부작 · [Stanford IR Book Ch.11](https://nlp.stanford.edu/IR-book/) · [rank_bm25](https://github.com/dorianbrown/rank_bm25) |

#### A3. 한국어 토크나이징

| 항목 | 내용 |
|---|---|
| **학습 목표** | 형태소 분석 vs subword(BPE/WordPiece), 조사·어미 처리가 sparse 검색에 미치는 영향 |
| **프로젝트 앵커** | kiwipiepy 형태소 토큰으로 BM25 인덱싱 · "조사 치환 수준이 아닌 실질적 패러프레이즈" 원칙 |
| **핵심 질문** | · "착오송금을"과 "착오송금은"이 BM25에서 다른 토큰이면 무슨 일이 벌어지나?<br>· dense 쪽은 왜 이 문제에 상대적으로 강한가?<br>· 명사 추출만 vs 전체 형태소 — 뭐가 나았나? |
| **검증 과제** | 동일 질의를 공백분리 / kiwi 형태소 / subword 3가지로 토크나이즈해 BM25 랭킹 비교 |
| **레퍼런스** | [kiwipiepy 문서](https://bab2min.github.io/kiwipiepy/) · [위키독스 — 딥러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155) · [HF Tokenizers 챕터](https://huggingface.co/learn/nlp-course/chapter6/1) |

#### A4. ANN 인덱스와 FAISS

| 항목 | 내용 |
|---|---|
| **학습 목표** | 완전탐색 vs 근사탐색, IVF·HNSW·PQ의 원리, recall–latency–메모리 트레이드오프 |
| **프로젝트 앵커** | `IndexFlatIP` (215청크 규모에서 완전탐색이 정답인 이유) |
| **핵심 질문** | · 몇 만 벡터부터 ANN이 의미 있어지나?<br>· HNSW의 `ef_search`·`M`이 recall에 미치는 영향<br>· 종합 단계에서 코퍼스가 10배 되면 뭘 바꿔야 하나? |
| **검증 과제** | 랜덤 벡터 10만 개로 Flat vs IVF vs HNSW의 recall@10과 검색 시간 측정 |
| **레퍼런스** | [FAISS Wiki — Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki) · [Pinecone Learn — Vector Indexes](https://www.pinecone.io/learn/) |

#### A5. Rank fusion과 RRF

| 항목 | 내용 |
|---|---|
| **학습 목표** | score fusion과 rank fusion의 근본적 차이, RRF 공식 `Σ 1/(k + rank)`, 상수 `k`의 의미 |
| **프로젝트 앵커** | `mode="hybrid"` RRF(k=60) · **"RRF `_score`는 dense cosine과 스케일이 달라 임계값 비교 금지"** |
| **핵심 질문** | · 왜 점수를 min-max 정규화해서 더하지 않고 순위를 쓰나?<br>· `k=60`을 60에서 10으로 낮추면 상위권 가중이 어떻게 변하나?<br>· RRF 출력값을 confidence로 못 쓰는 이유를 수식으로 설명하면? |
| **검증 과제** | 동일 질의에 대해 score fusion(정규화 합) vs RRF 결과를 나란히 놓고 순위 뒤집힘 관찰 |
| **레퍼런스** | Cormack et al. 2009, *Reciprocal Rank Fusion* (2p) · [Weaviate 블로그 — Hybrid Search](https://weaviate.io/blog) · [Elastic RRF 문서](https://www.elastic.co/docs) |

#### A6. 메타데이터 부스팅·하드필터·per-doc cap

| 항목 | 내용 |
|---|---|
| **학습 목표** | 후처리 재랭킹으로서의 부스팅, soft boost vs hard filter, 특정 문서 독점 방지 |
| **프로젝트 앵커** | `hybrid_bf` 모드 · `selectFaqNramtAply` 블랙홀(미적중 43문항 top-3에 61회 침투) 제거 |
| **핵심 질문** | · 하드필터의 위험(정답 배제)과 라우팅 오분류의 관계<br>· per-document chunk cap이 recall에 주는 부작용<br>· 부스팅 계수를 어떻게 정해야 과적합이 아닌가? |
| **검증 과제** | cap 없음 / cap=2 / cap=1 세 조건에서 업무별 hit@3 변화표 |
| **레퍼런스** | [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) · [LangChain — Retrieval 개념](https://python.langchain.com/docs/concepts/retrieval/) |

#### A7. Reranking

| 항목 | 내용 |
|---|---|
| **학습 목표** | cross-encoder 구조, bi-encoder와의 비용/성능 차이, ColBERT 같은 중간 지점 |
| **프로젝트 앵커** | 미도입 상태 · **리트리버 hit@k가 리랭킹의 이론적 상한**이라는 제약 · 페어 트리아지 "좋은 실패 16건"이 정확한 타깃 |
| **핵심 질문** | · 왜 cross-encoder를 전체 코퍼스에 못 돌리나?<br>· top-k를 몇으로 잡아야 리랭커가 의미 있나? (현재 hit@10 확인 필요)<br>· 한국어 cross-encoder 후보는? |
| **검증 과제** | 좋은 실패 16건에 대해 top-10 후보를 뽑고 cross-encoder로 재정렬 → hit@3 개선폭 측정 |
| **레퍼런스** | [Sentence-Transformers — Cross-Encoder](https://sbert.net/) · [Pinecone Learn — Rerankers](https://www.pinecone.io/learn/) · [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) |

---

### 트랙 B — 데이터 파이프라인

#### B1. 청킹 전략

| 항목 | 내용 |
|---|---|
| **학습 목표** | fixed / recursive / semantic / document-aware 청킹, overlap의 비용과 효과 |
| **프로젝트 앵커** | 38매니페스트 → 215청크 · 종합 단계 관리자 UI의 chunk size 튜닝 파라미터 |
| **핵심 질문** | · 청크가 작을수록 정밀도는 올라가는데 왜 답변 품질은 떨어질 수 있나?<br>· overlap이 평가 지표를 인위적으로 부풀리는 경로<br>· 표·목록이 섞인 안내 페이지는 어떻게 쪼개야 하나? |
| **검증 과제** | 동일 문서를 크기 3종으로 청킹해 hit@3와 평균 컨텍스트 길이 동시 비교 |
| **레퍼런스** | [Pinecone Learn — Chunking Strategies](https://www.pinecone.io/learn/) · [LangChain Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/) |

#### B2. 계층 메타데이터 설계

| 항목 | 내용 |
|---|---|
| **학습 목표** | 청크–문서–업무 계층 모델링, 메타데이터가 검색·평가·디버깅에 동시에 쓰이는 구조 |
| **프로젝트 앵커** | `parent_doc_id` · `business_function` · 오류분석의 ✅정답/⚠️타업무혼입/❌동일업무오답 태깅 |
| **핵심 질문** | · gt를 청크가 아니라 문서 단위로 잡은 판단은 어떤 대가를 치렀나?<br>· 메타데이터 없이 오류분석이 가능했을까?<br>· www/fins 중복 주제를 묶지 않고 URL로 구분한 이유 |
| **검증 과제** | 메타데이터 필드를 하나씩 제거하며 오류분석 리포트가 얼마나 무력해지는지 서술 |
| **레퍼런스** | [LlamaIndex — Metadata Extraction](https://docs.llamaindex.ai/) · [Pinecone Learn — Metadata Filtering](https://www.pinecone.io/learn/) |

#### B3. 크롤링 윤리와 robots.txt

| 항목 | 내용 |
|---|---|
| **학습 목표** | robots.txt 문법과 법적 지위, rate limiting·지터·백오프, fail-closed 설계 |
| **프로젝트 앵커** | H5(fins `/cm/bbs/` 차단) → 목적 명시 후 수집 허가 획득 · `POLICY_DISALLOW` 오버레이 이중 안전장치 |
| **핵심 질문** | · fail-open과 fail-closed 중 왜 후자인가?<br>· 허가를 받아도 별도 오버레이를 유지하는 이유<br>· 간격 1.5초 + 지터의 근거 |
| **검증 과제** | `robotparser`로 두 도메인 정책 파싱 → 스코프 URL 전체 허용/차단 판정표 |
| **레퍼런스** | [RFC 9309 — Robots Exclusion Protocol](https://www.rfc-editor.org/rfc/rfc9309.html) · [Python `urllib.robotparser`](https://docs.python.org/3/library/urllib.robotparser.html) |

#### B4. 소스 무결성과 stale data 처리

| 항목 | 내용 |
|---|---|
| **학습 목표** | 원문 불변 원칙, 답변 레이어 보정, 범위형 데이터의 별도 참조 테이블 관리 |
| **프로젝트 앵커** | 보호한도 5천만→1억(2025-09-01 시행) 보정 · `ProtSystProtLmts#01` stale 토큰 · 코퍼스 드리프트와 `baseline_*.lock` |
| **핵심 질문** | · 코퍼스를 직접 고치면 뭐가 망가지나? (재현성·출처 신뢰성)<br>· "한 곳뿐이면 수기, 반복 등장하면 별도 테이블" 기준의 근거<br>· SHA256 락 파일이 격리하는 변수는 정확히 무엇인가? |
| **검증 과제** | 보정 레이어 on/off로 동일 질의 답변 diff + 지표 무영향 확인 |
| **레퍼런스** | 프로젝트 `DECISIONS.md` (H6b·R2 계열) · [Great Expectations — 데이터 검증 개념](https://docs.greatexpectations.io/) |

---

### 트랙 C — 평가

#### C1. IR 메트릭

| 항목 | 내용 |
|---|---|
| **학습 목표** | hit@k / recall@k / precision@k / MRR / MAP / nDCG의 정의와 서로 다른 질문 |
| **프로젝트 앵커** | `eval.py` 지표군 · hybrid hit@3 0.945(400문항) · hit@1 0.613 |
| **핵심 질문** | · 복수 정답(허용집합)에서 hit@k와 recall@k는 왜 달라지나?<br>· MRR이 hit@3보다 민감한 상황<br>· 우리가 nDCG를 안 쓴 이유는? (등급 라벨 부재) |
| **검증 과제** | 동일 랭킹 결과에 5개 지표를 전부 계산해 서로 다른 결론이 나오는 케이스 찾기 |
| **레퍼런스** | [Stanford IR Book Ch.8](https://nlp.stanford.edu/IR-book/) · [Evidently AI 블로그 — Ranking Metrics](https://www.evidentlyai.com/blog) |

#### C2. micro vs macro 평균

| 항목 | 내용 |
|---|---|
| **학습 목표** | 문항 쏠림이 종합 지표를 왜곡하는 메커니즘, gap의 해석법 |
| **프로젝트 앵커** | 재라벨 전 gap +5.7%p → 후 −2.5%p 수렴 · 착오송금 0.205→0.909 |
| **핵심 질문** | · macro > micro면 무슨 뜻이고 macro < micro면 무슨 뜻인가?<br>· 정답 문서 1개가 전체의 25%를 차지하면 유효 표본은 실제로 얼마인가?<br>· 발표에 어떤 값을 인용해야 정직한가? |
| **검증 과제** | 미니 평가셋의 특정 업무 문항 수를 인위적으로 늘려 micro만 움직이는 걸 재현 |
| **레퍼런스** | [scikit-learn — averaging 옵션](https://scikit-learn.org/stable/modules/model_evaluation.html) · 프로젝트 `changelog_testset_eval.md` |

#### C3. 표본 크기와 신뢰구간

| 항목 | 내용 |
|---|---|
| **학습 목표** | 이항 비율의 Wilson 신뢰구간, 유의성 검정(부호검정), 필요 표본 수 계산 |
| **프로젝트 앵커** | n=150 → 95% CI 폭 ±8%p · 하락 20 / 상승 5 부호검정 p≈0.002 · 300문항 확장 |
| **핵심 질문** | · +2%p 개선을 "개선"이라 부르려면 표본이 몇 개 필요한가?<br>· 왜 Wald가 아니라 Wilson인가?<br>· 페어드 설계(원본↔자연어)가 검정력에 주는 이득 |
| **검증 과제** | n=52·150·300·400 각각의 CI 폭 계산표 → "몇 %p 차이부터 말할 수 있나" 결론 |
| **레퍼런스** | [statsmodels `proportion_confint`](https://www.statsmodels.org/stable/index.html) · [SciPy `binomtest`](https://docs.scipy.org/doc/scipy/) |

#### C4. 데이터 누수 (self-referential leakage)

| 항목 | 내용 |
|---|---|
| **학습 목표** | 평가셋 질문이 인덱싱된 문서에 그대로 들어있을 때 지표가 어떻게 상한에 붙는가 |
| **프로젝트 앵커** | 52문항 중 46건이 자기참조(FAQ 질문 문자열·헤딩 텍스트) · 원본 0.853 vs 자연어 0.753 · LCS<12자 게이트 |
| **핵심 질문** | · −10%p 갭이 왜 "실패"가 아니라 "검증 엄밀성의 증거"인가?<br>· BM25 단독은 0.90→0.73으로 붕괴했는데 hybrid는 −10%p에 그친 이유<br>· LCS 임계값을 12자로 잡은 근거는 무엇이어야 하나? |
| **검증 과제** | 질문↔gt청크 LCS 길이 분포 히스토그램 → LCS 구간별 hit@3 계층화 |
| **레퍼런스** | Kapoor & Narayanan, *Leakage and the Reproducibility Crisis in ML-based Science* (arXiv 2207.07048) · 프로젝트 `verify_natural.py` |

#### C5. 평가셋 라벨링 방법론

| 항목 | 내용 |
|---|---|
| **학습 목표** | ground truth 정의 원칙, 허용 정답 집합, 라벨 감사 절차, 어노테이션 일관성 |
| **프로젝트 앵커** | 68건 재라벨(0.580→0.873) · 라벨 감사 v2 23건 교정 · **"gt는 본문에 실제로 답하는 문서 전부, 리트리버 top-3 여부가 아니다"** |
| **핵심 질문** | · 근접중복 문서를 허용집합에 넣는 기준<br>· 라벨을 고쳐 점수가 오른 걸 "성능 개선"이라 부를 수 있나?<br>· 오답 83건 중 라벨 탓 7건 / 진짜 검색 문제 76건 — 이 분해를 어떻게 정당화하나? |
| **검증 과제** | 트리아지 3분류(좋은 실패 / gt 협소 / 저작 수정) 기준을 명문화하고 샘플 10건 직접 분류 |
| **레퍼런스** | [BEIR 벤치마크](https://github.com/beir-cellar/beir) · TREC 어노테이션 가이드라인 · 프로젝트 `label_audit.md` |

#### C6. RAG 전용 평가

| 항목 | 내용 |
|---|---|
| **학습 목표** | 검색 지표만으로 부족한 이유, faithfulness / answer relevancy / context precision·recall |
| **프로젝트 앵커** | 미도입 — 종합 단계 후보 · 현재는 retrieval hit + 사람 검수 병행 |
| **핵심 질문** | · 검색이 맞았는데 답이 틀리는 경우를 어떤 지표가 잡나?<br>· LLM-as-judge의 편향과 비용<br>· 우리 300문항 평가셋을 RAG 평가셋으로 승격하려면 뭐가 더 필요한가? (정답 answer 필드) |
| **검증 과제** | 데모 결과 12행(`demo_results.jsonl`)에 faithfulness를 수동 채점해 자동 지표와 비교 |
| **레퍼런스** | [Ragas 문서](https://docs.ragas.io/) · [LlamaIndex — Evaluation](https://docs.llamaindex.ai/) |

---

### 트랙 D — 생성 / LLM

#### D1. RAG 아키텍처 전반

| 항목 | 내용 |
|---|---|
| **학습 목표** | naive → advanced → modular RAG 계보, 각 단계에서 추가되는 컴포넌트 |
| **프로젝트 앵커** | 실전2 전체 · `rag_hcx.py`의 RAG vs 맨몸 대조 실험 |
| **핵심 질문** | · 파인튜닝 대신 RAG를 택하는 판단 기준<br>· 우리 구조는 계보상 어디에 있나?<br>· "세 층이 전부 규칙대로 작동해도 재료(검색)가 틀리면 답은 틀린다"를 아키텍처 언어로 옮기면? |
| **검증 과제** | 대표 6문항 RAG vs bare 답변 diff 표 → RAG가 실제로 무엇을 고쳤는지 유형화 |
| **레퍼런스** | *Retrieval-Augmented Generation for LLMs: A Survey* (arXiv 2312.10997) · Lewis et al. 2020 원논문 (arXiv 2005.11401) · [LlamaIndex 문서](https://docs.llamaindex.ai/) |

#### D2. 쿼리 라우팅

| 항목 | 내용 |
|---|---|
| **학습 목표** | 의도 분류, 라우팅 실패 비용의 비대칭성, 규칙 기반 vs LLM 기반 라우터 |
| **프로젝트 앵커** | `rag` / `link` / `reject` 3분기 · 오분류 사례 id 19(퇴직연금) · 계약 동결 후 라우팅만 교체한 사례 |
| **핵심 질문** | · false reject와 false accept 중 어느 쪽이 더 비싼가? (공공기관 맥락)<br>· 라우터가 틀리면 하드필터가 정답을 배제하는 연쇄<br>· 라우팅 정확도를 별도 지표로 재야 하는 이유 |
| **검증 과제** | 300문항 + OOD 질문 20건으로 라우팅 confusion matrix 작성 |
| **레퍼런스** | [LangChain — Routing](https://python.langchain.com/docs/concepts/) · [LlamaIndex Router Query Engine](https://docs.llamaindex.ai/) |

#### D3. 그라운딩과 출처 표시

| 항목 | 내용 |
|---|---|
| **학습 목표** | 환각의 유형, 컨텍스트 구속 프롬프트 설계, 인용 강제 기법, 답변–출처 정합 검증 |
| **프로젝트 앵커** | `SYS_RAG`("참고문서에만 근거해 답하라 / 없으면 모른다") · `answer()`의 `sources` 필드 · id 16 stale 데이터 동의 오류 |
| **핵심 질문** | · "모른다고 답하라"는 지시가 실제로 얼마나 지켜지나?<br>· 모델이 컨텍스트와 내부 지식이 충돌할 때 뭘 택하나?<br>· 인용 번호가 실제 출처와 어긋나는 걸 자동으로 잡으려면? |
| **검증 과제** | 컨텍스트에 없는 정보를 묻는 질문 10건 → 모른다 응답률 측정 |
| **레퍼런스** | [Anthropic 프롬프트 엔지니어링 가이드](https://docs.claude.com/en/docs/build-with-claude/prompt-engineering/overview) · [OpenAI Prompt Engineering](https://platform.openai.com/docs/guides/prompt-engineering) |

#### D4. 디코딩 파라미터

| 항목 | 내용 |
|---|---|
| **학습 목표** | greedy / temperature / top-k / top-p / beam의 차이, 재현성과 다양성의 트레이드오프 |
| **프로젝트 앵커** | `GEN_PARAMS = {temperature: 0.2, topP: 0.8, maxTokens: 512}` 고정 — 재현성 우선 |
| **핵심 질문** | · temperature 0과 0.2의 실질적 차이<br>· 안내 챗봇에서 다양성이 왜 리스크인가?<br>· maxTokens 512가 잘린 답변을 만든 적은 없나? |
| **검증 과제** | 동일 질문 × temperature 0/0.2/0.7 각 3회 → 답변 편차 정성 비교 |
| **레퍼런스** | [HF — Text Generation Strategies](https://huggingface.co/docs/transformers/generation_strategies) · [CLOVA Studio 문서](https://guide.ncloud-docs.com/) |

#### D5. 가드레일과 거절

| 항목 | 내용 |
|---|---|
| **학습 목표** | out-of-scope 탐지, 임계값 기반 거절, 거절 문구 설계, 안전한 폴백(타 기관 안내) |
| **프로젝트 앵커** | `reject` 라우트 · `link` 라우트(외부 기관 이관) · 데모 영상 reject 케이스(FDIC 해외기관 질문) |
| **핵심 질문** | · 거절 임계값을 dense cosine으로 잡을 때의 스케일 문제 (RRF `_score` 사용 불가)<br>· 거절이 너무 잦으면 어떤 사용자 경험 비용이 발생하나?<br>· "모른다" vs "우리 소관 아님" vs "여기로 가세요"의 구분 기준 |
| **검증 과제** | 스코프 밖 질문 20건 수집 → 거절률·오거절률(정상 질문 거절) 동시 측정 |
| **레퍼런스** | [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) · [Anthropic — Reducing Hallucinations](https://docs.claude.com/en/docs/) |

---

### 트랙 E — 엔지니어링

#### E1. 인터페이스 계약 동결

| 항목 | 내용 |
|---|---|
| **학습 목표** | 계약(contract) 개념, 안정 인터페이스 뒤에서 구현을 교체하는 설계, 회귀 테스트의 역할 |
| **프로젝트 앵커** | `answer() → {text, sources, confidence, route} + checks/_meta` 동결 후 라우팅 통째 교체 · `eval.py`는 불변 회귀 인프라 |
| **핵심 질문** | · "겉을 얼렸기에 속을 갈아끼울 수 있었다"를 일반 원칙으로 진술하면?<br>· 어떤 필드를 계약에 넣고 어떤 걸 `_meta`로 빼야 하나?<br>· `eval.py`를 아무도 안 고치기로 한 규칙의 값어치 |
| **검증 과제** | 계약 위반을 잡는 스모크 테스트 작성 (필드 존재·타입·범위) |
| **레퍼런스** | [Semantic Versioning](https://semver.org/lang/ko/) · Martin Fowler, *Contract Test* |

#### E2. 재현성

| 항목 | 내용 |
|---|---|
| **학습 목표** | 시드 고정, 아티팩트 해싱, 환경 고정, 오프라인 모드, 락 파일 설계 |
| **프로젝트 앵커** | `baseline_*.lock` (chunks.jsonl SHA256 + testset 버전) · `HF_HUB_OFFLINE=1` · **"발표 수치는 단일 로컬로 고정 인용"** · 크로스머신 재현 확인(H25) |
| **핵심 질문** | · 로컬마다 `--rebuild`하면 왜 ±2%p가 흔들렸나?<br>· 락 파일이 격리하는 변수와 못 잡는 변수<br>· 재현 실패를 조기에 감지하는 최소 장치는? |
| **검증 과제** | 코퍼스 SHA256 + 모델 리비전 + 라이브러리 버전을 찍는 `env_fingerprint()` 구현 |
| **레퍼런스** | [Papers with Code — ML Reproducibility Checklist](https://paperswithcode.com/) · [DVC 문서](https://dvc.org/doc) |

#### E3. 비용·레이턴시 예산

| 항목 | 내용 |
|---|---|
| **학습 목표** | 토큰 비용 모델, 호출 예산 가드, 백오프 전략, p50/p95 레이턴시, 캐싱 |
| **프로젝트 앵커** | `CALL_BUDGET = 20` · 429 시 5초 / 그 외 2초 백오프 · `latency_ms` 기록 |
| **핵심 질문** | · top-k를 3에서 5로 올리면 비용이 얼마나 늘고 품질은 얼마나 오르나?<br>· 임베딩 모델 로드를 싱글턴으로 만든 이유<br>· 관리자 콘솔에서 파라미터를 열어줄 때 비용 폭주를 어떻게 막나? |
| **검증 과제** | top-k별 컨텍스트 토큰 수 × 단가 → 월 예상 비용 시뮬레이션 표 |
| **레퍼런스** | [CLOVA Studio 요금·쿼터 문서](https://guide.ncloud-docs.com/) · [Google SRE Book — Handling Overload](https://sre.google/books/) |

---

## 공통 레퍼런스

책상 위에 상시 열어두는 것들.

| 사이트 | 커버 범위 | 왜 |
|---|---|---|
| [Pinecone Learn](https://www.pinecone.io/learn/) | A1·A4·A5·A7·B1·B2 | 벡터검색 전 영역을 그림으로 설명. 입문–중급 구간 최고 밀도 |
| [Stanford IR Book (무료 전문)](https://nlp.stanford.edu/IR-book/) | A2·C1 | BM25·역색인·평가지표의 원전. 애매할 때 여기로 돌아온다 |
| [Sentence-Transformers](https://sbert.net/) | A1·A7 | bi-encoder vs cross-encoder를 코드로 확인 |
| [Hugging Face NLP Course](https://huggingface.co/learn/nlp-course) | A1·A3·D4 | 토크나이저·트랜스포머 기초 |
| [위키독스 — 딥러닝을 이용한 자연어 처리 입문](https://wikidocs.net/book/2155) | A2·A3 | 한국어 설명이 필요할 때 |
| [LangChain 개념 문서](https://python.langchain.com/docs/concepts/) | B1·D1·D2 | 용어 표준화 참조 (구현은 직접) |
| [LlamaIndex 문서](https://docs.llamaindex.ai/) | B2·C6·D1·D2 | 검색·평가 개념 정리가 특히 좋다 |
| [Ragas](https://docs.ragas.io/) | C6 | RAG 지표 정의 |
| [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) | A1·A7 | 한국어 임베딩 모델 위치 확인 |
| [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) | A6·B1 | 우리 구조(청킹+BM25+하이브리드+리랭커)와 거의 동일한 실증 |
| [Evidently AI 블로그](https://www.evidentlyai.com/blog) | C1·C2 | 랭킹 지표 시각 설명 |
| [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki) | A4 | 인덱스 선택 가이드 |

---

## 진행 순서 (4주 기준)

| 주차 | 토픽 | 목표 |
|---|---|---|
| **1주** | A1 → A2 → A3 → A4 → A5 | 검색 엔진의 뼈대. 여기까지 하면 `rag.py`를 백지에서 다시 쓸 수 있어야 한다 |
| **2주** | B1 → B2 → C1 → C2 → C4 | 파이프라인 + 평가 신뢰도. C4는 포트폴리오 1순위 글감 |
| **3주** | C3 → C5 → A6 → A7 | 통계적 엄밀성 + 검색 고도화. A7은 다음 실험 설계로 직결 |
| **4주** | D1 → D2 → D3 → D5 → C6 | 생성·라우팅·RAG 평가 |
| **수시** | B3 · B4 · D4 · E1 · E2 · E3 | 짧은 회고형. 각 30분 이내 |

**포트폴리오 우선순위**: C4(누수) > C2(micro/macro) > A5(RRF 오해) > C5(라벨 감사) > B4(원문 무수정)
— 전부 "남들이 안 하는 걸 했다"가 숫자로 증명되는 항목이다.

---

## 노트북 템플릿

새 노트북 시작할 때 첫 마크다운 셀에 붙일 것.

```markdown
# {ID}. {토픽명}

| | |
|---|---|
| **트랙** | {A 검색 / B 파이프라인 / C 평가 / D 생성 / E 엔지니어링} |
| **선수 토픽** | {ID 목록} |
| **프로젝트 앵커** | {파일·결정·수치} |
| **작성일** | YYYY-MM-DD |
| **상태** | 초안 / 검증완료 |

## 한 줄 요약
> {이 노트북을 다 읽고 나면 무엇을 설명할 수 있는가}

## 1. 개념
## 2. 최소 구현
## 3. 프로젝트 앵커
## 4. 숫자 검증
## 5. 남은 질문
```
