# Reflection — Lab 18: Production RAG Pipeline

**Họ tên:** Phùng Đình Đạt
**MSSV:** 2A202601540
**Lớp:** AICB-K34 · Ngày 18
**Môi trường:** Python 3.11.7 (venv), Qdrant 1.19 (Docker), GPU RTX 3050 Laptop 3.68 GiB, LLM Gemini (`gemini-3.5-flash-lite`) qua endpoint OpenAI-compat

---

## Phần 1: Mapping bài giảng → code

| Lecture Concept | Module | Hàm cụ thể | Observation từ lần chạy thật |
|-----------------|--------|------------|------------------------------|
| Semantic chunking | M1 | `chunk_semantic()` (`src/m1_chunking.py:84`) | Threshold 0.85 + encoder `all-MiniLM-L6-v2` tạo **208 chunks** (avg 99 ký tự) so với **51 chunks** của `chunk_basic()` (avg 410). Ngưỡng 0.85 quá chặt với văn bản tiếng Việt: MiniLM là model đa ngữ yếu ở tiếng Việt nên similarity giữa 2 câu liền kề thường < 0.85, dẫn tới cắt gần như từng câu (min_len = 6 ký tự). Bài học: threshold phải tune theo encoder + ngôn ngữ, không có con số "chuẩn". |
| Hierarchical / parent-child | M1 | `chunk_hierarchical()` (`src/m1_chunking.py:143`) | Parent 2048 / child 256 cho **11 parents → 109 children** (avg 191 ký tự). Đây là strategy được chọn cho pipeline production: child đủ nhỏ để embedding không bị loãng, `parent_id` cho phép trả về nguyên section khi cần context rộng. |
| Structure-aware chunking | M1 | `chunk_structure_aware()` (`src/m1_chunking.py:181`) | Parse header `#{1,3}` → **106 chunks**, giữ nguyên tiêu đề section trong `metadata["section"]`. max_len 789 vì có bảng markdown dài không bị cắt giữa — đúng mục tiêu "không cắt giữa table". |
| Vietnamese tokenization | M2 | `segment_vietnamese()` (`src/m2_search.py:21`) | `underthesea.word_tokenize(format="text")` ghép "nghỉ_phép" thành 1 token rồi replace `_` → " ". Không segment thì BM25 coi "nghỉ" và "phép" là 2 token độc lập, query "nghỉ phép" match nhầm mọi tài liệu có chữ "phép". |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` (`src/m2_search.py:108`) | `score(d) = Σ 1/(k + rank + 1)`, k=60. RRF giải quyết bài toán **không so sánh được thang điểm**: BM25 trả điểm không chuẩn hoá (0–20+), cosine của bge-m3 trả 0–1. RRF chỉ dùng *thứ hạng* nên không cần normalize. Tài liệu chứa số liệu (25 triệu, 90 ngày) được BM25 kéo lên, còn câu hỏi diễn giải lại (paraphrase) được dense kéo lên. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` (`src/m3_rerank.py:39`) | `bge-reranker-v2-m3` chấm trực tiếp cặp (query, doc) thay vì so 2 vector độc lập → top-20 hybrid rút còn top-3. Xem bảng latency ở dưới: rerank là bước tốn thời gian nhất trong retrieval, đổi latency lấy precision. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` (`src/m4_eval.py:30`) | 4 metric chia làm 2 nhóm rõ rệt: `context_precision`/`context_recall` chấm **retrieval**, `faithfulness`/`answer_relevancy` chấm **generation**. Nhìn cặp metric nào thấp là biết lỗi nằm ở nửa nào của pipeline. |
| Failure analysis / Diagnostic Tree | M4 | `failure_analysis()` (`src/m4_eval.py:86`) | Map `worst_metric → (diagnosis, suggested_fix)`. Sắp xếp theo điểm trung bình 4 metric rồi lấy bottom-N — cách rẻ nhất để biết nên sửa module nào trước. |
| Contextual embeddings / enrichment | M5 | `_enrich_single_call()` (`src/m5_enrichment.py:148`) | Dùng combined mode: **1 API call/chunk** thay vì 4 (summary + hypothesis questions + context line + metadata trong 1 JSON). Với 116 chunks tiết kiệm 348 request — quan trọng vì free tier Gemini giới hạn theo ngày. `contextual_prepend` gắn 1 câu mô tả chunk nằm ở đâu trong tài liệu, giúp chunk mồ côi kiểu "Mức 3: 50 triệu trở lên" vẫn biết nó thuộc quy trình mua sắm. |

---

## Phần 2: Khó khăn & cách giải quyết

### 2.1. RAGAS trả `nan` cho cả 4 metric

**Exact error:**

```
Exception raised in Job[3]: TypeError(GenerativeServiceClient.generate_content()
got an unexpected keyword argument 'temperature')
```

**Debug:** RAGAS 0.1.22 gọi `langchain_llm.generate_prompt(..., temperature=...)` ở *call time*. `ChatGoogleGenerativeAI` (langchain-google-genai 1.0.10) chỉ nhận `temperature` ở *constructor*, nên forward thẳng xuống `generate_content()` và vỡ. Bốn job đều fail → `result.to_pandas()` toàn `nan`.

**Fix:** bỏ client Gemini native, dùng `ChatOpenAI` trỏ vào endpoint OpenAI-compat của Google (`src/llm.py:get_ragas_llm`). Thời gian eval 1 câu hỏi giảm từ **3 phút (4/4 job fail)** xuống **3 giây** với điểm số thật.

### 2.2. `Multiple candidates is not enabled for this model`

**Exact error:**

```
Error code: 400 - [{'error': {'code': 400, 'message': 'Multiple candidates is not
enabled for this model', 'status': 'INVALID_ARGUMENT'}}]
```

**Debug:** metric `answer_relevancy` có `strictness=3` → RAGAS sinh 3 câu hỏi ngược. Nếu LLM nằm trong `MULTIPLE_COMPLETION_SUPPORTED` (mà `ChatOpenAI` thì có), RAGAS gửi `n=3` lên API. Endpoint OpenAI-compat của Gemini không hỗ trợ `n>1`. Đây chính là lý do code ban đầu chọn client native — nhưng client native lại vỡ vì lỗi 2.1.

**Fix:** vá `ragas.llms.base.is_multiple_completion_supported` → `False` (`src/_ragas_compat.py:disable_multiple_completions`). RAGAS chuyển sang gửi cùng prompt 3 lần tuần tự — chậm hơn nhưng chạy được, và code fallback này đã có sẵn trong RAGAS.

### 2.3. LLM trả chuỗi rỗng dù API 200 OK

**Triệu chứng:** `chat()` trả `''`, không exception. Dump response thấy:

```
finish_reason='length', message.content=None,
usage=CompletionUsage(completion_tokens=0, prompt_tokens=5, total_tokens=22)
```

**Debug:** `completion_tokens=0` nhưng `total_tokens=22` → 17 token bị tiêu ở phần **thinking**. Gemini 3.x là thinking model, reasoning token tính vào `max_tokens`. Với `max_tokens=300` cho prompt enrichment dài, phần trả lời thật gần như không còn chỗ.

**Fix:** thêm `reasoning_effort="low"` và nhân 4 `max_tokens` cho nhánh Gemini (`src/llm.py:chat`).

### 2.4. Hết quota free tier giữa chừng

**Exact error:**

```
Error code: 429 - Quota exceeded for metric:
generativelanguage.googleapis.com/generate_content_free_tier_requests,
limit: 20, model: gemini-3.6-flash
```

**Debug:** `gemini-3.6-flash` chỉ có **20 request/ngày** ở free tier. Pipeline cần ~116 call enrichment + 40 call sinh câu trả lời + vài trăm call RAGAS judge. Thử `gemini-2.5-flash` thì nhận `404 - This model is no longer available to new users`.

**Fix ba lớp:**
1. Đổi sang `gemini-3.5-flash-lite` (quota cao hơn hẳn) và tách `GEMINI_JUDGE_MODEL` riêng để sau này nâng model judge mà không đổi model enrichment.
2. Retry với exponential backoff, đọc `retryDelay` do API trả về (`src/llm.py:_retry_delay`).
3. Cache enrichment trên đĩa theo `sha1(model|source|text)` (`src/m5_enrichment.py:_enrich_single_call`) — chạy lại pipeline không tốn thêm request nào.

Ngoài ra hạ `RunConfig(max_workers=4)` cho RAGAS (mặc định 16) để không đấm rate limit theo phút.

### 2.5. CUDA out of memory

**Exact error:**

```
⚠️  RAGAS evaluation failed: CUDA out of memory. Tried to allocate 16.00 MiB.
GPU 0 has a total capacity of 3.68 GiB of which 15.44 MiB is free.
```

**Debug:** RTX 3050 Laptop chỉ có 3.68 GiB. `DenseSearch` giữ `bge-m3` (~2.2 GB) trên GPU suốt quá trình eval, RAGAS lại nạp thêm một `bge-m3` nữa cho `answer_relevancy`, chưa kể cross-encoder 2.2 GB của M3.

**Fix:** phân bổ device theo khối lượng công việc (`config.py`): retrieval encoder giữ trên GPU vì phải encode 100+ chunk; reranker và embeddings của RAGAS đẩy xuống CPU vì mỗi query chỉ vài chục cặp.

### 2.6. Kiến thức còn thiếu → cách bổ sung

- **Chưa nắm cơ chế nội bộ RAGAS:** phải đọc thẳng `.venv/.../ragas/llms/base.py` mới hiểu `is_multiple_completion_supported` và luồng `generate_text`. Bài học: khi thư viện lỗi, đọc source cài trong venv nhanh hơn tra docs.
- **Chưa biết thinking model tính token thế nào:** khắc phục bằng cách luôn in `usage` và `finish_reason` khi response rỗng, thay vì chỉ log exception.
- **Chưa quen tune threshold cho semantic chunking tiếng Việt:** dự định benchmark 0.5/0.65/0.8 với encoder `bge-m3` thay cho MiniLM.

---

## Phần 3: Action Plan cho project cá nhân

### Project: Trợ lý hỏi–đáp nội bộ trên tài liệu quy trình công ty (Vietnamese RAG)

**Hiện tại**
- Pipeline: chunk cố định 500 ký tự → embed → dense search top-5 → LLM trả lời. Không hybrid, không rerank, không đo lường.
- Known issues: (1) câu hỏi chứa số liệu ("phạt bao nhiêu %", "trên 50 triệu") thường retrieve trượt vì dense search coi nhẹ con số; (2) tài liệu có nhiều phiên bản (`nghi_phep_nam_v2023` vs `v2024`, `mat_khau_v1` vs `v2`) nên hệ thống trả lời theo bản cũ; (3) không có metric nào để biết sửa có tốt lên không.

**Plan áp dụng**

1. [ ] **Chunking:** dùng `chunk_hierarchical` (parent 2048 / child 256). Lý do: tài liệu quy trình có cấu trúc "mục lớn → điều khoản nhỏ"; retrieve child cho precision, trả parent cho LLM đủ ngữ cảnh. Bỏ semantic chunking vì threshold rất nhạy với encoder và không ổn định trên tiếng Việt (bằng chứng: 208 vs 51 chunks ở trên).
2. [ ] **Search:** Hybrid BM25 + Dense + RRF. Lý do: đúng vào issue (1) — BM25 bắt số liệu và mã điều khoản, dense bắt paraphrase. Bắt buộc `segment_vietnamese()` trước khi index BM25.
3. [ ] **Reranking:** có, dùng `bge-reranker-v2-m3` trên CPU, top-20 → top-3. Nếu latency vượt ngân sách thì hạ xuống top-10 → top-3 hoặc thử `flashrank` (đã có sẵn khung `FlashrankReranker`).
4. [ ] **Evaluation:** RAGAS 4 metric làm chuẩn, chạy trên test set ≥ 30 câu tự viết theo tài liệu thật. Bổ sung 1 custom metric "version correctness" — kiểm tra câu trả lời có trích đúng phiên bản tài liệu mới nhất không, vì đây là issue (2) mà RAGAS không bắt được.
5. [ ] **Enrichment:** ưu tiên `contextual_prepend` + auto metadata, chạy combined single-call. Metadata `effective_date` + `version` để filter phiên bản cũ ngay ở tầng retrieval — đánh trực diện issue (2).

**Timeline**

| Tuần | Việc |
|------|------|
| Tuần 1 | Dựng test set 30 câu + chạy RAGAS trên pipeline hiện tại để có baseline. Không sửa code. |
| Tuần 2 | Thay chunking sang hierarchical, bật hybrid search + RRF. Đo lại, so với baseline tuần 1. |
| Tuần 3 | Thêm reranking + latency breakdown. Chốt ngân sách latency p95 cho 1 câu hỏi. |
| Tuần 4 | Enrichment (contextual + metadata version/effective_date) + filter phiên bản. Chạy failure analysis bottom-5, viết report so sánh 4 mốc. |

---

## Phụ lục: Kết quả đo được

Xem `analysis/failure_analysis.md` và `ragas_report.json`.
