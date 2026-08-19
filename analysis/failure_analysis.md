# Failure Analysis — Lab 18: Production RAG

**Họ tên:** Phùng Đình Đạt · **MSSV:** 2A202601540 · **Lớp:** AICB-K34
**Bài cá nhân** — implement cả 5 modules (M1–M5).

---

## 1. RAGAS Scores — Naive Baseline vs Production

- **Naive baseline:** paragraph chunking (500 ký tự) → dense-only search top-3 → LLM trả lời.
- **Production:** hierarchical chunking (2048/256) → enrichment 1 call/chunk → hybrid BM25+Dense+RRF top-20 → cross-encoder rerank top-3 → LLM trả lời.
- Cùng judge model (`gemini-3.1-flash-lite`), cùng test set 20 câu, cùng prompt sinh câu trả lời, cả hai đều chấm đủ 20/20 câu.

| Metric | Naive Baseline | Production | Δ |
|--------|---------------:|-----------:|---:|
| Faithfulness | 0.8500 | 0.8900 | +0.0400 |
| Answer Relevancy | 0.6906 | 0.7304 | +0.0398 |
| Context Precision | 0.7500 | 0.6958 | -0.0542 |
| Context Recall | 0.8250 | 0.8083 | -0.0167 |

**Số câu chấm được / 20** (RAGAS trả NaN khi không trích được statement từ câu trả lời quá ngắn — NaN bị loại khỏi trung bình, không quy về 0):

| Metric | Naive | Production |
|--------|------:|-----------:|
| Faithfulness | 20/20 | 20/20 |
| Answer Relevancy | 20/20 | 20/20 |
| Context Precision | 20/20 | 20/20 |
| Context Recall | 20/20 | 20/20 |

## 2. Latency Breakdown (production, ms/query)

| Stage | avg | p50 | p95 |
|-------|----:|----:|----:|
| Hybrid search (BM25+Dense+RRF) | 31.8 | 30.3 | 83.5 |
| Cross-encoder rerank (CPU) | 6019.7 | 5767.0 | 12569.5 |
| LLM sinh câu trả lời | 1254.4 | 1296.9 | 1646.5 |
| **TỔNG** | 7305.9 | 7109.5 | 13759.1 |

## 3. Đọc kết quả

**Nhóm generation tăng, nhóm retrieval giảm.**

- `faithfulness` +0.04 và `answer_relevancy` +0.04: enrichment (contextual prepend) gắn 1 câu mô tả chunk nằm ở đâu trong tài liệu, nên LLM ít suy diễn ngoài context hơn. Rerank cắt top-20 xuống top-3 cũng làm prompt sạch, bớt nhiễu.
- `context_precision` -0.05 và `context_recall` -0.02: đây là cái giá của chunk nhỏ. Baseline chunk 500 ký tự (51 chunks, avg 410) — mỗi chunk gần như trọn một mục chính sách, nên hầu như luôn chứa đủ câu trả lời. Production chunk child 256 ký tự (116 chunks, avg 191) nên một điều khoản bị tách làm 2–3 mảnh; lấy top-3 sau rerank thì dễ trúng mảnh chứa tiêu đề mà thiếu mảnh chứa con số, hoặc ngược lại.
- Câu chốt: **hierarchical chunking chỉ phát huy khi retrieve child rồi TRẢ VỀ parent.** Pipeline hiện tại đã gắn `parent_id` vào metadata nhưng bước `run_query` vẫn trả thẳng text của child. Đây là fix rẻ nhất và có tác động lớn nhất — xem mục 5.

## 4. Bottom-5 Failures

### #1 — Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?

- **Expected (ground truth):** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = 18 ngày phép. Lương Senior (P3-P4): 20-35 triệu VNĐ/tháng.
- **Got (answer):** Dựa trên context: - Nhân viên có 9 năm thâm niên được **18 ngày phép** (15 + 3). - Context không đề cập đến mức lương của nhân viên này. Không tìm thấy thông tin về mức lương.
- **Điểm:** Faithfulness 1.00 · Answer Relevancy 0.00 · Context Precision 0.00 · Context Recall 0.50
- **Worst metric:** `answer_relevancy` = 0.0000 (avg 4 metric = 0.3750)
- **Diagnosis (Diagnostic Tree):** Answer doesn't match question
- **Suggested fix:** Improve prompt template
- **Contexts retrieved (top-3 sau rerank):**
  1. Đoạn văn nằm ở phần quy định về thâm niên công tác trong chính sách phép năm. ## Thâm niên công tác Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho …
  2. Đoạn văn nằm ở phần quy định về số ngày phép năm trong tài liệu chính sách nghỉ phép. ## Số ngày phép năm Mỗi nhân viên chính thức được hưởng **12 ngày phép năm** có lương. Số ngà…
  3. Đoạn văn nằm ở phần 'Điều kiện áp dụng' trong tài liệu quy định về nghỉ phép không lương. ## Điều kiện áp dụng Nhân viên chính thức đã hết phép năm có thể xin nghỉ phép không lươn…

### #2 — Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?

- **Expected (ground truth):** KHÔNG. Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI. Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got (answer):** Không tìm thấy.
- **Điểm:** Faithfulness 0.00 · Answer Relevancy 0.00 · Context Precision 0.58 · Context Recall 1.00
- **Worst metric:** `faithfulness` = 0.0000 (avg 4 metric = 0.3958)
- **Diagnosis (Diagnostic Tree):** LLM hallucinating
- **Suggested fix:** Tighten prompt, lower temperature
- **Contexts retrieved (top-3 sau rerank):**
  1. Đoạn văn nằm ở phần cuối của tài liệu quy định về chính sách thử việc, ngay sau phần đề cập đến quyền lợi bảo hiểm sức khỏe PVI. ược hưởng gói bảo hiểm sức khỏe PVI. ## Đánh giá t…
  2. Đoạn văn nằm ở phần phúc lợi nhân sự trong tài liệu hướng dẫn về bảo hiểm sức khỏe của công ty. ## Bảo hiểm cho nhân viên Công ty cung cấp gói bảo hiểm sức khỏe toàn diện qua **PV…
  3. Đoạn văn nằm ở phần quy định về quyền lợi dành cho nhân viên trong giai đoạn thử việc của tài liệu chính sách nhân sự. ## Quyền lợi trong thử việc Nhân viên thử việc **KHÔNG được …

### #3 — Muốn mua thiết bị trị giá 55 triệu cần ai phê duyệt?

- **Expected (ground truth):** Đơn hàng trên 50.000.000 VNĐ cần Tổng Giám đốc (CEO) phê duyệt.
- **Got (answer):** Không tìm thấy.
- **Điểm:** Faithfulness 1.00 · Answer Relevancy 0.00 · Context Precision 1.00 · Context Recall 0.00
- **Worst metric:** `answer_relevancy` = 0.0000 (avg 4 metric = 0.5000)
- **Diagnosis (Diagnostic Tree):** Answer doesn't match question
- **Suggested fix:** Improve prompt template
- **Contexts retrieved (top-3 sau rerank):**
  1. Đoạn văn nằm ở phần quy định về thẩm quyền phê duyệt trong tài liệu hướng dẫn mua sắm. ## Thẩm quyền phê duyệt | Giá trị đơn hàng | Người phê duyệt | |-------------------|--------…
  2. Đoạn văn nằm ở phần quy định về phê duyệt trong tài liệu hướng dẫn tạm ứng. ## Phê duyệt Tạm ứng dưới 5.000.000 VNĐ: trưởng phòng phê duyệt. Từ 5.000.000 VNĐ trở lên: cần thêm phê…
  3. Đoạn văn nằm trong phần quy định về phê duyệt chi phí của tài liệu hướng dẫn chi phí. ## Phê duyệt Tất cả chi phí cần được phê duyệt bởi **quản lý trực tiếp** trước khi nộp phòng …

### #4 — Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?

- **Expected (ground truth):** Laptop 30 triệu nằm trong khoảng 5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt. Ngoài ra, mua sắm thiết bị CNTT cần có xác nhận cấu hình kỹ thuật từ phòng CNTT trước khi đề xuất. Cần đính kèm ít nhất 3 báo …
- **Got (answer):** Dựa trên context: - **Ai phê duyệt:** Đơn hàng trị giá 30.000.000 VNĐ nằm trong khoảng từ 5.000.000 - 50.000.000 VNĐ, do **Giám đốc phòng ban (Director)** phê duyệt. - **Cần gì từ phòng CNTT:** Cần có **xác nhận của phò…
- **Điểm:** Faithfulness 0.80 · Answer Relevancy 0.89 · Context Precision 0.00 · Context Recall 0.67
- **Worst metric:** `context_precision` = 0.0000 (avg 4 metric = 0.5894)
- **Diagnosis (Diagnostic Tree):** Too many irrelevant chunks
- **Suggested fix:** Add reranking or metadata filter
- **Contexts retrieved (top-3 sau rerank):**
  1. Đoạn văn nằm trong phần 'Lưu ý đặc biệt' của tài liệu hướng dẫn mua sắm. ## Lưu ý đặc biệt Mua sắm thiết bị CNTT (laptop, server, phần mềm) cần có xác nhận của phòng CNTT về cấu h…
  2. Đoạn văn nằm ở phần quy định về thẩm quyền phê duyệt trong tài liệu hướng dẫn mua sắm. ## Thẩm quyền phê duyệt | Giá trị đơn hàng | Người phê duyệt | |-------------------|--------…
  3. Đoạn văn nằm ở phần quy định về phê duyệt trong tài liệu hướng dẫn tạm ứng. ## Phê duyệt Tạm ứng dưới 5.000.000 VNĐ: trưởng phòng phê duyệt. Từ 5.000.000 VNĐ trở lên: cần thêm phê…

### #5 — Thâm niên bao nhiêu năm thì được cộng thêm ngày phép?

- **Expected (ground truth):** Theo chính sách v2024 hiện hành, nhân viên có thâm niên từ 3 năm trở lên được cộng thêm 1 ngày phép cho mỗi 3 năm. Chính sách cũ v2023 yêu cầu 5 năm.
- **Got (answer):** Dựa trên context, có hai quy định khác nhau được nhắc đến: - Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm 1 ngày phép cho mỗi 3 năm làm việc liên tục. - Nhân viên có thâm niên từ **5 năm trở lên** được cộn…
- **Điểm:** Faithfulness 1.00 · Answer Relevancy 0.83 · Context Precision 0.00 · Context Recall 1.00
- **Worst metric:** `context_precision` = 0.0000 (avg 4 metric = 0.7083)
- **Diagnosis (Diagnostic Tree):** Too many irrelevant chunks
- **Suggested fix:** Add reranking or metadata filter
- **Contexts retrieved (top-3 sau rerank):**
  1. Đoạn văn nằm ở phần quy định về thâm niên công tác trong chính sách phép năm. ## Thâm niên công tác Nhân viên có thâm niên từ **3 năm trở lên** được cộng thêm **1 ngày phép** cho …
  2. Đoạn văn nằm trong phần quy định về thâm niên công tác của chính sách phép năm 2023. ## Thâm niên công tác Nhân viên có thâm niên từ **5 năm trở lên** được cộng thêm **1 ngày phép…
  3. Đoạn văn nằm ở phần 'Điều kiện áp dụng' trong tài liệu quy định về nghỉ phép không lương. ## Điều kiện áp dụng Nhân viên chính thức đã hết phép năm có thể xin nghỉ phép không lươn…

## 5. Error Tree + hành động tiếp theo

Đi theo Diagnostic Tree cho nhóm lỗi lớn nhất (`context_precision` thấp):

```
Output sai?
 └─ Có → Context có chứa câu trả lời không?
     ├─ Không → lỗi RETRIEVAL
     │    └─ BM25 có tìm ra không?  → có, nhưng chunk bị cắt đôi
     │    └─ Dense có tìm ra không? → có, cùng vấn đề
     │    └─ Rerank có đẩy nó lên top-3 không? → đẩy nhầm mảnh
     │         => Fix ở bước CHUNKING/ASSEMBLY, không phải ở search
     └─ Có → lỗi GENERATION (prompt / hallucination)
```

**Nếu có thêm 1 giờ, làm theo thứ tự này:**

1. **Trả parent thay vì child** (~15 phút, tác động lớn nhất). Index child để retrieve chính xác, nhưng khi build context thì dùng `parent_id` lấy nguyên parent 2048 ký tự. Kỳ vọng `context_recall` vượt baseline vì không còn điều khoản bị cắt đôi.
2. **Dedupe theo `parent_id` trước khi rerank** (~10 phút). Hiện top-20 hybrid có thể chứa 3 mảnh của cùng một parent, chiếm chỗ của tài liệu khác — trực tiếp kéo `context_precision` xuống.
3. **Index cả hypothesis questions** (~15 phút). M5 đã sinh sẵn 3 câu hỏi/chunk và lưu trong cache nhưng chưa được đưa vào text đem embed. Đây là dữ liệu đã trả tiền API rồi mà chưa dùng.
4. **Đưa reranker lên GPU khi không index** (~20 phút). Rerank đang chiếm 82% latency (6020ms/7306ms) vì chạy CPU. Giải phóng bge-m3 khỏi GPU sau khi index xong là đủ chỗ cho cross-encoder.

---

_Sinh tự động từ `naive_baseline_report.json` + `ragas_report.json`._