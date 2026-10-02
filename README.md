# LegalRAG class-based pipeline

Bản này giữ lại hướng chạy thí nghiệm ban đầu:

1. Build embedding, FAISS, BM25 và graph khi cần.
2. Load bộ câu hỏi đánh giá.
3. Chia train/test cố định bằng random seed.
4. Tune các cấu hình retrieval/graph trên train.
5. Chọn variant tốt nhất theo `evaluation.selection_metric`.
6. Chạy đúng một lần trên test với variant đã chọn.
7. Lưu metrics tổng hợp và kết quả từng query.

## Chuẩn bị dữ liệu

```text
data/legal_chunks.parquet
data/evaluation_queries.jsonl
embedding_output/embeddings.npy
embedding_output/embedding_metadata.parquet
faiss_output/legal_chunks.faiss
bm25_output/
graph/nodes.parquet
graph/edges.parquet
```

`evaluation_queries.jsonl` hỗ trợ dạng:

```json
{"query_id":"q_0001","query":"Nội dung câu hỏi","relevant_chunk_ids":["123_dieu_1_0001"]}
```

## Chạy train/tune + test + evaluation

Đây là chế độ mặc định:

```bash
python -m src.main
```

hoặc:

```bash
python -m src.main --mode experiment
```

Pipeline sẽ tune các variant trong `evaluation.variants` trên 300 train query, chọn variant tốt nhất theo `mrr@10`, sau đó đánh giá trên 100 test query.

## Chỉ build index và graph

```bash
python -m src.main --mode build
```

Để build trước khi evaluation trong cùng một lệnh, đặt:

```json
"rebuild_before_evaluation": true
```

## Chạy thử một query

```bash
python -m src.main --mode search --query "Điều kiện cấp giấy chứng nhận là gì?"
```

## Kết quả đầu ra

```text
evaluation_output/experiment_report.json
evaluation_output/variant_summary.csv
evaluation_output/train_query_results.jsonl
evaluation_output/test_query_results.jsonl
```

Báo cáo có các metric:

- Recall@K
- Precision@K
- Hit@K
- MRR@K

Và các diagnostics:

- `mapped_zero`: không seed retrieval nào map được vào graph.
- `graph_result_zero`: graph expansion không sinh candidate.
- `match_zero`: top-K không chứa relevant chunk.
- `seed_mapping_rate`: tỷ lệ retrieval seed map được vào graph.

## Schema graph hiện tại

`edges.parquet` của project dùng:

```text
source
target
edge_type
reference_count
```

Các tên này đã được đặt đúng trong `config.json`. `GraphExpander` vẫn có cơ chế tự nhận diện schema cũ như `source_node_id/target_node_id`.
