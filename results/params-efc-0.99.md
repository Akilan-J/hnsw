| efConstruction | HNSW |
|---:|---:|
| 50 | 1,295 |
| 100 | 1,124 |
| 200 | 1,114 |
| 400 | 1,059 |

distances per query needed to reach recall@10 = 0.99, interpolated between efSearch points. &le; means even the smallest ef measured was past the
target, so that cost is an upper bound; "not reached" means the largest ef measured fell short.
