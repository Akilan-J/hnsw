| clusters | HNSW | flat + heuristic |
|---:|---:|---:|
| 1 | 8,473 | 8,444 |
| 64 | 653 | 1,257 |

distances per query needed to reach recall@10 = 0.95, interpolated between efSearch points. &le; means even the smallest ef measured was past the
target, so that cost is an upper bound; "not reached" means the largest ef measured fell short.
