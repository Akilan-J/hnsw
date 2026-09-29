| vectors indexed | HNSW | flat + heuristic | flat NSW (stage 2) |
|---:|---:|---:|---:|
| 100000 | 627 | 726 | 1,026 |
| 300000 | 827 | 883 | 1,464 |
| 1000000 | 1,200 | 1,244 | 2,117 |

distances per query needed to reach recall@10 = 0.95, interpolated between efSearch points. &le; means even the smallest ef measured was past the
target, so that cost is an upper bound; "not reached" means the largest ef measured fell short.
