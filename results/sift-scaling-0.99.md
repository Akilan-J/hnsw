| vectors indexed | HNSW | flat + heuristic | flat NSW (stage 2) |
|---:|---:|---:|---:|
| 100000 | 1,124 | 1,195 | 1,739 |
| 300000 | 1,731 | 1,745 | 2,601 |
| 1000000 | 2,788 | 2,733 | 4,109 |

distances per query needed to reach recall@10 = 0.99, interpolated between efSearch points. &le; means even the smallest ef measured was past the
target, so that cost is an upper bound; "not reached" means the largest ef measured fell short.
