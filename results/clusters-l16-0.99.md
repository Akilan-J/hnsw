| clusters | HNSW | flat + heuristic |
|---:|---:|---:|
| 1 | 1,343 | 1,357 |
| 64 | 443 | 893 |

distances per query needed to reach recall@10 = 0.99, interpolated between efSearch points. &le; means even the smallest ef measured was past the
target, so that cost is an upper bound; "not reached" means the largest ef measured fell short.
