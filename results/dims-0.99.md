| latent (intrinsic) dimension | HNSW | flat + heuristic |
|---:|---:|---:|
| 4 | &le; 134 | &le; 221 |
| 8 | 252 | 323 |
| 16 | 443 | 893 |
| 32 | 684 | 1,382 |
| 64 | 896 | 1,175 |
| 128 | 927 | 1,555 |

distances per query needed to reach recall@10 = 0.99, interpolated between efSearch points. &le; means even the smallest ef measured was past the
target, so that cost is an upper bound; "not reached" means the largest ef measured fell short.
