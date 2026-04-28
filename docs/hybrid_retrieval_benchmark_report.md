# Hybrid Retrieval Benchmark Report

## Methodology

- Recall@10 is computed as relevant_count / 10 for the retrieved top-10 results.
- NDCG@10 uses binary relevance labels from the judge caches.
- Hybrid scores are built from the top-10 dense and sparse scores after per-source min-max normalization.
- RRF uses 1 / (1 + rank) for ranks 1 to 10 from each retriever.
- Pilot hybrid uses a 50-query pilot pool, evaluates alpha values from 0.00 to 1.00 in 0.05 steps, and sets the final alpha to the mean of the top 3 alphas by Recall@10 on the pilot pool.
- Pilot hybrid is skipped for datasets with fewer than 100 queries.
- The front dataset section shows AIHub topics merged by TL+VL pair; KorMedMCQA appears only in the group-level section.

## Dataset-Level Analysis

### 기타(TL+VL)

- Queries: 2030 | Pilot holdout: 1730
- Pilot hybrid: final alpha=0.4500, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.3000, 0.5000, 0.5500], holdout=1730, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.0939    | 0.2232  | BM25 top-10        |
| Dense only                | 0.1003    | 0.2312  | Dense top-10       |
| Alpha sweep (best recall) | 0.1067    | 0.2498  | alpha=0.4000       |
| Alpha sweep (best NDCG)   | 0.1059    | 0.2522  | alpha=0.5000       |
| RRF                       | 0.1054    | 0.2513  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.1045    | 0.2475  | alpha=0.4500       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.0984    | 0.2292  |
| 0.10  | 0.1022    | 0.2355  |
| 0.20  | 0.1041    | 0.2405  |
| 0.30  | 0.1058    | 0.2451  |
| 0.40  | 0.1067    | 0.2498  |
| 0.50  | 0.1059    | 0.2522  |
| 0.60  | 0.1063    | 0.2520  |
| 0.70  | 0.1061    | 0.2496  |
| 0.80  | 0.1053    | 0.2455  |
| 0.90  | 0.1040    | 0.2402  |
| 1.00  | 0.1004    | 0.2313  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.3000 | 0.1157    | 0.2585  |
| 2    | 0.5000 | 0.1150    | 0.2639  |
| 3    | 0.5500 | 0.1150    | 0.2636  |

### 마취통증의학과(TL+VL)

- Queries: 836 | Pilot holdout: 536
- Pilot hybrid: final alpha=0.0833, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.0000, 0.0500, 0.2000], holdout=536, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.1359    | 0.2769  | BM25 top-10        |
| Dense only                | 0.1184    | 0.2569  | Dense top-10       |
| Alpha sweep (best recall) | 0.1395    | 0.2831  | alpha=0.0000       |
| Alpha sweep (best NDCG)   | 0.1342    | 0.2862  | alpha=0.4000       |
| RRF                       | 0.1344    | 0.2863  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.1375    | 0.2866  | alpha=0.0833       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.1395    | 0.2831  |
| 0.10  | 0.1378    | 0.2830  |
| 0.20  | 0.1377    | 0.2852  |
| 0.30  | 0.1362    | 0.2857  |
| 0.40  | 0.1342    | 0.2862  |
| 0.50  | 0.1340    | 0.2859  |
| 0.60  | 0.1331    | 0.2837  |
| 0.70  | 0.1323    | 0.2810  |
| 0.80  | 0.1294    | 0.2750  |
| 0.90  | 0.1267    | 0.2698  |
| 1.00  | 0.1184    | 0.2569  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.0000 | 0.1437    | 0.2790  |
| 2    | 0.0500 | 0.1420    | 0.2781  |
| 3    | 0.2000 | 0.1417    | 0.2820  |

### 방사선종양학과(TL+VL)

- Queries: 101
- Pilot hybrid: skipped (query_count<600)

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.1238    | 0.3209  | BM25 top-10        |
| Dense only                | 0.1525    | 0.3427  | Dense top-10       |
| Alpha sweep (best recall) | 0.1554    | 0.3628  | alpha=0.7000       |
| Alpha sweep (best NDCG)   | 0.1505    | 0.3705  | alpha=0.5000       |
| RRF                       | 0.1525    | 0.3725  | 1/(1+rank), top-10 |
| Pilot hybrid              | SKIPPED   | SKIPPED | query_count<600    |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.1347    | 0.3321  |
| 0.10  | 0.1446    | 0.3445  |
| 0.20  | 0.1465    | 0.3521  |
| 0.30  | 0.1465    | 0.3580  |
| 0.40  | 0.1465    | 0.3678  |
| 0.50  | 0.1505    | 0.3705  |
| 0.60  | 0.1515    | 0.3678  |
| 0.70  | 0.1554    | 0.3628  |
| 0.80  | 0.1554    | 0.3615  |
| 0.90  | 0.1535    | 0.3534  |
| 1.00  | 0.1525    | 0.3427  |

### 병리과(TL+VL)

- Queries: 136
- Pilot hybrid: skipped (query_count<600)

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.0625    | 0.1771  | BM25 top-10        |
| Dense only                | 0.0699    | 0.1788  | Dense top-10       |
| Alpha sweep (best recall) | 0.0721    | 0.1886  | alpha=0.9000       |
| Alpha sweep (best NDCG)   | 0.0713    | 0.1906  | alpha=0.8000       |
| RRF                       | 0.0676    | 0.1938  | 1/(1+rank), top-10 |
| Pilot hybrid              | SKIPPED   | SKIPPED | query_count<600    |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.0625    | 0.1802  |
| 0.10  | 0.0596    | 0.1802  |
| 0.20  | 0.0610    | 0.1809  |
| 0.30  | 0.0625    | 0.1816  |
| 0.40  | 0.0662    | 0.1875  |
| 0.50  | 0.0662    | 0.1896  |
| 0.60  | 0.0669    | 0.1884  |
| 0.70  | 0.0691    | 0.1883  |
| 0.80  | 0.0713    | 0.1906  |
| 0.90  | 0.0721    | 0.1886  |
| 1.00  | 0.0699    | 0.1788  |

### 비뇨의학과(TL+VL)

- Queries: 882 | Pilot holdout: 582
- Pilot hybrid: final alpha=0.4000, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.4000, 0.3500, 0.4500], holdout=582, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.2502    | 0.4689  | BM25 top-10        |
| Dense only                | 0.2370    | 0.4345  | Dense top-10       |
| Alpha sweep (best recall) | 0.2705    | 0.4937  | alpha=0.2000       |
| Alpha sweep (best NDCG)   | 0.2704    | 0.5011  | alpha=0.4000       |
| RRF                       | 0.2680    | 0.4963  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.2680    | 0.4986  | alpha=0.4000       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.2617    | 0.4797  |
| 0.10  | 0.2669    | 0.4860  |
| 0.20  | 0.2705    | 0.4937  |
| 0.30  | 0.2696    | 0.4975  |
| 0.40  | 0.2704    | 0.5011  |
| 0.50  | 0.2693    | 0.5006  |
| 0.60  | 0.2677    | 0.4903  |
| 0.70  | 0.2659    | 0.4819  |
| 0.80  | 0.2633    | 0.4745  |
| 0.90  | 0.2588    | 0.4644  |
| 1.00  | 0.2370    | 0.4345  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.4000 | 0.2750    | 0.5062  |
| 2    | 0.3500 | 0.2730    | 0.5061  |
| 3    | 0.4500 | 0.2723    | 0.5010  |

### 신경과신경외과(TL+VL)

- Queries: 1933 | Pilot holdout: 1633
- Pilot hybrid: final alpha=0.5000, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.4500, 0.5500, 0.5000], holdout=1633, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.1600    | 0.3535  | BM25 top-10        |
| Dense only                | 0.1662    | 0.3578  | Dense top-10       |
| Alpha sweep (best recall) | 0.1781    | 0.3928  | alpha=0.5000       |
| Alpha sweep (best NDCG)   | 0.1781    | 0.3928  | alpha=0.5000       |
| RRF                       | 0.1770    | 0.3894  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.1778    | 0.3893  | alpha=0.5000       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.1663    | 0.3608  |
| 0.10  | 0.1721    | 0.3712  |
| 0.20  | 0.1745    | 0.3790  |
| 0.30  | 0.1756    | 0.3843  |
| 0.40  | 0.1762    | 0.3892  |
| 0.50  | 0.1781    | 0.3928  |
| 0.60  | 0.1781    | 0.3883  |
| 0.70  | 0.1772    | 0.3836  |
| 0.80  | 0.1758    | 0.3777  |
| 0.90  | 0.1752    | 0.3727  |
| 1.00  | 0.1662    | 0.3577  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.4500 | 0.1807    | 0.4161  |
| 2    | 0.5500 | 0.1807    | 0.4118  |
| 3    | 0.5000 | 0.1793    | 0.4118  |

### 안과(TL+VL)

- Queries: 684 | Pilot holdout: 384
- Pilot hybrid: final alpha=0.7500, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.8000, 0.7500, 0.7000], holdout=384, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.2111    | 0.4052  | BM25 top-10        |
| Dense only                | 0.2411    | 0.4544  | Dense top-10       |
| Alpha sweep (best recall) | 0.2501    | 0.4736  | alpha=0.8000       |
| Alpha sweep (best NDCG)   | 0.2501    | 0.4736  | alpha=0.8000       |
| RRF                       | 0.2455    | 0.4725  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.2437    | 0.4639  | alpha=0.7500       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.2208    | 0.4150  |
| 0.10  | 0.2338    | 0.4350  |
| 0.20  | 0.2364    | 0.4434  |
| 0.30  | 0.2389    | 0.4523  |
| 0.40  | 0.2439    | 0.4618  |
| 0.50  | 0.2452    | 0.4725  |
| 0.60  | 0.2474    | 0.4729  |
| 0.70  | 0.2481    | 0.4730  |
| 0.80  | 0.2501    | 0.4736  |
| 0.90  | 0.2497    | 0.4697  |
| 1.00  | 0.2411    | 0.4544  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.8000 | 0.2577    | 0.4887  |
| 2    | 0.7500 | 0.2567    | 0.4871  |
| 3    | 0.7000 | 0.2563    | 0.4894  |

### 예방의학(TL+VL)

- Queries: 675 | Pilot holdout: 375
- Pilot hybrid: final alpha=0.6500, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.6000, 0.6500, 0.7000], holdout=375, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.0679    | 0.1936  | BM25 top-10        |
| Dense only                | 0.0670    | 0.1882  | Dense top-10       |
| Alpha sweep (best recall) | 0.0735    | 0.2147  | alpha=0.5000       |
| Alpha sweep (best NDCG)   | 0.0735    | 0.2147  | alpha=0.5000       |
| RRF                       | 0.0745    | 0.2155  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.0696    | 0.2039  | alpha=0.6500       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.0698    | 0.1956  |
| 0.10  | 0.0707    | 0.1985  |
| 0.20  | 0.0713    | 0.1996  |
| 0.30  | 0.0726    | 0.2056  |
| 0.40  | 0.0732    | 0.2092  |
| 0.50  | 0.0735    | 0.2147  |
| 0.60  | 0.0733    | 0.2121  |
| 0.70  | 0.0727    | 0.2085  |
| 0.80  | 0.0719    | 0.2032  |
| 0.90  | 0.0710    | 0.1973  |
| 1.00  | 0.0670    | 0.1882  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.6000 | 0.0770    | 0.2196  |
| 2    | 0.6500 | 0.0763    | 0.2176  |
| 3    | 0.7000 | 0.0763    | 0.2156  |

### 외과(TL+VL)

- Queries: 3384 | Pilot holdout: 3084
- Pilot hybrid: final alpha=0.1167, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.1000, 0.0500, 0.2000], holdout=3084, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.1322    | 0.3204  | BM25 top-10        |
| Dense only                | 0.1202    | 0.2979  | Dense top-10       |
| Alpha sweep (best recall) | 0.1383    | 0.3347  | alpha=0.2000       |
| Alpha sweep (best NDCG)   | 0.1368    | 0.3405  | alpha=0.5000       |
| RRF                       | 0.1369    | 0.3381  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.1385    | 0.3341  | alpha=0.1167       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.1360    | 0.3260  |
| 0.10  | 0.1380    | 0.3310  |
| 0.20  | 0.1383    | 0.3347  |
| 0.30  | 0.1381    | 0.3371  |
| 0.40  | 0.1374    | 0.3393  |
| 0.50  | 0.1368    | 0.3405  |
| 0.60  | 0.1357    | 0.3351  |
| 0.70  | 0.1347    | 0.3289  |
| 0.80  | 0.1329    | 0.3216  |
| 0.90  | 0.1310    | 0.3156  |
| 1.00  | 0.1202    | 0.2979  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.1000 | 0.1360    | 0.3125  |
| 2    | 0.0500 | 0.1350    | 0.3106  |
| 3    | 0.2000 | 0.1347    | 0.3149  |

### 의료법규(TL+VL)

- Queries: 114
- Pilot hybrid: skipped (query_count<600)

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.0316    | 0.0770  | BM25 top-10        |
| Dense only                | 0.0325    | 0.0820  | Dense top-10       |
| Alpha sweep (best recall) | 0.0368    | 0.0899  | alpha=0.2000       |
| Alpha sweep (best NDCG)   | 0.0351    | 0.0927  | alpha=0.6000       |
| RRF                       | 0.0342    | 0.0870  | 1/(1+rank), top-10 |
| Pilot hybrid              | SKIPPED   | SKIPPED | query_count<600    |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.0342    | 0.0808  |
| 0.10  | 0.0360    | 0.0884  |
| 0.20  | 0.0368    | 0.0899  |
| 0.30  | 0.0368    | 0.0893  |
| 0.40  | 0.0351    | 0.0844  |
| 0.50  | 0.0351    | 0.0907  |
| 0.60  | 0.0351    | 0.0927  |
| 0.70  | 0.0351    | 0.0918  |
| 0.80  | 0.0351    | 0.0919  |
| 0.90  | 0.0342    | 0.0904  |
| 1.00  | 0.0325    | 0.0820  |

### 이비인후과(TL+VL)

- Queries: 510
- Pilot hybrid: skipped (query_count<600)

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.1167    | 0.3021  | BM25 top-10        |
| Dense only                | 0.1104    | 0.2923  | Dense top-10       |
| Alpha sweep (best recall) | 0.1271    | 0.3351  | alpha=0.4000       |
| Alpha sweep (best NDCG)   | 0.1255    | 0.3380  | alpha=0.5000       |
| RRF                       | 0.1251    | 0.3364  | 1/(1+rank), top-10 |
| Pilot hybrid              | SKIPPED   | SKIPPED | query_count<600    |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.1218    | 0.3117  |
| 0.10  | 0.1245    | 0.3182  |
| 0.20  | 0.1253    | 0.3251  |
| 0.30  | 0.1271    | 0.3346  |
| 0.40  | 0.1271    | 0.3351  |
| 0.50  | 0.1255    | 0.3380  |
| 0.60  | 0.1253    | 0.3355  |
| 0.70  | 0.1229    | 0.3244  |
| 0.80  | 0.1210    | 0.3160  |
| 0.90  | 0.1192    | 0.3095  |
| 1.00  | 0.1104    | 0.2923  |

### 정신건강의학과(TL+VL)

- Queries: 1789 | Pilot holdout: 1489
- Pilot hybrid: final alpha=0.9333, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.8500, 0.9500, 1.0000], holdout=1489, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.1311    | 0.2554  | BM25 top-10        |
| Dense only                | 0.1550    | 0.3126  | Dense top-10       |
| Alpha sweep (best recall) | 0.1550    | 0.3126  | alpha=1.0000       |
| Alpha sweep (best NDCG)   | 0.1532    | 0.3132  | alpha=0.7000       |
| RRF                       | 0.1491    | 0.3073  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.1520    | 0.3087  | alpha=0.9333       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.1363    | 0.2636  |
| 0.10  | 0.1433    | 0.2751  |
| 0.20  | 0.1459    | 0.2824  |
| 0.30  | 0.1470    | 0.2891  |
| 0.40  | 0.1477    | 0.2950  |
| 0.50  | 0.1490    | 0.3060  |
| 0.60  | 0.1505    | 0.3109  |
| 0.70  | 0.1532    | 0.3132  |
| 0.80  | 0.1531    | 0.3126  |
| 0.90  | 0.1541    | 0.3128  |
| 1.00  | 0.1550    | 0.3126  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.8500 | 0.1613    | 0.3318  |
| 2    | 0.9500 | 0.1613    | 0.3314  |
| 3    | 1.0000 | 0.1613    | 0.3312  |

### 피부과(TL+VL)

- Queries: 682 | Pilot holdout: 382
- Pilot hybrid: final alpha=0.0500, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.1000, 0.0500, 0.0000], holdout=382, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.1194    | 0.2379  | BM25 top-10        |
| Dense only                | 0.1000    | 0.2080  | Dense top-10       |
| Alpha sweep (best recall) | 0.1232    | 0.2452  | alpha=0.1000       |
| Alpha sweep (best NDCG)   | 0.1218    | 0.2468  | alpha=0.3000       |
| RRF                       | 0.1194    | 0.2414  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.1223    | 0.2415  | alpha=0.0500       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.1227    | 0.2419  |
| 0.10  | 0.1232    | 0.2452  |
| 0.20  | 0.1216    | 0.2433  |
| 0.30  | 0.1218    | 0.2468  |
| 0.40  | 0.1211    | 0.2451  |
| 0.50  | 0.1217    | 0.2441  |
| 0.60  | 0.1179    | 0.2365  |
| 0.70  | 0.1160    | 0.2332  |
| 0.80  | 0.1151    | 0.2300  |
| 0.90  | 0.1135    | 0.2259  |
| 1.00  | 0.1000    | 0.2080  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.1000 | 0.1247    | 0.2494  |
| 2    | 0.0500 | 0.1233    | 0.2472  |
| 3    | 0.0000 | 0.1230    | 0.2444  |

## Group-Level Analysis

### AIHub TL+VL

- Queries: 13756 | Pilot holdout: 13456
- Pilot hybrid: final alpha=0.4500, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.4500, 0.5000, 0.4000], holdout=13456, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.1361    | 0.2989  | BM25 top-10        |
| Dense only                | 0.1367    | 0.2997  | Dense top-10       |
| Alpha sweep (best recall) | 0.1476    | 0.3294  | alpha=0.5000       |
| Alpha sweep (best NDCG)   | 0.1476    | 0.3294  | alpha=0.5000       |
| RRF                       | 0.1473    | 0.3280  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.1475    | 0.3277  | alpha=0.4500       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.1412    | 0.3056  |
| 0.10  | 0.1451    | 0.3129  |
| 0.20  | 0.1465    | 0.3179  |
| 0.30  | 0.1471    | 0.3223  |
| 0.40  | 0.1475    | 0.3260  |
| 0.50  | 0.1476    | 0.3294  |
| 0.60  | 0.1474    | 0.3267  |
| 0.70  | 0.1470    | 0.3230  |
| 0.80  | 0.1459    | 0.3182  |
| 0.90  | 0.1446    | 0.3132  |
| 1.00  | 0.1367    | 0.2997  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.4500 | 0.1433    | 0.3255  |
| 2    | 0.5000 | 0.1427    | 0.3226  |
| 3    | 0.4000 | 0.1420    | 0.3189  |

### KorMedMCQA_dentist

- Queries: 1412 | Pilot holdout: 1112
- Pilot hybrid: final alpha=0.2167, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.3000, 0.1500, 0.2000], holdout=1112, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.0445    | 0.1042  | BM25 top-10        |
| Dense only                | 0.0431    | 0.0978  | Dense top-10       |
| Alpha sweep (best recall) | 0.0492    | 0.1120  | alpha=0.6000       |
| Alpha sweep (best NDCG)   | 0.0489    | 0.1135  | alpha=0.4000       |
| RRF                       | 0.0489    | 0.1130  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.0500    | 0.1135  | alpha=0.2167       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.0465    | 0.1071  |
| 0.10  | 0.0474    | 0.1091  |
| 0.20  | 0.0491    | 0.1123  |
| 0.30  | 0.0488    | 0.1131  |
| 0.40  | 0.0489    | 0.1135  |
| 0.50  | 0.0487    | 0.1129  |
| 0.60  | 0.0492    | 0.1120  |
| 0.70  | 0.0489    | 0.1108  |
| 0.80  | 0.0488    | 0.1090  |
| 0.90  | 0.0476    | 0.1058  |
| 1.00  | 0.0431    | 0.0978  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.3000 | 0.0450    | 0.1091  |
| 2    | 0.1500 | 0.0450    | 0.1079  |
| 3    | 0.2000 | 0.0450    | 0.1078  |

### KorMedMCQA_doctor

- Queries: 2489 | Pilot holdout: 2189
- Pilot hybrid: final alpha=0.4667, sample_size=300, alpha_step=0.0500, top_k=3, selected=[0.5000, 0.3500, 0.5500], holdout=2189, objective=recall_at_10

### Method Summary

| Method                    | Recall@10 | NDCG@10 | Notes              |
| ------------------------- | --------- | ------- | ------------------ |
| Sparse only               | 0.1121    | 0.2190  | BM25 top-10        |
| Dense only                | 0.1189    | 0.2337  | Dense top-10       |
| Alpha sweep (best recall) | 0.1239    | 0.2483  | alpha=0.5000       |
| Alpha sweep (best NDCG)   | 0.1239    | 0.2483  | alpha=0.5000       |
| RRF                       | 0.1242    | 0.2483  | 1/(1+rank), top-10 |
| Pilot hybrid              | 0.1249    | 0.2449  | alpha=0.4667       |

### Alpha Sweep Curve

| Alpha | Recall@10 | NDCG@10 |
| ----- | --------- | ------- |
| 0.00  | 0.1165    | 0.2251  |
| 0.10  | 0.1217    | 0.2338  |
| 0.20  | 0.1228    | 0.2380  |
| 0.30  | 0.1224    | 0.2378  |
| 0.40  | 0.1229    | 0.2413  |
| 0.50  | 0.1239    | 0.2483  |
| 0.60  | 0.1231    | 0.2453  |
| 0.70  | 0.1230    | 0.2435  |
| 0.80  | 0.1237    | 0.2426  |
| 0.90  | 0.1236    | 0.2424  |
| 1.00  | 0.1189    | 0.2337  |

### Pilot Hybrid Details

| Rank | Alpha  | Recall@10 | NDCG@10 |
| ---- | ------ | --------- | ------- |
| 1    | 0.5000 | 0.1177    | 0.2424  |
| 2    | 0.3500 | 0.1177    | 0.2379  |
| 3    | 0.5500 | 0.1170    | 0.2432  |

## Summary

- Full alpha curves and per-dataset details are also included in the JSON companion file.
- File-level pooling is query-weighted; group-level results pool all queries belonging to the group.
