# DSCI 553: Foundations and Applications of Data Mining
**USC Viterbi School of Engineering** · Spring 2025
 
Coursework for DSCI 553, covering large-scale data mining algorithms implemented in **PySpark** and **Scala**. Built on real Yelp and Ta Feng grocery datasets processed with Apache Spark 3.1.2.
 
---
 
## Repository Structure
 
```
data-mining/
├── lsh/                   # Locality Sensitive Hashing (MinHash + Jaccard similarity)
├── son-algorithm/         # Frequent Itemset Mining (SON + PCY / A-priori)
├── recommender-system/    # Collaborative filtering + XGBoost recommendation
├── girvan-newman/         # Community detection in social graphs
├── bloom-filtering/       # Bloom filter & Flajolet-Martin (data streams)
├── bfr/                   # BFR clustering algorithm
└── spark-operations/      # Core Spark utilities and Scala implementations
```
 
---
 
## Assignments
 
### LSH — Locality Sensitive Hashing
**Folder:** `lsh/`
 
Finds similar businesses in the Yelp dataset using MinHash signatures and LSH banding, approximating Jaccard similarity without brute-force pairwise comparison.
 
- MinHash with 60 hash functions
- Band size of 2 for candidate pair generation
- Jaccard similarity threshold: 0.5
- Implemented in both **PySpark** and **Scala**
**Stack:** PySpark, Scala, Spark 3.1.2
 
---
 
### SON Algorithm — Frequent Itemset Mining
**Folder:** `son-algorithm/`
 
SON algorithm with PCY and A-priori candidate generation across two datasets:
 
- **Yelp data:** Two basket configurations — users as baskets of businesses (Case 1), or businesses as baskets of users (Case 2)
- **Ta Feng grocery data:** Retail transaction preprocessing pipeline generating `DATE-CUSTOMER_ID → PRODUCT_ID` format before mining
**Stack:** PySpark, itertools, collections
 
---
 
### Recommender System
**Folder:** `recommender-system/`
 
Three approaches to predicting user ratings for Yelp businesses, plus a competition entry:
 
| Approach | Method | Notes |
|---|---|---|
| Item-based CF | Pearson similarity | Top-15 neighbors, avg rating fallbacks |
| Model-based | XGBoost regression | User, business, and review features |
| Hybrid | Weighted combination | 10% CF + 90% XGBoost |
| **Competition** | Advanced XGBoost | RMSE: **0.9799**, ~398s runtime |
 
**Competition feature engineering:**
- User: review count, avg stars, fans, years on Yelp, elite status, compliment totals, tip counts
- Business: stars, review count, price range, reservations, takeout availability
- Engagement: check-in frequency, photo presence, review sentiment signals (useful/funny/cool)
**Stack:** PySpark, XGBoost, NumPy
 
---
 
### Girvan-Newman — Community Detection
**Folder:** `girvan-newman/`
 
Detects communities in social networks using the Girvan-Newman algorithm, which iteratively removes edges with the highest betweenness centrality to reveal community structure.
 
**Stack:** PySpark, graph traversal (BFS)
 
---
 
### Bloom Filtering — Data Streams
**Folder:** `bloom-filtering/`
 
Streaming data mining algorithms for memory-efficient probabilistic analysis:
 
- **Bloom Filter:** Space-efficient membership testing with controlled false positive rate
- **Flajolet-Martin:** Approximate distinct element counting in a data stream
**Stack:** Python, PySpark, bitarray
 
---
 
### BFR Clustering
**Folder:** `bfr/`
 
Implementation of the Bradley-Fayyad-Reina (BFR) algorithm for clustering large datasets that don't fit in memory, using sufficient statistics to summarize cluster shape.
 
**Stack:** Python, NumPy
 
---
 
## Tech Stack
 
| Category | Tools |
|---|---|
| Distributed Computing | Apache Spark 3.1.2 (PySpark + Scala) |
| ML / Modeling | XGBoost, NumPy, scikit-learn |
| Languages | Python 3, Scala |
| Datasets | Yelp Academic Dataset, Ta Feng Grocery Dataset |
 
---
 
## Running the Code
 
All PySpark scripts are designed for `spark-submit`. Example commands are included as comments at the bottom of each file. General pattern:
 
```bash
/opt/spark/spark-3.1.2-bin-hadoop3.2/bin/spark-submit \
  <script>.py <input_folder> <test_file> <output_file>
```
 
---
 
## Course Info
 
- **Course:** DSCI 553 — Foundations and Applications of Data Mining
- **Topics covered:** MapReduce · Frequent Itemsets · LSH/MinHash · Collaborative Filtering · Graph Analysis · Data Streams · Clustering · Link Analysis · Web Advertising
 
