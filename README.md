# User Viewing Preference Analysis

An unsupervised machine learning and collaborative filtering project focused on mapping latent viewer co-watch dynamics from MovieLens interaction history. By pivoting multi-user movie ratings, applying active threshold filters, and binarizing user-item matrix rows, this pipeline leverages the tree-structured FP-Growth algorithm to uncover frequent movie affinity bundles with optimal memory efficiency.

## Dataset Description
The analysis operates on a collaborative watch-log format split across two distinct relational data files:
* `movies.csv`: Links individual `movieId` keys to descriptive movie `title` entries (including release year tags) and genre string arrays (e.g., *Adventure|Animation|Children|Comedy|Fantasy*).
* `ratings.csv`: Tracks continuous historical viewer evaluation streams mapping `userId`, `movieId`, explicit `rating` points (scaled from 0.5 to 5.0), and core `timestamp` references.

## Methodology
1. **Relational Data Merging:** Combined categorical movie metadata files with interaction rating streams using shared relational database keys (`movieId`).
2. **Dense Matrix Filters:** Applied density filters to exclude sparse outliers:
   * Retained active users who evaluated more than 50 movies (`user_counts > 50`).
   * Retained high-visibility movie titles reviewed by more than 50 separate users (`movie_counts > 50`).
3. **Tabular Pivot Re-Mapping:** Constructed an optimized user-item matrix re-mapping `userId` tracks as row indices against global `title` columns, filling missing evaluations with null zeros.
4. **Preference Binarization:** Transformed the continuous rating grid into strict logical flags, treating any score $\ge 3.0$ as a confirmed positive viewer preference (`True`) and lower ranks as a lack of engagement (`False`).
5. **Frequent Pattern Mining:** Passed the sparse boolean dataframe straight into the optimized `fpgrowth` pipeline, enforcing a pattern support threshold constraint of `min_support=0.02` to extract high-affinity watch sets.

## Results & Engine Capabilities
By deploying the tree-structured FP-Growth algorithm over traditional iterative market-basket approaches (such as standard Apriori), the pipeline compresses transaction lists directly into an implicit hierarchical FP-Tree structure. This removes the need for expensive candidate generation cycles and allows media platforms to scale background recommendation processing smoothly over massive catalog interaction histories.

## How to Run Locally
1. Clone this repository to your local computer directory.
2. Install standard scientific dependencies and the pattern mining framework via pip:
   ```bash
   pip install numpy pandas mlxtend jupyter
