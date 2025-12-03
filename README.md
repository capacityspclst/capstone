# Graph Neural Networks for CMDB Intelligence

This repository contains all code for a capstone project focused on using Graph Neural Networks (GNNs) and XGBoost to analyze a synthetic CMDB-style graph. The project includes node classification, link prediction, and graph-level classification, along with a complete data generation and noise simulation pipeline.

## Repository Structure

```
Baseline Models/
│   Graph_Classification_XGB_Final.py
│   Link_Prediction_XGB_Final.py
│   Node_Classification_XGB_Final.py
│
Data Generation/
│   DataGeneration.py
│   EDA_Noise_Creation.py
│
GNN/
    Graph_Classification_GNN.py
    Link_Prediction_GNN.py
    Node_Classification_GNN.py
```

## File Descriptions

### Baseline Models (XGBoost & Gradient Boosting)

#### Node_Classification_XGB_Final.py
- Loads clean or noisy node CSVs and engineered metadata
- Builds features including degree stats, structural metrics, encoded node types, timestamps, and optional PageRank
- Trains an XGBoostClassifier with randomized hyperparameter tuning
- Handles class imbalance (class weights or resampling)
- Outputs accuracy, F1, AUC, calibration curves, threshold tuning, and feature importance

#### Link_Prediction_XGB_Final.py
- Generates positive/negative edge samples
- Computes classical link prediction heuristics: CN, Jaccard, Adamic-Adar, RA, degree-based features, shortest-path stats, preferential attachment
- Trains an XGBoost model for binary link prediction
- Outputs ROC/PR curves, permutation importance, calibration plots, and predictions

#### Graph_Classification_XGB_Final.py
- Builds one feature vector per LOB using aggregated node/edge statistics
- Creates binary labels based on the number of high-risk nodes
- Trains GradientBoostingClassifier and HistGradientBoostingClassifier
- Hyperparameter tuning via RandomizedSearchCV
- Supports outer CV or a standard train/val/test split
- Tunes the decision threshold
- Saves metrics, feature importances, and plots (single-split mode)

### Data Generation

#### DataGeneration.py
- Creates a synthetic CMDB-style graph with LOBs, applications, servers, DBs, queues, and networks
- Generates edges using rule-based and probabilistic logic
- Assigns low/medium/high labels and creates train/val/test splits
- Outputs node and edge CSVs plus metadata for modeling

#### EDA_Noise_Creation.py
- Injects realistic noise: missing edges, wrong node types, duplicate nodes, corrupted metadata, incorrect relationships
- Produces clean vs noisy summaries and exports noisy datasets

### GNN Models (PyTorch Geometric)

#### Node_Classification_GNN.py
- Loads graph data into PyTorch Geometric
- Trains a GraphSAGE-style GNN with dropout, batch norm, and residuals
- Supports EMA, DropEdge, consistency loss, and optional Node2Vec
- Saves accuracy, F1, ROC-AUC, and training curves

#### Link_Prediction_GNN.py
- Trains a GNN encoder to learn node embeddings
- Uses dot-product or MLP decoding for link prediction
- Includes negative sampling, evaluation, and embedding export

#### Graph_Classification_GNN.py
- Treats each LOB as a graph for graph-level classification
- Uses a GIN-based model with pooling to generate graph embeddings
- Outputs graph-level accuracy, F1, and confusion matrices

## Outputs

- Accuracy, precision, recall, F1, AUC
- ROC and PR curves
- Confusion matrices
- Feature importance plots
- GNN training curves and embeddings
- Link prediction score distributions
- Clean vs noisy comparison reports
- Saved model artifacts and predictions
