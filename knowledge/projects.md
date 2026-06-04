---
topic: projects
language: en
---

# Featured projects

## Cloud ML Demand Forecasting Pipeline
**Repo:** [aws-ml-production-pipeline](https://github.com/KevDP/aws-ml-production-pipeline)

End-to-end retail demand forecasting system on AWS. Designed the full ML
lifecycle from data ingestion to real-time inference.

- **Stack:** Python, AWS SageMaker (XGBoost), Lambda, API Gateway, S3, boto3,
  scikit-learn, Pandas, NumPy
- **Architecture:** S3 layered storage → SageMaker training job → model
  artifact → Lambda inference → API Gateway REST endpoint (`POST /predict`)
- **Metrics:** R² = 0.84 · MAPE = 9.1% · RMSE = 18.3 · MAE = 11.4
- Includes architecture diagram and a documented quickstart.

## AI Knowledge Assistant (EVA)
**Repo:** [ai-knowledge-assistant](https://github.com/KevDP/ai-knowledge-assistant)
· **Live demo:** EVA widget on [kev-blog](https://github.com/KevDP/kev-blog)

This very assistant. RAG-powered chatbot that answers questions about
Kevin. Built in two phases:

- **Phase 0 (local):** Python + `sentence-transformers` embeddings + JSON
  vector store + Claude API. Zero infrastructure, pure code, used to
  validate retrieval quality before paying for cloud.
- **Phase 1 (AWS):** API Gateway → Lambda → Amazon Bedrock (Claude Haiku)
  → DynamoDB. Terraform-managed. Live behind kev-blog.

## Other repos (portfolio)
- **PELUSAS** — ROS autonomous navigation with Bug0 + ArUco markers + LiDAR,
  simulated in Gazebo.
- **Img-preprocessing-techniques** — OpenCV image filter pipeline with a
  tkinter GUI (histogram equalization + Gaussian convolution).
- **BusinessCase_Statistics** — statistical analysis and hypothesis testing
  notebook.
- **Actividades-Integrales / Formativas** — C++ implementations of sorting
  algorithms, linked lists, BST/Heap/Splay trees, and graph algorithms
  (BFS/DFS).
- **Proyecto_POO** — C++ OOP banking card simulation (Visa / Mastercard
  hierarchy with CVV + PIN validation).
