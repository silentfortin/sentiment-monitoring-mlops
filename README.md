# Sentiment Monitoring MLOps – MachineInnovators Inc.
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1-iHDJ34Qu6vTh_Aaxu7inzflA-LOYYNC?usp=sharing)


> Developed as part of **AI Engineering Master – Week 10**
> This project builds a full **MLOps pipeline** for **sentiment analysis**, integrating **Continuous Integration, Continuous Training, and Conditional Deployment** on Hugging Face.

---

## 📌 Project Overview

MachineInnovators Inc. aims to continuously monitor and improve its **online reputation** by automating the analysis of user sentiment on social media.
This system integrates **Hugging Face Transformers**, **GitHub Actions**, and **MLOps best practices** to enable:

* Continuous retraining with new data
* Automated model comparison and promotion
* End-to-end deployment on **Hugging Face Hub**
* Continuous monitoring of sentiment trends

The pipeline ensures that only **models with improved performance** are deployed, maintaining reliability and performance over time.

---

## 🧱 Technologies and Concepts

* ✅ **Python 3.10**
* ✅ **Hugging Face Transformers & Datasets**
* ✅ **PyTorch** for fine-tuning sentiment models
* ✅ **scikit-learn** for metrics computation
* ✅ **GitHub Actions** for CI/CD automation
* ✅ **Hugging Face Hub** for model versioning and deployment
* ✅ **Pandas & Matplotlib** for monitoring and visualization
* ✅ **Evaluate** for performance metrics
* ✅ **JSON-based metrics tracking** for model governance

---

## 🚀 Features

* 🔹 End-to-end **MLOps pipeline** with automatic testing, training, and deployment
* 🔹 **Conditional model promotion:** deploys only if new accuracy > current model
* 🔹 **Integration tests** to ensure data consistency and correct pipeline behavior
* 🔹 **Automatic artifact versioning** and model tagging
* 🔹 **Hugging Face Hub integration** for seamless model hosting
* 🔹 **Monitoring simulation** that logs sentiment predictions over time

---

## 📂 Project Structure

```
sentiment-monitoring-mlops/
├── .github/workflows/ci_cd.yml       # CI/CD pipeline definition
├── data/
│   ├── raw/                          # Datasets (training & retraining)
│   ├── processed/                    # Folder to store reprocessed data (if needed(
│   └── monitoring/                   # Monitoring logs and sentiment trend visualization
├── models/
│   ├── base_model_metrics.json       # Baseline accuracy reference
│   ├── roberta_sentiment_model/      # Locally trained model
│   └── production_model/             # Promoted model for deployment
├── src/
│   ├── preprocess.py                 # Data cleaning and preprocessing
│   ├── train.py                      # Model training and fine-tuning
│   ├── model_compare.py              # Performance comparison and model promotion
│   ├── monitor.py                    # Performance monitoring
│   └── monitor_utils.py              # Helper utilities
├── tests/
│   └── integration_test.py           # End-to-end integration tests
├── notebooks/
│   └── sentiment_model.ipynb         # Model evaluation and monitoring simulation
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation
```

---

## ⚙️ CI/CD Pipeline Overview

| Step | Description                                              |
| ---- | -------------------------------------------------------- |
| 1    | Load dataset and run **sanity tests** (data + tokenizer) |
| 2    | **Train** model with Hugging Face `Trainer`              |
| 3    | Run **integration tests** to validate model output       |
| 4    | **Compare** model accuracy vs production model           |
| 5    | If improved → **Deploy automatically** to Hugging Face   |
| 6    | Save **artifacts** and tag model version                 |
| 7    | (Optional) Log metrics for long-term monitoring          |

---

## 📊 Example Results

| Model Version          | Accuracy | Status                            |
| ---------------------- | -------- | --------------------------------- |
| `Base (HF pretrained)` | 0.923    | Reference                         |
| `Retrained v20251104`  | 0.918    | ❌ Lower accuracy → not deployed   |
| `Retrained v20251101`  | 0.937    | ✅ Promoted and deployed to HF Hub |

---

## 📈 Continuous Monitoring Simulation

The notebook `sentiment_model.ipynb` simulates continuous monitoring by:

1. Sampling new social media posts
2. Running predictions via the deployed model
3. Logging outputs to `data/monitoring/sentiment_log.csv`
4. Generating trend visualizations (`sentiment_trend.png`)

This enables early detection of **concept drift** and performance degradation over time.

---

## 🧩 Future Extensions

| Component                 | Description                                     |
| ------------------------- | ----------------------------------------------- |
| **MLflow Tracking**       | Log and visualize model metrics across versions |
| **Prometheus + Grafana**  | Real-time monitoring of prediction service      |
| **FastAPI Inference API** | Serve predictions via REST endpoint             |
| **Dockerization**         | Containerize for scalable deployment            |
| **Scheduled retraining**  | Automate model updates via cron/Airflow         |

---

## 📎 License & Credits

This project is part of the **AI Engineering Master Portfolio**.
All models are based on the Hugging Face architecture:
[`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)

---

## 🔗 Useful Links

* **Model on Hugging Face:** [CapDimble/sentiment-monitoring-model](https://huggingface.co/CapDimble/sentiment-monitoring-model)
* **GitHub Repository:** [silentfortin/sentiment-monitoring-mlops](https://github.com/silentfortin/sentiment-monitoring-mlops)
