# Legal Question Answering System Based on Knowledge Graph

**An intelligent legal consultation platform combining Knowledge Graphs (Neo4j) with Deep Learning (ELECTRA+BiLSTM+CRF).**

---

## Introduction

This project is a web-based **Legal Question Answering System** developed using the **Django** framework. It addresses the need for professional, accurate, and interpretable legal consultation by constructing a domain-specific **Knowledge Graph (KG)** for Criminal Law.

Unlike general-purpose search engines or LLMs, this system relies on structured legal data stored in **Neo4j** to provide precise answers regarding crimes, statutes, punishments, and judicial interpretations. It incorporates a hybrid NLP approach (Rule-based + Deep Learning) for intent understanding and entity recognition.

---

## Key Features

### 1. Intelligent Legal Q&A
* **Hybrid Parsing**: Utilizes a "Rule Matching + Deep Learning" strategy to analyze user questions.
* **Precise Retrieval**: Converts natural language into **Cypher** queries to retrieve accurate answers from the Neo4j Knowledge Graph.
* **Structured Answers**: Returns comprehensive information including crime definitions, sentencing standards, and legal basis.

### 2. Knowledge Graph Visualization
* **Interactive Graph**: Uses **ECharts** to visualize the legal knowledge graph.
* **Exploration**: Users can search for specific crimes (e.g., "Theft") and expand nodes to explore relationships between crimes, laws, and punishments.

### 3. Crime Prediction (Multi-Label Classification)
* **AI-Powered Analysis**: Users can input complex case descriptions.
* **Model**: Uses a fine-tuned **ELECTRA** pre-trained model (trained on the **CAIL2018** dataset) to predict potential charges and their probabilities.

### 4. Dynamic Process Demonstration
* **Explainable AI**: Visualizes the internal reasoning process using **Mermaid.js**.
* **Flow Visualization**: Shows the step-by-step pipeline: Entity Recognition -> Question Classification -> Graph Search -> Template Matching -> Answer Generation.

### 5. System Comparison
* **Side-by-Side View**: Compares the answers from this Knowledge Graph system against a general LLM (GPT-3.5 via API) to highlight the differences in professionalism and accuracy.

---

## System Architecture

The system follows a 4-layer architecture:
1.  **Presentation Layer**: Django templates rendering HTML/CSS, ECharts for graphs, and Mermaid.js for flowcharts.
2.  **Business Logic Layer**:
    * **NLP Module**: Entity Recognition & Intent Classification using **ELECTRA + BiLSTM + CRF**.
    * **Service Interface**: **FastAPI** wraps the AI models as independent microservices.
    * **QA Logic**: Template matching and answer organization.
3.  **Data Layer**:
    * **Neo4j**: Stores the Legal Knowledge Graph (Nodes & Relationships).
    * **SQLite**: Stores user account data.

---

## Methodology & Algorithms

### 1. Entity Recognition & Question Classification
We employ a **Multi-task Learning Model**:
* **Model Architecture**: `ELECTRA` (Embedding) + `BiLSTM` (Context) + `CRF` (Sequence Labeling).
* **Tasks**:
    1.  **NER (Named Entity Recognition)**: Identifies entities like `Crime`, `Law`, `Punishment`.
    2.  **Intent Classification**: Classifies questions into 7 types (e.g., Definition, Sentencing, Constitution).
* **Performance**: Achieved **79.57%** Accuracy in Entity Recognition and **79.54%** in Question Classification.

### 2. Knowledge Graph Construction
* **Data Source**: Criminal Law dataset (crawled from legal websites).
* **Construction Method**: Combined Top-down (Schema design) and Bottom-up (Information extraction).
* **Schema Design**:
    * **Entities**: `Crime`, `Law`, `Constituent`, `Punishment`, `Interpretation`.
    * **Relationships**: `Belongs_to`, `Based_on`, `Has_constituent`, `Has_punishment`, `Has_interpretation`.

### 3. Crime Prediction
* Fine-tuned the `hfl/chinese-legal-electra-small-discriminator` model.
* Implemented dynamic thresholding to handle multi-label classification for complex cases.

---

## Tech Stack

* **Web Framework**: Django, FastAPI
* **Database**: Neo4j (Graph DB), SQLite
* **NLP & AI**: PyTorch, Transformers (Hugging Face), Py2neo
* **Frontend Visualization**: ECharts, Mermaid.js, Bootstrap
* **Language**: Python 3.x

---

## How to Run

### Prerequisites
* Python 3.8+
* Neo4j Database (Community Edition)
* Anaconda (Recommended)

### Steps

1.  **Start Neo4j Database**
2.  **Start the NLP Model Service (FastAPI)**
3.  **Start the Web Server (Django)**
4.  **Access the System**
