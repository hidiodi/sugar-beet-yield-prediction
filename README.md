# Sugar Beet Yield Prediction for Germany

This repository contains a pipeline for predicting sugar beet yields at the district level in Germany. The project utilizes a hybrid modeling approach, integrating agronomic crop simulations, meteorological data, and satellite imagery to achieve high-accuracy predictions.

## 🌟 Key Features

* **Hybrid Modeling Approach:** Combines process-based crop simulation models (e.g., WOFOST) with advanced machine learning techniques.
* **Multi-Source Data Integration:** Utilizes diverse data sources including:
  * Meteorological/Weather data
  * Agronomic soil and site data
  * Google Earth Engine (GEE) for large-scale winter satellite imagery processing.
* **Automated Data Pipelines:** Streamlined scripts for data downloading, preprocessing, and feature engineering.
* **Modular Architecture:** Clear separation of data processing, modeling, and analysis workflows.

## 📂 Project Structure

```text
.
├── data/               # Raw, intermediate, and processed datasets
├── docs/               # Project documentation
├── reports/            # Generated analysis reports and figures
├── src/                # Python source code
│   ├── 01_data/        # Data downloading and processing pipelines
│   ├── 02_models/      # Feature engineering, simulations (WOFOST), component models, and ensemble pipelines
│   └── 03_analysis/    # Visualization and comprehensive analysis tools
└── README.md           # Project overview and setup instructions
```

## 🚀 Getting Started

### Prerequisites

* Python 3.8+ (Recommended)
* Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/sugar-beet-yield-prediction.git
   cd sugar-beet-yield-prediction
   ```

2. **Set up a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   *(Note: A `requirements.txt` file is expected for a complete environment setup. You may need to create one with the necessary packages such as pandas, numpy, scikit-learn, etc.)*
   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

The project is structured into modular pipelines. You can execute these pipelines from the project root directory.

### 1. Data Downloading & Processing
To download and preprocess the required input data (weather, satellite, soil, etc.):
```bash
python src/01_data/download_all_data_pipeline.py
python src/01_data/process_input_data_pipeline.py
```

### 2. Model Pipeline Execution
The modeling pipeline includes simulation preparation, feature engineering, and ensemble modeling. You can control which steps run by editing the `SCRIPTS` list in `src/02_models/execute_hybrid_pipeline.py`.
```bash
python src/02_models/execute_hybrid_pipeline.py
```

### 3. Analysis & Visualization
Run the analysis scripts to generate reports, evaluate the components, and visualize model performance:
```bash
python src/03_analysis/comprehensive_pipeline_analysis.py
```

## 🛠️ Built With

* **Python:** Core programming language
* **Google Earth Engine (GEE):** Geospatial data extraction and pipeline processing
* **WOFOST:** Crop simulation model integration
* **Machine Learning:** Standard data science and ML stacks (scikit-learn, pandas)
