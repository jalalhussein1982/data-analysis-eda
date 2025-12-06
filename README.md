# Data Preparation Pipeline

A comprehensive, GDPR-compliant data preparation tool for cleaning, validating, and transforming datasets before correlation analysis.

## Features

- **Multi-format ingestion:** CSV, Excel, Parquet, JSON
- **Schema enforcement:** Type casting with user validation
- **Duplicate detection:** Exact, subset, and near-duplicate identification
- **Data sanitation:** Constraint enforcement and missing value imputation
- **Outlier handling:** IQR, Z-score, and winsorization methods
- **Multicollinearity screening:** VIF and correlation matrix analysis
- **Feature engineering:** Encoding and scaling transformations
- **Distribution inspector:** On-demand visualization with normality testing
- **Branching rollback:** Non-destructive state management with comparison

## Pipeline Phases

1. **Phase I: Ingestion & Schema Enforcement** - Load data and validate types
2. **Phase I-A: Duplicate Detection** - Identify and resolve duplicates
3. **Phase II: Scope Definition** - Select relevant columns
4. **Phase III: Data Sanitation** - Enforce constraints and impute missing values
5. **Phase IV: Outlier Handling** - Detect and resolve extreme values
6. **Phase IV-A: Multicollinearity Screening** - Identify redundant features
7. **Phase V: Feature Engineering** - Encode categoricals and scale numericals

## Installation

### Prerequisites

- Python 3.8+
- pip

### Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/data-analysis-eda.git
cd data-analysis-eda

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Fork/clone this repository to your GitHub account
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click "New app"
4. Select your repository and `app.py` as the entry point
5. Deploy

## Project Structure

```
data-analysis-eda/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # This file
├── PRIVACY.md                  # GDPR privacy policy
├── .streamlit/
│   └── config.toml             # Streamlit configuration
├── phases/
│   ├── __init__.py
│   ├── phase_1_ingestion.py    # File upload, schema enforcement
│   ├── phase_1a_duplicates.py  # Duplicate detection
│   ├── phase_2_scope.py        # Column selection
│   ├── phase_3_sanitation.py   # Constraints & imputation
│   ├── phase_4_outliers.py     # Outlier handling
│   ├── phase_4a_multicollinearity.py
│   └── phase_5_feature_eng.py  # Encoding & scaling
├── utilities/
│   ├── __init__.py
│   ├── state_manager.py        # StateManager class
│   ├── distribution_inspector.py
│   ├── normality_tests.py
│   ├── export_handler.py
│   └── config.py               # PipelineConfig dataclasses
└── components/
    ├── __init__.py
    ├── constraint_builder.py
    ├── imputation_selector.py
    └── rollback_interface.py
```

## Privacy & GDPR

This application processes all data in-memory only. No data is stored, logged, or transmitted to third parties. See [PRIVACY.md](PRIVACY.md) for full details.

Key privacy features:
- In-memory only processing
- Session-based storage (deleted on tab close)
- No analytics or tracking
- User-controlled data deletion

## Usage

1. **Upload your data file** (CSV, Excel, Parquet, or JSON)
2. **Review and confirm the schema** - adjust data types as needed
3. **Handle duplicates** - remove exact or subset duplicates
4. **Select columns** - keep only relevant features
5. **Sanitize data** - enforce constraints and impute missing values
6. **Handle outliers** - detect and resolve extreme values
7. **Screen for multicollinearity** - remove redundant features
8. **Engineer features** - encode categoricals and scale numericals
9. **Export** - download your prepared dataset

## Configuration Export

The pipeline configuration can be exported as JSON for reproducibility. This includes all decisions made during each phase.

## Author

**Jalal Hussein**
- Email: jalalhussein@gmail.com
- GitHub: [https://github.com/jalalhussein1982/](https://github.com/jalalhussein1982/)

## License

MIT License
