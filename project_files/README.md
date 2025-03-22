```bash
tf_binding_prediction/
├── README.md                  # Project documentation
├── requirements.txt           # Project dependencies
├── data/
│   ├── raw/                   # Original data from JASPAR
│   └── processed/             # Processed data (one-hot encoded sequences)
├── notebooks/                 # Analysis notebooks
│   ├── exploratory.ipynb      # Data exploration
│   └── results.ipynb          # Results visualization
├── src/                       # Source code (simplified from package structure)
│   ├── data.py                # Data loading and preprocessing
│   ├── model.py               # CNN and baseline model implementations
│   ├── train.py               # Training functions
│   └── evaluate.py            # Evaluation metrics and functions
├── scripts/
│   ├── download_data.sh       # Script to download data
│   ├── train_model.py         # Script to train the model
│   └── evaluate_model.py      # Script to evaluate the model
├── tests/                     # Basic tests for critical functionality
│   └── test_data.py           # Test data processing functions
└── config.yaml                # Single configuration file
