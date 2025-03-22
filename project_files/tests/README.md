# Tests Directory

This directory contains tests 

## Files

```
tests/
├── README.md        # This file
└── test_data.py     # Tests for data processing functions
```

## Test Module Descriptions

### test_data.py

Unit tests for functions in `src/data.py`.

**Tests:**
- `test_one_hot_encode`: Tests DNA sequence one-hot encoding
- `test_reverse_complement`: Tests DNA reverse complement function
- `test_parse_jaspar_pfm`: Tests parsing of JASPAR PFM files
- `test_pfm_to_pwm`: Tests conversion of PFM to PWM
- `test_generate_negative_samples`: Tests negative sample generation methods

**Usage:**
```bash
# Run all tests
python -m unittest tests/test_data.py

# Run a specific test
python -m unittest tests.test_data.TestDataProcessing.test_one_hot_encode
```

## Adding Tests

When extending functionality, add corresponding tests:

1. For new data processing functions, add tests to `test_data.py`
2. For model functions, create a new `test_model.py` file
3. For training functions, create a new `test_train.py` file
4. For evaluation functions, create a new `test_evaluate.py` file
