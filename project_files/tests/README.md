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

## Testing Guidelines

When implementing tests:

1. Use unittest assertions for validation
2. Test edge cases and typical cases
3. Use small, synthetic examples when possible
4. Include docstrings explaining what each test verifies
5. For data-dependent tests, create minimal test fixtures

## Test-Driven Development

Consider following test-driven development (TDD) for new features:

1. Write tests for the desired functionality
2. Run tests to confirm they fail
3. Implement the functionality
4. Run tests again to validate implementation
5. Refactor code as needed, ensuring tests continue to pass

## Expected Test Coverage

The project aims for high test coverage of critical components:

- Data processing functions: 90%+ coverage
- Model architecture functions: 80%+ coverage
- Training and evaluation functions: 70%+ coverage
