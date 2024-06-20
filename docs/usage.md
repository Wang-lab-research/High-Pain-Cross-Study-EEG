This file provides examples and instructions on how to use the various components of the project. It should include code snippets and explanations.

TODO: verify all below is accurate and updated

# Usage Instructions

## Data Preprocessing

### Cleaning Raw Data
Use the `clean_raw_data` function to clean the raw EEG data.

```python
from src.preprocessing.clean_data import clean_raw_data

input_path = 'data/raw/subject1.csv'
output_path = 'data/cleaned/subject1_cleaned.csv'
clean_raw_data(input_path, output_path)
```

### Standardizing Data
Use the standardize_data function to standardize the cleaned data.

```python
from src.preprocessing.standardize_data import standardize_data

input_path = 'data/cleaned/subject1_cleaned.csv'
output_path = 'data/standardized/subject1_standardized.csv'
standardize_data(input_path, output_path)
```

# Data Analysis
Performing Statistical Analysis
Use the perform_statistical_analysis function to analyze the standardized data.

```python
from src.analysis.statistical_analysis import perform_statistical_analysis

input_path = 'data/standardized/subject1_standardized.csv'
output_path = 'analysis/results/subject1_analysis.txt'
perform_statistical_analysis(input_path, output_path)
```

### Extracting Features
Use the extract_features function to extract features from the data.
```python
from src.analysis.feature_extraction import extract_features

input_path = 'data/standardized/subject1_standardized.csv'
output_path = 'analysis/features/subject1_features.csv'
extract_features(input_path, output_path)
```

# Data Visualization
Plotting Results
Use the plot_data function to generate plots from the data.
```python
from src.visualization.plot_results import plot_data

input_path = 'data/standardized/subject1_standardized.csv'
output_path = 'visualization/plots/subject1_plot.png'
plot_data(input_path, output_path)
```

# Generating Reports
Use the generate_report function to create reports from the analysis results.

```python
from src.visualization.generate_reports import generate_report

input_path = 'analysis/results/subject1_analysis.txt'
output_path = 'visualization/reports/subject1_report.pdf'
generate_report(input_path, output_path)
```