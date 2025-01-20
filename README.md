# Fourier-Based Defect Detection Pipeline

This repository contains a comprehensive Python-based pipeline for defect detection using Fourier Transform techniques. The project is designed for analyzing images of materials or patterns to identify potential defects by comparing a reference image and an inspected image.

---

## Features

- **Preprocessing**: Align and normalize images for accurate defect analysis.
- **Fourier Transform**: Analyze and manipulate images in the frequency domain.
- **Defect Detection**: Identify defects based on spectrum subtraction and adaptive thresholding.
- **Partitioning**: Divide images into subregions for granular analysis.
- **Visualization**: Display intermediate and final results at each stage of the pipeline.

---

## Prerequisites

Ensure you have the following installed:

- Python 3.8+
- Required Python libraries (install using `requirements.txt`)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/friedsun314/Defect-Detection
   cd defect-detection-pipeline

2.	Install dependencies:
   pip install -r requirements.txt

## Usage

1.	Place your reference and inspected images in the data/defective_examples/ directory.
2.	Update the paths in main.py to point to your images.
3.	Run the pipeline: python main.py
4.	View the output in the output/ directory.

## Directory Structure
data/
├── defective_examples/
├── non_defective_examples/
main/
├── __pycache__/
├── __init__.py
├── fft.py
├── filtered_fft_inspected.py
├── ift.py
├── main.py
├── partitioning.py
├── postprocessing.py
├── preprocessing.py
├── tests.py
├── threshold_masking.py
├── visualization.py
old/
output/
README.md
requirements.txt

## Contributing
Contributions are welcome! If you have ideas for improving the pipeline or adding new features, feel free to submit a pull request.

## License
This project is licensed under the MIT License.