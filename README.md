# Defect Detection Project

This repository contains the LaTeX code and resources for a project exploring defect detection methods in misaligned and non-identical images.

## Project Overview
The project investigates traditional and Fourier-based methods to detect defects in images. It compares a defect-free reference image to a test image containing potential defects under challenging conditions, such as misalignment and lighting differences.

## Repository Structure
├── main.tex              # Main LaTeX file
├── references.bib        # Bibliography file
├── Figures/              # Folder containing all figures
│   ├── defected_mask.png
│   ├── defected_spectrum.png
│   ├── defected_partition.png


## How to Compile
1. Make sure you have LaTeX installed on your system (e.g., TeX Live, MikTeX, Overleaf).
2. Use the `main.tex` file to compile the document. It includes:
   - **BibLaTeX** for references.
   - Figures from the `Figures` directory.
3. Run the following sequence if using a local LaTeX compiler:
pdflatex main.tex
biber main
pdflatex main.tex
pdflatex main.tex

## Dependencies
- **BibLaTeX**: Ensure `biblatex` with `backend=bibtex` is installed.
- **Figures**: Ensure the `Figures` folder is in the same directory as `main.tex`.

## Credits
This project was created by [Your Name].

## To Do
- Improve the defect detection methodology using Fourier transforms.
- Optimize LaTeX formatting for enhanced readability.