# PEAK (Pattern rEcognition frAmewoRk)

![python](https://img.shields.io/badge/Python%20tested-3.9.x-blue)
![version](https://img.shields.io/badge/version-v1.0-blue)
![coverage](https://img.shields.io/badge/Coverage-100%25-orange)
![requirements](https://img.shields.io/badge/requirements-up%20to%20date-brightgreen)
![last_update](https://img.shields.io/badge/last%20update-April%2002%2C%202021-yellowgreen)
![license](https://img.shields.io/badge/License-PEAK%20by%20Giuseppe%20Sgroi%20is%20licensed%20under%20CC%20BY--NC--SA%204.0-red)

PEAK is a Python tool designed to make easier the basic steps of pattern recognition, data collec-tion, data exploration, data correlation, regression analysis, and/or classification. Therefore, PEAK allows less experienced users to reduce the time required for analysing data and promote the dis-covery of unknown relationships between different data.

![PEAK-functional-scheme](https://user-images.githubusercontent.com/15036433/113410100-ee83aa00-93b2-11eb-8f0e-2ebbef8eb6db.png)

## Installation

### Download zip
```bash
wget https://github.com/Pex2892/PEAK/archive/main.zip
unzip PEAK-main.zip
```
or
### Clone repository
```bash
git clone https://github.com/Pex2892/PEAK.git
```

---

## Setting Environment
A typical user can install the libraries using the following command:
``` bash
python3 -m pip install -r requirements.txt
```

---

# Run

Before starting the analysis, PEAK requires to configure "__settings.cfg__". In this file, there are four sections.
- __Settings:__ this section contains the general settings of the tool, such as the seed, the number of CPUs, and the clearing of the previous results.
- __Dataset:__ this section contains parameters related to the dataset, such as file name, separator, and the number of rows to skip.
- __Regression:__ it contains all the settings needed to start a regression analysis, such as the dependent variable name and the parameters useful for the resampling phase.
- __Classification:__ it contains all the settings required for classification, such as the variable to be classified, the targets and the parameters needed for the resampling phase.

You can test that you have correctly installed the PEAK 
by running the following command:
```bash
python3 main.py
```
