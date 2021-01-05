# Explain the Data, Not Just the Model

**Abstract**:  Machine learning technologies have achieved remarkable performance on visual tasks. Despite their power, these technologies have been shown to be fragile to data distribution shifts, where, state-of-the-art models could have their performance compromised. Indeed, there are numerous approaches for detecting the shifts; however, none of them offer explanations of what triggers them, thus limiting their utility. This project aims to develop robust explanation techniques that can detect and explain factors causing data distribution shifts by incorporating methods from explainable AI, namely concept-based explanations, to ensure the safe deployment of decision automation systems in high-stake domains.  

This repository contains source code of the system and experimentation results.

**Note**:
- Key accompanying papers include [Lipton, Wang, and Smola (2020)](https://arxiv.org/pdf/1802.03916.pdf), [Koh et al. (2020)](https://arxiv.org/pdf/2007.04612.pdf), [Rabanser, Gunnemann, and Lipton (2019)](https://arxiv.org/pdf/1810.11953.pdf), [Kazhdan et al. (2020)](https://arxiv.org/pdf/2010.13233.pdf).

## Requirements:
- Python 3.6+
- numpy
- pandas
- matplotlib
- jupyter

## Folder Structure:
- *data*: contains datasets used for experimentation.
- *notebooks*: contains experimentation prototype codes.
- *results*: contains experimentation results, model configurations, pickles.
- *scripts*: contains clean source code files.

## Setup:
```bash
git clone https://github.com/maleakhiw/explaining-dataset-shifts.git
cd scripts
pip install -r requirements.txt
```

## Authors:
- Maleakhi Wijaya
- Dmitry Kazhdan (Supervisor)
- Botty Dimanov (Supervisor)
- Mateja Jamnik (Supervisor)