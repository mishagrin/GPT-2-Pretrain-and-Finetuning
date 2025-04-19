# GPT-2 for Russian Poetry Generation

This repository provides an implementation of training and fine-tuning GPT-2 for generating Russian poetry. It follows a custom text preprocessing approach that incorporates phonetics, syllables, and stresses to help the model better understand the language and its rules.

## Setup

Follow these steps to set up your environment and start working with the notebooks and scripts.

### Create and activate the Anaconda environment

```bash
conda create -n poetry-gen python=3.9
conda activate poetry-gen
```

### Install required libraries

Install the necessary dependencies for the project using the following commands

```bash
pip install jupyterlab ipywidgets nltk num2words regex pandas numpy
pip install git+https://github.com/Desklop/StressRNN
pip install git+https://github.com/Koziev/rusyllab
pip install transformers datasets tokenizers
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Download the Taiga Proza Russian Corpus
In the first notebook `01. Download Taiga Proza Russian Corpus.ipynb`, you will download the Taiga Proza Russian Corpus. Follow the steps outlined in the notebook to download and prepare the dataset.

## Text Preprocessing & Evaluation
After downloading the dataset, proceed with the text preprocessing in `02. Text Preprocessing & Evaluation.ipynb`. This step prepares the data for training, ensuring it adheres to the custom preprocessing rules that include phonetic and syllabic information.

## Pretrain GPT-2
To pretrain GPT-2 on the processed Russian text, use the script located in the `scripts/pretrain.py` folder. This script will initialize and train GPT-2 on the dataset. See script file for parameters.

## Fine-Tune GPT-2 on Russian Poetry
Once the pretraining is complete, you can fine-tune GPT-2 on Russian poetry in `03. Fine-tune GPT-2 on Russian Poetry.ipynb`. The fine-tuning step is important for generating poetry with rhythm and structure. Follow the steps provided in the respective notebook for this process.

## Reproduce Results
By following the notebooks and running the provided scripts, you should be able to reproduce the results of training GPT-2 to generate Russian poetry.

## Conclusion

This repository demonstrates the use of GPT-2 for generating Russian poetry by incorporating advanced text preprocessing techniques. The combination of these methods allows for more accurate and natural generation of poetry in the Russian language.

*Special thanks to Ilya Koziev for original idea.*

https://github.com/Koziev/verslibre