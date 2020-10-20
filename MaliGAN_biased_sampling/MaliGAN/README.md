## Requirement
* **TensorFlow >= 1.5.0**
* Numpy 1.12.1
* Scipy 0.19.0
* NLTK 3.2.3
* CUDA 7.5+ (Suggested for GPU speed up, not compulsory)    
* Biopython
* Modlamp

Or just type `pip install -r requirements.txt` in your terminal.

## Implemented Models and Original Papers
* **MaliGAN** - [Maximum-Likelihood Augmented Discrete Generative Adversarial Networks](https://arxiv.org/abs/1702.07983)

## Get Started
```bash
# run MaliGAN with setting tuned for Protein Data
python3 main.py
```
Settings:
* Data:[here](data/real_sequences_amp.txt)<br>
* Encoded Data:[here](save/oracle.txt)<br>
* detailed documentation for the platform and code setup is provided [here](docs/doc.md).

## Evaluation Results
Documentation  [here](Visualization&Validation/EvaluationCriteria.md).

