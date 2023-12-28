 # Wav2vec_swahili_model
Fine_tuning Facebook's XLSR wav2vec  transcriber model for the Swahili language
# Speech-to-Text using XLSR-Wav2Vec2

This repository contains code for training, evaluating, and performing inference with a Speech-to-Text (STT) model using the XLSR-Wav2Vec2 architecture. The model is trained on the Common Voice Swahili dataset.

## Getting Started

### Prerequisites

- Python 3.6 or later
- [Google Colab](https://colab.research.google.com/) (for notebook execution)
- [Transformers](https://huggingface.co/transformers/installation.html)
- [Datasets](https://huggingface.co/docs/datasets/installation.html)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Accelerate](https://huggingface.co/docs/accelerate/)
- [torchaudio](https://pytorch.org/audio/stable/index.html)
- [librosa](https://librosa.org/doc/main/install.html)
- [jiwer](https://pypi.org/project/jiwer/)

## Installing Dependencies


pip install transformers datasets torch jiwer accelerate sentencepiece torchaudio librosa

# Dataset Preparation
The Common Voice Swahili dataset is used for training and evaluation. The dataset is loaded using the Hugging Face datasets library.

# Model Architecture
The XLSR-Wav2Vec2 model is used for the STT task. The model is pretrained and fine-tuned on the Swahili Common Voice dataset.

# Training

The training process involves loading and preprocessing the audio data, augmenting the audio data for improved model generalization, tokenizing the text using a custom vocabulary, feature extraction using the XLSR-Wav2Vec2 processor, and training the model with gradient checkpointing enabled. Training parameters, such as batch size, learning rate, and evaluation strategy, can be customized in the TrainingArguments.

# Evaluation

The model is evaluated on a separate validation set using the Word Error Rate (WER) metric. The trained model is then saved for future use.

# Inference
Inference is performed on a test set, and the transcriptions are saved to a CSV file.

# Results
The trained model achieves competitive performance on the validation set, as measured by the WER metric.

# Contributors
Tony Ndung'u Munene
Wayne Otwori Maranga




