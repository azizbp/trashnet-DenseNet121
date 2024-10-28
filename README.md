# TrashNet Classification using DenseNet121

This repository contains a deep learning model for classifying different types of trash using DenseNet121 architecture. The model is trained on the TrashNet dataset and can classify images into six categories: cardboard, glass, metal, paper, plastic, and trash.

## 📋 Project Structure

```
├── .github
│   └── workflows
│       └── build.yaml            # GitHub Actions workflow configuration
├── .gitattributes                # Git attributes file
├── .gitignore                    # Git ignore file
├── bestModel-trashnet_v9-densenet121.h5    # Trained model weights
├── push_to_hub.py                # Script to push model to Hugging Face Hub
├── requirements.txt              # Python dependencies
├── train_model.py                # Main training script
└── trashnet_model.ipynb          # Jupyter notebook with model development
```

## 🚀 Features

- Image classification into 6 waste categories
- Built with DenseNet121 architecture
- Data augmentation for better generalization
- Wandb integration for experiment tracking
- GitHub Actions for automated training
- Model deployment to Hugging Face Hub

## 🛠️ Technologies Used

- TensorFlow 2.x
- Python 3.10
- Wandb
- GitHub Actions
- Hugging Face Hub

## 📊 Model Architecture

- Base Model: DenseNet121
- Input Shape: 224x224x3
- Output Classes: 6
- Training Features:
  - Data Augmentation
  - Transfer Learning
  - Class Weight Balancing
  - Learning Rate Reduction
  - Early Stopping

## 🔧 Setup and Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/trashnet-DenseNet121.git
cd trashnet-DenseNet121
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. Training the model:
```bash
python train_model.py
```

## 🚂 Training

The model training includes:
- Image augmentation (rotation, flipping, brightness adjustment)
- Validation split: 80% training, 20% validation
- Batch size: 32
- Learning rate: 0.001
- Early stopping and learning rate reduction
- Class weight balancing for handling imbalanced data

## 📈 Model Performance

The model achieves:
- Training Accuracy: 0.9502
- Validation Accuracy: 0.8688
- Training Loss : 0.3500
- Validation Loss : 0.6191

## 🤗 Hugging Face Hub Integration

The trained model is automatically pushed to Hugging Face Hub using the `push_to_hub.py` script. You can find the model at:
[[Hugging Face](https://huggingface.co/azizbp/trashnet-densenet121)]

## 🔄 GitHub Actions Workflow

The repository includes automated training pipeline using GitHub Actions:
- Triggers on push to main branch
- Sets up Python environment
- Installs dependencies
- Runs training script
- Logs metrics to Weights & Biases
- Pushes model to Hugging Face Hub

## 📊 Weights & Biases Integration

Training metrics and experiments are tracked using Weights & Biases. You can view the project at:
[[Wandb.ai](https://wandb.ai/azizbp-gunadarma-university/trashnet-model/table?nw=nwuserazizbp)]

## 🙏 Acknowledgements

- [TrashNet Dataset](https://github.com/garythung/trashnet)
- [DenseNet Paper](https://arxiv.org/abs/1608.06993)
- Wandb for experiment tracking
- Hugging Face for model hosting

## 📜 Citation

If you use this model in your research, please cite:

```bibtex
@misc{your-model-name,
  author = {Your Name},
  title = {TrashNet Classification using DenseNet121},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/trashnet-DenseNet121}}
}
```
