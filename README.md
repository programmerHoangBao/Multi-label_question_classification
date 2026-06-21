# Multi-Label Question Classification

This project implements a **BERT-Bi-LSTM-Attention (BBLA) Multi-Label Classification Model** for classifying questions into multiple categories simultaneously using advanced deep learning techniques.

---

## Table of Contents

- [Overview](#overview)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Requirements](#requirements)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project tackles the **multi-label classification problem** where each input question can be assigned multiple labels (tags) simultaneously. Unlike single-label classification, multi-label allows for more flexible and realistic categorization.

### Key Features

- ✅ **Pretrained CodeBERT**: Leverages contextual embeddings from CodeBERT
- ✅ **Bidirectional LSTM**: Captures sequential dependencies in both directions
- ✅ **Multi-Head Attention**: Focuses on relevant parts of the input
- ✅ **Multiple Classification Heads**: Independent binary classifiers for each label
- ✅ **GPU Support**: Accelerated training with CUDA
- ✅ **Comprehensive Metrics**: Macro/Micro Precision, Recall, F1, Exact Match

---

## Model Architecture

### BBLA Model Overview

The **BBLAMultiLabelModel** combines four key components:

```
Input Text
    ↓
[1] CodeBERT Embeddings (768-dim)
    ↓
[2] Bi-LSTM Layer (2 layers, 512-dim each direction)
    ↓
[3] Multi-Head Self-Attention (4 heads)
    ↓
[4] Classification Heads (10 independent binary classifiers)
    ↓
Output Probabilities [0, 1] × 10 tags
```

### Component Details

#### 1. **CodeBERT Embeddings** (Frozen)
- Pre-trained transformer model from Microsoft
- Converts input text into 768-dimensional contextual embeddings
- **Frozen layers** to preserve learned knowledge and reduce training time
- Input: Tokenized text (max length: 512 tokens)
- Output: `[batch_size, seq_len, 768]`

#### 2. **Bi-LSTM Layers**
- **Architecture**: 2 stacked LSTM layers with bidirectional processing
- **Input size**: 768 (from CodeBERT)
- **Hidden size**: 512 per direction (total 1024 after concatenation)
- **Dropout**: 0.2 between layers
- **Purpose**: Capture sequential dependencies and long-range relationships
- Output: `[batch_size, seq_len, 1024]`

```python
self.bilstm = nn.LSTM(
    input_size=768,           # CodeBERT output
    hidden_size=512,          # Hidden dimension per direction
    num_layers=2,             # 2 stacked layers
    batch_first=True,
    bidirectional=True,       # Process left-to-right and right-to-left
    dropout=0.2
)
```

#### 3. **Multi-Head Self-Attention**
- **Number of heads**: 4
- **Input dimension**: 1024 (Bi-LSTM output size)
- **Purpose**: Dynamically weight different parts of the sequence
- **Mechanism**: Each attention head focuses on different aspects of the input
- Includes Layer Normalization and residual connections
- Output: `[batch_size, seq_len, 1024]`

```python
self.attention = CustomMultiHeadAttention(
    embed_dim=1024,           # Bi-LSTM output size
    num_heads=4,              # 4 attention heads
    dropout=0.2
)
```

#### 4. **Classification Heads** (10 Independent Heads)
- One binary classifier for each label/tag
- Architecture per head:
  ```
  Dropout → Linear(1024→256) → ReLU → BatchNorm → Dropout 
  → Linear(256→128) → ReLU → Dropout → Linear(128→1) → Sigmoid
  ```
- Each head outputs a probability in [0, 1]
- Independent: Each label decision doesn't affect others
- Output: `[batch_size, num_tags]` where each element ∈ [0, 1]

### Data Flow Example

```
Input: "How to handle exceptions in Python?"
    ↓
CodeBERT: 768-dim embedding for each token
    ↓
Bi-LSTM: Bidirectional sequence processing
    ↓
Attention: Weight tokens based on importance
    ↓
Pooling: Take last token as sequence representation
    ↓
10 Classifiers: [0.92, 0.15, 0.78, 0.33, ...] (probabilities)
    ↓
Output (threshold=0.5): [python, exception-handling, ...]
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- CUDA 11.0+ (for GPU acceleration, optional but recommended)
- 8GB+ RAM
- 4GB+ free disk space (for model downloads)

### Step 1: Clone the Repository

```bash
git clone https://github.com/programmerHoangBao/Multi-label_question_classification.git
cd Multi-label_question_classification
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n multilabel python=3.9
conda activate multilabel
```

### Step 3: Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers pandas numpy scikit-learn tqdm
```

Or install from requirements (if available):
```bash
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Model

The CodeBERT model will be auto-downloaded on first run, but you can pre-download it:

```bash
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/codebert-base')"
```

---

## Project Structure

```
Multi-label_question_classification/
├── main.py                          # Main entry point for training
├── config.py                        # Configuration settings
├── BBLAMultiLabelModel.py          # BBLA model implementation
├── CustomMultiHeadAttention.py      # Custom attention mechanism
├── Trainer.py                       # Training and evaluation logic
├── data_loader.py                   # Data loading and preprocessing
├── README.md                        # This file
├── data_100k_10_tags/               # Dataset directory
│   ├── train.parquet                # Training data (100k samples)
│   ├── val.parquet                  # Validation data
│   └── test.parquet                 # Test data
├── models/                          # Saved models directory (created at runtime)
│   ├── bbla_model.pt                # Model checkpoint
│   ├── TAGS.txt                     # Label mapping
│   ├── TAG_TO_IDX.txt              # Label to index mapping
│   └── IDX_TO_TAG.txt              # Index to label mapping
└── test_results.csv                 # Test metrics (created after training)
```

---

## Configuration

Edit `config.py` to customize training parameters:

```python
class Config:
    # Data paths
    TRAIN_PATH = "./data_100k_10_tags/train.parquet"
    VAL_PATH = "./data_100k_10_tags/val.parquet"
    TEST_PATH = "./data_100k_10_tags/test.parquet"
    
    # Model parameters
    MODEL_PATH = "./codebert-base"           # Pre-trained model
    LSTM_HIDDEN_SIZE = 512                   # Bi-LSTM hidden size
    NUM_ATTENTION_HEADS = 16                 # Attention heads
    DROPOUT = 0.2                            # Dropout rate
    
    # Training parameters
    TRAIN_BATCH_SIZE = 32                    # Training batch size
    VAL_BATCH_SIZE = 64                      # Validation batch size
    TEST_BATCH_SIZE = 64                     # Test batch size
    NUM_EPOCHS = 10                          # Number of epochs
    LEARNING_RATE = 1e-4                     # Learning rate
    WEIGHT_DECAY = 1e-5                      # L2 regularization
    
    # Device
    DEVICE = "cuda"                          # Use "cpu" if GPU not available
    
    # Prediction threshold
    PREDICTION_THRESHOLD = 0.5               # Probability threshold for label assignment
    
    # Other
    MAX_LENGTH = 512                         # Max token length
    SEED = 42                                # Random seed for reproducibility
    SAVE_PATH = "./models/bbla_model.pt"    # Model save location
```

### Configuration Tips

- **TRAIN_BATCH_SIZE**: Increase for faster training (if memory allows), decrease if OOM errors occur
- **NUM_EPOCHS**: Start with 10-20 epochs, monitor validation loss
- **LEARNING_RATE**: 1e-4 works well; decrease if loss doesn't converge
- **PREDICTION_THRESHOLD**: Adjust based on your precision/recall trade-off needs
- **NUM_ATTENTION_HEADS**: Must divide LSTM output size (1024)

---

## Usage

### Quick Start

Run the complete pipeline (train → validate → test):

```bash
python main.py
```

This will:
1. Load and prepare data
2. Create the BBLA model
3. Train for specified epochs
4. Evaluate on validation set
5. Report test metrics
6. Save model and results

### Expected Output

```
================================================================================
Multi-Label Classification with Attention-based Multi-Head
================================================================================

Loading data...
Creating model...
Total parameters: 109,234,560
Trainable parameters: 7,234,560

Training model...
Epoch 1/10: Train Loss: 0.4532 | Val Loss: 0.3821 | Val F1: 0.7234
Epoch 2/10: Train Loss: 0.3421 | Val Loss: 0.3234 | Val F1: 0.7654
...

================================================================================
TEST SET RESULTS
================================================================================
Test Loss:            0.3012
Test Macro Precision: 0.7891
Test Macro Recall:    0.7654
Test Macro F1:        0.7771
Test Micro Precision: 0.8234
Test Micro Recall:    0.8156
Test Micro F1:        0.8195
Test Exact Match:     0.6234
Train time: 3600.1234 s
================================================================================
```

### Customizing Training

To modify training behavior, edit `config.py` and run:

```python
# Example: Train for more epochs with larger batch size
# In config.py:
NUM_EPOCHS = 20
TRAIN_BATCH_SIZE = 64
LEARNING_RATE = 5e-5
```

Then run:
```bash
python main.py
```

---

## Training Details

### Training Process (in `main.py`)

1. **Data Loading**: Loads train/val/test data and creates DataLoaders
2. **Model Initialization**: Creates BBLA model with frozen CodeBERT
3. **Training Loop**:
   - Forward pass through model
   - Calculate multi-label loss (Binary Cross-Entropy per label)
   - Backpropagation
   - Optimizer step
4. **Validation**: Evaluate on validation set after each epoch
5. **Best Model Saving**: Save model with best validation F1 score
6. **Test Evaluation**: Final evaluation on held-out test set

### Loss Function

Binary Cross-Entropy (BCE) with logits:
```
Loss = -1/N ∑∑ [y_ij * log(ŷ_ij) + (1-y_ij) * log(1-ŷ_ij)]
```
Where:
- N = batch size
- i = sample index
- j = label index
- y_ij = true label (0 or 1)
- ŷ_ij = predicted probability

### Optimization

- **Optimizer**: AdamW (Adam with weight decay)
- **Learning Rate Scheduler**: Optional (can be added in Trainer)
- **Gradient Clipping**: Prevents exploding gradients
- **Mixed Precision Training**: Supported for faster training

---

## Evaluation Metrics

### Metrics Explained

- **Macro Precision**: Average precision across all labels
- **Macro Recall**: Average recall across all labels
- **Macro F1**: Harmonic mean of macro precision and recall
- **Micro Precision**: Precision computed globally
- **Micro Recall**: Recall computed globally
- **Micro F1**: Harmonic mean of micro precision and recall
- **Exact Match**: Percentage of samples where predicted labels exactly match ground truth
- **Inference Time**: Time required for predictions

### Interpreting Results

```
High Macro F1 + Low Micro F1 → Good at rare labels, poor on common ones
High Micro F1 + Low Macro F1 → Good at common labels, poor on rare ones
High Exact Match → Model predicts complete label sets correctly
```

---

## Results

### Expected Performance

On a 100k sample dataset with 10 labels:

| Metric | Value |
|--------|-------|
| Macro F1 | ~0.75-0.80 |
| Micro F1 | ~0.80-0.85 |
| Exact Match | ~0.60-0.70 |
| Training Time (per epoch) | ~30-60 minutes (on GPU) |
| Inference Speed | ~100-200 samples/second |

### Improving Performance

1. **Increase LSTM Hidden Size**: 512 → 1024
2. **Use More Attention Heads**: 4 → 8
3. **Fine-tune CodeBERT**: Remove frozen layers
4. **Data Augmentation**: Add synonyms, paraphrasing
5. **Ensemble Methods**: Combine multiple models
6. **Hyperparameter Tuning**: GridSearch or Bayesian optimization

---

## Requirements

### Python Packages

```
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

### System Requirements

- **Minimum**: 4GB RAM, CPU-only
- **Recommended**: 8GB+ RAM, GPU with 6GB+ VRAM
- **Disk Space**: 5GB for models and data
- **OS**: Linux, macOS, or Windows

### GPU Support

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# List available GPUs
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## Troubleshooting

### Common Issues

#### 1. **CUDA Out of Memory**
```bash
# Solution: Reduce batch size in config.py
TRAIN_BATCH_SIZE = 16  # from 32
VAL_BATCH_SIZE = 32    # from 64
```

#### 2. **Model Takes Too Long to Train**
```bash
# Solution 1: Use GPU
DEVICE = "cuda"

# Solution 2: Reduce epochs
NUM_EPOCHS = 5  # from 10

# Solution 3: Increase batch size (if memory allows)
TRAIN_BATCH_SIZE = 64  # from 32
```

#### 3. **Poor Model Performance**
- Increase LSTM_HIDDEN_SIZE (512 → 1024)
- Reduce DROPOUT (0.2 → 0.1)
- Increase LEARNING_RATE (1e-4 → 5e-4)
- Use more training data
- Check data quality and label balance

#### 4. **Data Loading Errors**
```bash
# Solution: Verify data file paths
# Check if files exist:
ls -la data_100k_10_tags/

# Re-download or regenerate data if corrupted
```

#### 5. **ModuleNotFoundError: No module named 'xxx'**
```bash
# Solution: Install missing package
pip install <module_name>

# Or reinstall all requirements
pip install -r requirements.txt --force-reinstall
```

#### 6. **CodeBERT Download Issues**
```bash
# Solution: Pre-download model
python -c "from transformers import AutoModel; AutoModel.from_pretrained('microsoft/codebert-base', cache_dir='./models')"

# Then update config.py:
MODEL_PATH = "./models/codebert-base"
```

---

## Advanced Usage

### Using Pretrained Model for Inference

```python
import torch
from BBLAMultiLabelModel import BBLAMultiLabelModel

# Load model
model = BBLAMultiLabelModel(model_path="microsoft/codebert-base")
model.load_state_dict(torch.load("./models/bbla_model.pt"))
model.eval()

# Load label mappings
with open("./models/TAGS.txt") as f:
    TAGS = [line.strip() for line in f]

# Inference
with torch.no_grad():
    # Your tokenized input
    predictions = model(input_ids, attention_mask)
    # Apply threshold
    labels = (predictions > 0.5).int()
```

### Modifying Model Architecture

Edit `BBLAMultiLabelModel.py`:

```python
# Increase LSTM layers
self.bilstm = nn.LSTM(
    num_layers=3,  # Changed from 2
    ...
)

# Add more attention heads
self.attention = CustomMultiHeadAttention(
    num_heads=8,  # Changed from 4
    ...
)
```

---

## Citation

If you use this project in your research, please cite:

```bibtex
@software{hoangbao2024multilabel,
  author = {Hoang Bao, Programmer},
  title = {Multi-Label Question Classification with BERT-Bi-LSTM-Attention},
  year = {2024},
  url = {https://github.com/programmerHoangBao/Multi-label_question_classification}
}
```

---

## License

This project is open source. Please check the LICENSE file for details.

---

## Contact & Support

For questions or issues, please:
1. Check the [Troubleshooting](#troubleshooting) section
2. Open a GitHub Issue
3. Contact the maintainer

---

## Changelog

### Version 1.0.0 (2024)
- Initial release
- BBLA model architecture
- Training and evaluation pipeline
- Comprehensive documentation

---

## References

- **CodeBERT**: [Microsoft CodeBERT Paper](https://arxiv.org/abs/2002.08155)
- **LSTM**: [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf)
- **Attention Mechanism**: [Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)
- **Multi-Label Learning**: [Zhang & Zhou (2014)](https://ieeexplore.ieee.org/document/6471714)

---

**Happy Training! 🚀**

For more information and updates, visit: https://github.com/programmerHoangBao/Multi-label_question_classification
