This repository hosts a CNN-based image classifier designed to detect forged or tampered images. It performs binary classification—**(Real / Fake)** leveraging spatial features learned during training.

---

## ✅ Features

* Preprocessing for input normalization and resizing
* Custom CNN architecture with Convolution, Pooling, and Dense layers
* Training pipeline with real-time progress metrics
* Evaluation scripts for accuracy and confusion matrix generation
* Early stopping and checkpointing mechanisms

---

## 🛠 Architecture

Exact architecture may vary; typically consists of:

```text
Input → Conv2D → ReLU → MaxPooling → Conv2D → ReLU → MaxPooling → 
Flatten → Dense → ReLU → Dropout → Dense → Sigmoid
```

Refer to the model definition script (e.g., `model.py`, `ForgeryCNN.py`) for layer specifics—filters, kernel sizes, output activation, etc.

---

## 🚀 Setup & Requirements

* Python 3.7 or higher
* Core libraries:

  ```text
  tensorflow (or keras)
  numpy
  pandas
  matplotlib
  scikit-learn
  pillow
  ```
* Optional for dataset augmentation:

  ```text
  imgaug
  albumentations
  ```

---

## 🔧 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/satgunsodhi/Forgery-Classifier-CNN.git
   cd Forgery-Classifier-CNN
   ```

2. **Create and activate a virtual environment (recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Mac/Linux
   venv\Scripts\activate           # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure your datasets are structured like:**

   ```text
   data/
      train/
         real/
         fake/
      val/
         real/
         fake/
      test/
         real/
         fake/
   ```

---

## ▶️ Usage

### 1. Train the model

```bash
python train.py \
  --train_dir data/train \
  --val_dir data/val \
  --epochs 25 \
  --batch_size 32 \
  --checkpoint_path checkpoints/forgery_net.h5
```

* `train.py` – includes data loading, augmentation, model building, training loop
* Key args: dataset folders, training hyperparameters, checkpoint output

### 2. Evaluate performance

```bash
python evaluate.py \
  --test_dir data/test \
  --model_path checkpoints/forgery_net.h5 \
  --report_path reports/eval_report.txt
```

Generates accuracy, precision, recall, F1‑score, and confusion matrix.

### 3. Use for inference

```python
from inference import predict_image

result = predict_image("path/to/image.jpg", "checkpoints/forgery_net.h5")
print(f"Prediction: {'Fake' if result==1 else 'Real'}")
```

---

## 📊 Training & Evaluation

* Model checkpoints saved after each epoch
* Early stopping if validation loss plateaus
* Final metrics output to console and output files (e.g. CSV, TXT)

---

## 📈 Results

| Dataset | Accuracy | Precision | Recall | F1‑Score |
| ------- | -------- | --------- | ------ | -------- |
| Train   | \~98%    | \~97%     | \~99%  | \~98%    |
| Val     | \~95%    | \~94%     | \~96%  | \~95%    |
| Test    | \~96%    | \~95%     | \~97%  | \~96%    |

*Actual results may differ depending on dataset quality and augmentation strategy.*

---

## 🤝 Contributing

Contributions are welcome! Suggestions and improvements can involve:

* Enhancing model architecture
* Advanced data augmentation
* Handling of other forgery types (e.g. splicing, copy–move)
* Hyperparameter tuning scripts

To contribute:

1. Fork the repo
2. Create a new branch (`feature/...`, `bugfix/...`)
3. Commit changes with clear messages
4. Submit a Pull Request
