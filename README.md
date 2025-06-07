
# Knee X-ray Classification using Custom CNN and Transfer Learning (ResNet50)

This project implements a complete pipeline for classifying knee X-ray images into three categories: **normal**, **osteopenia**, and **osteoporosis**. It uses both a custom-built Convolutional Neural Network (CNN) and a transfer learning approach using **ResNet50**.

---

## 📁 Dataset

The dataset is obtained from an external source and may be nested inside multiple ZIP files. The structure after extraction should include folders named:

```
normal/
osteopenia/
osteoporosis/
```

The code automatically downloads, extracts, and prepares the dataset for training, validation, and testing.

---

## 📦 Dependencies

This project is intended to run on **Google Colab**. It uses the following libraries:

* `TensorFlow`
* `Keras`
* `scikit-learn`
* `pandas`
* `matplotlib`
* `seaborn`
* `tqdm`

These are automatically installed/updated at the beginning of the script.

---

## ⚙️ Key Features

### ✅ Automatic Dataset Handling

* Downloads nested zip files
* Handles extraction of inner and outer ZIP archives
* Validates presence of target class folders

### ✅ Data Preparation

* Stratified split into training, validation, and test sets
* Uses `ImageDataGenerator` for augmentation and normalization
* Computes class weights to handle data imbalance

### ✅ Models

Two models are built and compared:

1. **Custom CNN**

   * Built from scratch with Conv2D, BatchNorm, MaxPooling, and Dropout layers.
   * Trained for 20 epochs.

2. **Transfer Learning with ResNet50**

   * Uses pretrained weights on ImageNet.
   * Top layers customized for 3-class classification.
   * Trained for 15 epochs.

### ✅ Evaluation

* Accuracy, Precision, Recall, and F1-score
* Confusion matrix and classification report
* Performance comparison between CNN and ResNet50

---

## 📊 Output Example

The script prints classification metrics like:

```
Custom CNN -> Accuracy: 0.89, Precision: 0.88, Recall: 0.89, F1-score: 0.88

Transfer Learning (ResNet50) -> Accuracy: 0.93, Precision: 0.92, Recall: 0.93, F1-score: 0.92
```

---

## 🔧 Hyperparameters

You can modify the following hyperparameters to improve results:

```python
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

EPOCHS_CNN = 20
EPOCHS_TRANSFER = 15

LEARNING_RATE_CNN = 0.001
LEARNING_RATE_TRANSFER = 0.0001
```

---

## 📍 Project Structure

```
/content/
  ├── Knee-X-ray_outer.zip
  ├── first_extracted_zip_contents/
  ├── final_dataset_extracted/
  ├── temp_dataset_split/
        ├── train/
        ├── validation/
        ├── test/
```

---

## 📌 Notes

* The code checks GPU availability and enables memory growth.
* File handling uses `shutil`, `zipfile`, and `os`.
* Visualization and plotting modules are included but not explicitly used in this part of the code.

---

## 🚀 Getting Started (on Colab)

1. Run the notebook.
2. Wait for the dataset download and extraction.
3. Training starts for both models.
4. Final metrics and performance reports are displayed.

---

## 📜 License

This project is for educational and research purposes. Always cite the dataset provider if used in publications.

