# 🎬 Action Recognition Using CNN–LSTM

**Hybrid MobileNetV2 + LSTM Model for Sequence-Based Action Classification**

This repository implements a **deep learning pipeline for human action recognition** using a **CNN–LSTM hybrid model**.
It also includes a **visual testing module** that loads a random video from the test set, processes it into frame sequences, predicts the action, and displays the **true vs predicted** labels.

---

## 📌 Features

### 🔹 **1. Action Recognition Model (CNN + LSTM)**

* Uses **MobileNetV2** for spatial feature extraction
* Uses **LSTM** to capture temporal motion patterns
* Trained on a selected subset of **UCF101** containing 5 actions:

  ```
  WalkingWithDog
  PushUps
  BrushingTeeth
  BlowingCandles
  Typing
  ```

### 🔹 **2. Preprocessing Pipeline**

* Converts raw videos into evenly sampled sequences of **16 frames**
* Performs resizing, normalization, and RGB conversion
* Handles short/long video sequences with automatic padding and sampling

### 🔹 **3. Optimizer Benchmarking**

Trains the model using three optimizers:

* **SGD**
* **Adam**
* **Adagrad**

For each optimizer, the code logs:

* Training accuracy & validation accuracy
* Training loss & validation loss
* Total training time
* Best validation accuracy

### 🔹 **4. Training Curve Visualization**

Generates accuracy and loss plots for all optimizers to compare training behavior.

---

## 🛠 Dependencies

* Python 3.x
* TensorFlow / Keras
* OpenCV (cv2)
* NumPy
* Matplotlib

---

## 🧪 Dataset

This project assumes the **UCF101** dataset is already organized into:

```
train/
val/
test/
```

Only these 5 classes are used to reduce training time:

* WalkingWithDog
* PushUps
* BrushingTeeth
* BlowingCandles
* Typing

---

## 📈 Results

* The notebook identifies the **best-performing optimizer** based on validation accuracy
* Training curves help visualize model convergence
* Test-time visualization ensures correct understanding of true vs predicted actions

---
