# PyTorch Deep Learning Pipelines: CIFAR-10 Image Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Google%20Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)

An end-to-end implementation and progression of Deep Neural Network architectures for image classification on the CIFAR-10 dataset using PyTorch. This project demonstrates different levels of PyTorch abstractions—from low-level functional APIs (Barebones) to object-oriented modules (`nn.Module`) and high-level sequential containers (`nn.Sequential`).

---

## 📌 Repository Name Ideas

Here are a few recommended naming options for your GitHub repository:
- `pytorch-cifar10-deep-learning` *(Recommended)*
- `cifar10-pytorch-abstractions`
- `pytorch-image-classification-cifar10`

---

## 🚀 Project Overview

The goal of this project is to build, train, and evaluate various deep learning architectures using PyTorch while exploring its different levels of abstraction. 

The implementations cover:
1. **Low-Level Functional API (Barebones)**: Manual weight initialization and custom forward passes using explicit tensor operations.
2. **PyTorch `nn.Module` API**: Object-oriented neural network building blocks utilizing `torch.nn`.
3. **PyTorch `nn.Sequential` API**: Streamlined model composition using sequential containers paired with advanced optimizers (e.g., Nesterov Momentum).

---

## 📊 Dataset: CIFAR-10

The [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html) consists of 60,000 $32 \times 32$ color images across 10 classes (6,000 images per class):
* **Classes**: Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.
* **Split**: 49,000 training images, 1,000 validation images, and 10,000 test images.

---

## 📈 Model Performance & Results

All models were evaluated on the CIFAR-10 validation set over 1 epoch (or short training iterations) with a batch size of 64.

| Architecture | API / Abstraction Level | Optimizer | Val Accuracy | Target Accuracy | Status |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Two-Layer FC** | Barebones PyTorch | Custom SGD | **43.10%** | > 40.00% | ✅ Passed |
| **Three-Layer ConvNet** | Barebones PyTorch | Custom SGD | **53.00%** | > 42.00% | ✅ Passed |
| **Two-Layer FC** | `nn.Module` API | SGD | **45.60%** | > 40.00% | ✅ Passed |
| **Three-Layer ConvNet** | `nn.Module` API | SGD | **48.50%** | > 45.00% | ✅ Passed |
| **Three-Layer ConvNet** | `nn.Sequential` API | Nesterov Momentum | **55.90%** (Peak: 58.10%) | > 55.00% | ✅ Passed |

---

## 🛠️ Model Architectures

### 1. Two-Layer Fully Connected Network
- **Input**: $3 \times 32 \times 32$ image vector flattened to $3072$
- **Hidden Layer**: Fully connected layer ($3072 \to H$) + ReLU activation
- **Output Layer**: Fully connected layer ($H \to 10$)

### 2. Three-Layer Convolutional Network
- **Layer 1**: Conv2D ($3 \to C$, kernel size $5 \times 5$, padding 2) + ReLU
- **Layer 2**: Conv2D ($C \to C_2$, kernel size $3 \times 3$, padding 1) + ReLU
- **Layer 3**: Fully connected layer ($C_2 \times 32 \times 32 \to 10$)

---

## 💻 Tech Stack & Dependencies

* Python 3.x
* PyTorch (CUDA enabled for GPU acceleration)
* torchvision
* NumPy
* Matplotlib

---

## 🔧 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/pytorch-cifar10-deep-learning.git
   cd pytorch-cifar10-deep-learning
   ```

2. **Install dependencies**:
   ```bash
   pip install torch torchvision numpy matplotlib
   ```

3. **Run in Jupyter Notebook / Google Colab**:
   Open the `.ipynb` file in Google Colab or Jupyter Notebook and ensure GPU hardware acceleration is enabled under `Runtime -> Change runtime type -> Hardware accelerator (GPU)`.

---

## 📝 Key Takeaways

* **GPU Acceleration**: Leveraging PyTorch `cuda` tensors drastically speeds up convolutional computations.
* **Abstraction Benefits**: Moving from manual Barebones PyTorch to `nn.Module` and `nn.Sequential` simplifies model definitions and gradient tracking (`autograd`).
* **Optimization Matters**: Switching from standard SGD to **Nesterov Momentum** yields higher validation accuracy in fewer training steps.