# Higher-Order Transformers with Kronecker-Structured Attention

This repository contains the official implementation of [our paper](https://openreview.net/forum?id=QN0aXcKFkT),
accepted at **Transactions on Machine Learning Research (TMLR), 2025**.

---

## 🚀 Setup

To install and run the code locally:

```bash
git clone https://github.com/s-omranpour/HOT.git
cd HOT
pip install -r requirements.txt
```

---

## 🧠 Tasks

### ⏱️ Time Series Forecasting

First, download and extract the datasets from [here](https://drive.google.com/file/d/1rEhROxFG9a4vFerP1MfZJZiqxJQzyPJ3/view?usp=sharing).
Then, run the following example command:

```bash
python timeseries_main.py --data_path=data/timeseries --name=weather --attention_type=kronecker_product ...
```

For a full list of available arguments, please refer to the `timeseries_main.py` file.

---

### 🧬 3D Medical Image Classification

Run the following example command:

```bash
python medmnist_main.py --data_path=data/medmnist --name=organ --attention_type=kronecker_product ...
```

The corresponding **MedMNIST3D** datasets will be downloaded automatically when running the script.

---

### 🌍 Multispectral Image Segmentation

Run the following example command:

```bash
python segmentation_main.py --data_path=data/ssl4eo-l --task=cdl --attention_type=kronecker_product ...
```

The **SSL4EO-L** dataset will be downloaded automatically when running the script.

---

## 📖 Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{omranpour2025hot,
  title={Higher-Order Transformers with Kronecker-Structured Attention},
  author={Omranpour, Soroush and Rabusseau, Guillaume and Rabbany, Reihaneh},
  journal={Transactions on Machine Learning Research},
  year={2025}
}
```
