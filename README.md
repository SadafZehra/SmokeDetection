# Industrial Smoke Detection: From Image Classification to Real-Time Localization

Built during my computer vision internship at **The Disrupt Labs** (Karachi), for safety monitoring on factory camera feeds.

Smoke is hard to detect: it has no fixed shape, it is partly transparent, and it is easy to confuse with steam, fog or bright walls. This project tackles it in two stages:

1. **Classification:** does this image contain smoke? (EfficientNetB2)
2. **Localization:** where exactly is the smoke? (YOLOv5s, real time)

## Key results

| Stage | Model | Result |
|---|---|---|
| Localization | **YOLOv5s** | **94.7% mAP@0.5** and 58.5% mAP@0.5:0.95 (best epoch), about **15 ms per image** on a Tesla T4 |
| Classification | EfficientNetB2 (ImageNet pretrained) | best validation accuracy 94.05% |

The detector is the final model: it is fast enough for live camera feeds and also tells the operator *where* the smoke is.

## Data

- Images from **factory cameras** plus smoke images from **Kaggle**.
- I **annotated about 1,000 images by hand** with bounding boxes, and managed the dataset and its versions in Roboflow.
- Detection dataset: **3,816 training** and **859 validation** images.
- Classification dataset: two classes (*Smoke*, *NotSmoke*), **19,331 training** images after augmentation and **1,313 validation** images.

## Stage 1: Smoke classification with EfficientNetB2

- **Backbone:** EfficientNetB2 pretrained on ImageNet, input 250 × 250.
- **Head:** global max pooling, dropout 0.4, dense layer (256, ReLU), softmax over 2 classes. 361K trainable parameters on top of a 7.8M-parameter frozen base.
- **Two-step transfer learning:**
  1. train only the new head with the backbone frozen (Adam, learning rate 1e-3);
  2. unfreeze the **top 20 layers** and fine-tune with a lower learning rate (1e-4).
- **Augmentation:** offline horizontal, vertical and combined flips and 30° shear to enlarge the dataset; online rotation (20°), zoom (0.2) and flips during training.

**What I learned from this stage:** validation accuracy moved a lot between epochs, and the best score was picked on the validation set itself, so 94.05% is optimistic. Classification also cannot say where the smoke is. Both points led to the second stage.

## Stage 2: Real-time localization with YOLOv5s

- **Model:** YOLOv5s (small variant), fine-tuned from pretrained weights.
- **Training:** 100 epochs on a Tesla T4 GPU (Google Colab).
- **Result:** best mAP@0.5 of **94.7%** (epoch 70) and **58.5% mAP@0.5:0.95**. At the final epoch, mAP@0.5 was 89.8%, so the best checkpoint (`best.pt`) is the one to deploy.
- **Speed:** about **15 ms per image**, so it can process live camera streams.

## How to run

1. Open the notebooks in Google Colab with a GPU runtime.
2. Add your own Roboflow API key as an environment variable or Colab secret (do not paste keys into the notebook):
   ```python
   import os
   from roboflow import Roboflow
   rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
   ```
3. **Detector:** download the dataset in YOLOv5 format, then train and run detection with the YOLOv5 `train.py` and `detect.py` scripts.
4. **Classifier:** point the Keras `ImageDataGenerator` to the `Train/` and `Validation/` folders (one sub-folder per class) and run the training cells.

## Main libraries

PyTorch · YOLOv5 · TensorFlow / Keras · EfficientNet · Roboflow · OpenCV · NumPy · Matplotlib

## Possible improvements

- Add hard negatives (steam, fog, dust, white walls) to reduce false alarms.
- Use short video clips instead of single frames, since smoke is easier to recognize by how it moves.
- Report results on a separate test set, not only on validation data.

## Author

**Sadaf Zehra**, MS Data Science, NED University of Engineering and Technology
[GitHub](https://github.com/SadafZehra) · [LinkedIn](https://www.linkedin.com/in/sadaf-zehra-796858230) · [ORCID](https://orcid.org/0009-0009-7530-9006)
