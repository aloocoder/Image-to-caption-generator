## 🖼️ Image Caption Generator using Flickr8k Dataset

This project builds an **Image Caption Generator** using deep learning techniques and the **Flickr8k** dataset. The core objective is to generate meaningful captions for a given image by combining **Computer Vision** (for feature extraction) and **Natural Language Processing** (for language modeling).

### 🔍 Overview

The model is trained on the Flickr8k dataset, which contains 8,000 images, each with 5 corresponding human-written captions. The model architecture uses a combination of **CNNs (VGG16)** for image feature extraction and **LSTM** layers for generating natural language captions.


### 🧰 Technologies & Tools
- **Python**
- **TensorFlow / Keras**
- **NumPy / Pandas**
- **Matplotlib**
- **VGG16 (pre-trained model)**
- **LSTM (for sequence modeling)**
- **Tokenizer (Keras)**


### 📦 Dataset
- **Images:** Sourced from the Flickr8k dataset.
- **Captions:** Cleaned and preprocessed text associated with each image.
- Directory: `Flickr8k_Dataset` and `Flickr8k_text`.

### 📌 Workflow Summary

#### 1. 📁 **Image Feature Extraction**
- Uses the **VGG16** model pre-trained on ImageNet.
- The last classification layer is removed.
- Images are passed through the model to extract feature vectors (4096-dim).

#### 2. 📝 **Captions Preprocessing**
- Captions are loaded and cleaned.
- Start (`<start>`) and end (`<end>`) tokens are added to each caption.
- Mapped with corresponding image IDs.

#### 3. 🧠 **Model Architecture**
- **Image Features → Dense Layer**
- **Captions → Embedding Layer → LSTM**
- Both branches are merged using the `add` layer and passed through Dense layers.
- Final output predicts the next word in the caption sequence.

#### 4. 🧪 **Training**
- Maximum caption length and vocabulary size are computed.
- Tokenizer is fitted on all captions.
- Data generator is used for efficient training on image-text pairs.
- Model is trained using `categorical_crossentropy`.

#### 5. 🔮 **Inference**
- At test time, images are passed to the trained model.
- The model generates one word at a time until the `<end>` token is predicted or max length is reached.


### 📈 Results
- The model is capable of generating syntactically correct and contextually relevant captions.
- Example:
  - Input: 🐶 (dog running in field)
  - Output: `"a dog is running through the grass"`


### 🚀 Future Improvements
- Use more advanced models like **InceptionV3**, **ResNet**, or **Transformer-based** models (e.g., ViT + GPT).
- Incorporate **BLEU Score** or **METEOR** for quantitative evaluation.
- Add attention mechanism for better alignment between image regions and words.

