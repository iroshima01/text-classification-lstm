# text-classification-lstm
## Sentiment Analysis using LSTM on IMDB Dataset

This project involves implementing an **LSTM-based model** for **binary sentiment classification** using the **IMDB Movie Reviews Dataset**. The implementation includes data preprocessing, word embeddings, custom LSTM cell implementation, and model evaluation.

---

## **1. Data Preparation**

### **1.1 Load and Explore the IMDB Dataset**
- Load the dataset in CSV format using **pandas**.
- Perform **Exploratory Data Analysis (EDA)** to understand the dataset.
- Optionally, use a **subset** of the dataset to balance computational efficiency and accuracy.

### **1.2 Text Preprocessing**
- Convert text to **lowercase**, remove **special characters, punctuation, and extra spaces**.
- Tokenize text into sequences of words using **nltk** or **torchtext**.
- Convert sentiment labels to numerical values.

### **1.3 Word Embeddings using Word2Vec**
- Load pre-trained **Word2Vec embeddings** (e.g., `fasttext-wiki-news-subwords-300` from `gensim.downloader`).
- Map words in reviews to their corresponding **vector representations**.

### **1.4 Visualizing Word Vectors**
- Select **100–500** common or sentiment-rich words.
- Retrieve **word vectors** and reduce dimensions using **PCA**.
- Visualize word distributions using **2D scatter plots** and **3D interactive plots**.

### **1.5 Preparing Data for Model Training**
- Convert tokens to **indices** from the vocabulary.
- **Pad sequences** to a uniform length (decide an optimal padding length).
- Create an **embedding matrix** with shape `[vocab_size, embedding_dim]`.
- Load pre-trained embeddings using **`nn.Embedding.from_pretrained`** in PyTorch.
- Decide whether to **freeze** or **fine-tune** embeddings during training.

### **1.6 Dataset Management**
- Use **PyTorch’s Dataset and DataLoader** classes.
- Split dataset into:
  - **70% Training Set**
  - **20% Validation Set**
  - **10% Test Set**

---

## **2. LSTM Model Implementation**

### **2.1 Custom LSTM Cell Implementation**
- Implement **forget, input, and output gates**.
- Update **cell state** and compute **hidden state** for each time step.
- Explain the role of each gate in processing sequential data.

### **2.2 Building the LSTM Model**
- Integrate the **embedding layer** into the model.
- Implement a **custom LSTM layer**.
- Use a **fully connected layer** with **sigmoid activation** for binary classification.

### **2.3 Training Process**
- Define an **appropriate loss function** for binary classification.
- Choose an **optimizer (Adam / SGD)** and tune its parameters.
- Train the model using **mini-batches**:
  - Implement forward pass to compute predictions.
  - Compute loss and backpropagate gradients.
  - Update model parameters using the optimizer.
- Monitor model performance on **validation set**.
- Train for at least **50 epochs** and experiment with different **learning rates** and **batch sizes**.

### **2.4 Model Evaluation**
- Test the best model on the test set.
- Compute:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
- Visualize the **confusion matrix**.
- Analyze performance and summarize findings.

---
