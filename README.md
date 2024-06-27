# Disaster Tweets Classification

This project aims to classify tweets as either disaster-related or non-disaster-related by exploring two distinct machine learning strategies: a traditional approach using TF-IDF vectorization with a RandomForestClassifier, and a more advanced approach leveraging pre-trained FastText embeddings within a bidirectional LSTM neural network.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Overview
We compare the effectiveness of two different strategies in classifying tweets to enhance our understanding of how different text processing techniques and model architectures can impact performance in text classification tasks. The first strategy uses traditional TF-IDF vectorization and a RandomForestClassifier, while the second strategy uses pre-trained FastText embeddings within a bidirectional LSTM neural network.

## Dataset
The dataset for this project consists of tweets labeled as disaster-related or non-disaster-related. The following files are used:
- `train.csv` - Training data
- `test.csv` - Test data
- `sample_submission.csv` - Sample submission file

You can download the dataset from the [Kaggle competition page](https://www.kaggle.com/c/nlp-getting-started/data).

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/disaster-tweets-classification.git
    cd disaster-tweets-classification
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Download pre-trained FastText embeddings:
    ```bash
    wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip
    unzip wiki-news-300d-1M.vec.zip
    ```

## Usage
1. Preprocess the data and train the models by running the Jupyter notebook:
    ```bash
    jupyter notebook disaster_tweets_classification.ipynb
    ```

2. To generate predictions and create a submission file for Kaggle, follow the steps in the notebook.

## Results
### TF-IDF and RandomForestClassifier
- **Accuracy:** 77%
- **Confusion Matrix:**
    ```
    [[731 143]
     [209 440]]
    ```

### FastText and Bidirectional LSTM
- **Accuracy:** 80%
- **Confusion Matrix:**
    ```
    [[805  69]
     [228 421]]
    ```

### Kaggle Submission
- **Public Score:** 0.77076

## Future Work
To further enhance model performance, we could:
- Experiment with different neural network architectures, such as GRU or Transformer models.
- Fine-tune hyperparameters more extensively.
- Use more advanced embedding techniques like BERT or GPT.
- Implement additional data augmentation techniques to increase the robustness of the model.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
