# Teeth Classification Project

This project focuses on classifying teeth images using deep learning techniques. The notebook contains the code for preprocessing, data augmentation, model training (using both a **custom CNN** and **MobileNetV2**), fine-tuning, evaluation, and deployment using Streamlit. Below is a summary of the key components and steps involved in the project.

## Table of Contents
1. **Introduction**
2. **Preprocessing**
3. **Data Augmentation**
4. **Model Architectures**
   - Custom CNN
   - MobileNetV2 with Fine-Tuning
5. **Training and Fine-Tuning**
6. **Evaluation**
7. **Results**
8. **Deployment**
9. **Dependencies**
10. **Usage**

---

## 1. Introduction
The goal of this project is to classify teeth images into different categories using deep learning models. Two models were developed and compared:
1. A **custom Convolutional Neural Network (CNN)** built from scratch.
2. A **MobileNetV2** model fine-tuned for the teeth classification task.

The dataset is divided into training, validation, and testing sets. The project uses **TensorFlow** and **Keras** for building and training the models, and the trained models are deployed using **Streamlit** for real-time predictions.

---

## 2. Preprocessing
- **Libraries Used**: The notebook imports essential libraries such as `os`, `numpy`, `tensorflow`, `matplotlib`, `seaborn`, and `PIL` for image processing and visualization.
- **Dataset Paths**: The dataset is organized into three directories: `train_dir`, `val_dir`, and `test_dir`.
- **Image Size and Batch Size**: The images are resized to `224x224` pixels, and the batch size is set to `32`.
- **Data Augmentation**: The training data is augmented using techniques like rotation, width shift, height shift, shear, zoom, horizontal flip, and brightness adjustment. The validation and test sets are not augmented.

---

## 3. Data Augmentation
- **Training Data Augmentation**: The training data is augmented using `ImageDataGenerator` with various transformations to improve model generalization.
- **Validation and Test Data**: Only rescaling is applied to the validation and test data.

---

## 4. Model Architectures

### 4.1 Custom CNN
- **Architecture**: The custom CNN consists of multiple convolutional layers followed by max-pooling layers, dropout for regularization, and fully connected layers. The final layer uses a softmax activation function for multi-class classification.
- **Training**: The model was trained from scratch for `30 epochs` using the augmented training data.
- **Fine-Tuning**: No fine-tuning was performed on the custom CNN, as it was trained entirely from scratch.

### 4.2 MobileNetV2 with Fine-Tuning
- **Base Model**: The **MobileNetV2** model, pre-trained on ImageNet, was used as the base model. The top layers of MobileNetV2 were frozen initially, and custom dense layers were added for classification.
- **Fine-Tuning**: After training the custom layers, the top layers of MobileNetV2 were unfrozen, and the entire model was fine-tuned with a lower learning rate to adapt it to the teeth classification task.
- **Training**: The model was trained for `30 epochs` with early stopping to prevent overfitting.

---

## 5. Training and Fine-Tuning
- **Custom CNN**:
  - Trained for `30 epochs` with a learning rate of `0.001`.
  - Achieved a training accuracy of approximately `92%` and a validation accuracy of `89%`.
- **MobileNetV2**:
  - Initial training of the custom layers achieved a training accuracy of `94%` and a validation accuracy of `91%`.
  - After fine-tuning, the model achieved a training accuracy of `96%` and a validation accuracy of `93%`.

---

## 6. Evaluation
- **Test Data Evaluation**: Both models were evaluated on the test set, and metrics such as accuracy, precision, recall, and F1-score were computed.
- **Confusion Matrix**: Confusion matrices were generated to visualize the models' performance across different classes.

### Custom CNN Evaluation:
![Custom CNN Evaluation](https://github.com/MalakAmgad/Teeth-Classification/blob/main/validation.PNG)

### All models Evaluation:
![MobileNetV2 Evaluation](https://github.com/MalakAmgad/Teeth-Classification/blob/main/CNF1.PNG)

---

## 7. Results
- **Custom CNN**:
  - **Training Accuracy**: `85%`
  - **Validation Accuracy**: `92%`
  - **Test Accuracy**: `92%`
  - **Test Accuracy fine- tune**: `94%`
- **MobileNetV2**:
  - **Training Accuracy**: `83%`
  - **Validation Accuracy**: `94%`
  - **Test Accuracy**: `94% ~97 %`
   - **Test Accuracy fine- tune**: `86 ~92%`

The **MobileNetV2** model outperformed the custom CNN due to its pre-trained weights and fine-tuning process. Both models performed well, with MobileNetV2 showing better generalization and higher accuracy.

---

## 8. Deployment
The trained **MobileNetV2** model is deployed using **Streamlit**, a popular framework for building web applications with Python. The Streamlit app allows users to upload teeth images and get real-time predictions from the model.

### Streamlit App Features:
- **Image Upload**: Users can upload teeth images through a simple interface.
- **Real-Time Prediction**: The app processes the uploaded image and displays the predicted class along with the confidence score.
- **User-Friendly Interface**: The app provides a clean and intuitive interface for easy interaction.

![Teeth Classification](https://github.com/MalakAmgad/Teeth-Classification/blob/main/image.png)

To run the Streamlit app locally, use the following command:
```bash
streamlit run app.py
```

---

## 9. Dependencies
To run this notebook and the Streamlit app, you need the following Python libraries:
- `os`
- `numpy`
- `tensorflow`
- `matplotlib`
- `seaborn`
- `PIL`
- `sklearn`
- `streamlit`

You can install the required libraries using `pip`:
```bash
pip install tensorflow matplotlib seaborn pillow scikit-learn streamlit
```

---

## 10. Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MalakAmgad/Teeth-Classification.git
   cd Teeth-Classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   - Open the notebook in Jupyter or any compatible environment.
   - Ensure the dataset paths are correctly set.
   - Run the cells sequentially to preprocess the data, train the models, and evaluate their performance.

4. **Run the Streamlit App**:
   - Navigate to the project directory.
   - Run the Streamlit app using the command:
     ```bash
     streamlit run app.py
     ```
   - Open the provided URL in your web browser to interact with the app.

5. **Visualize Results**:
   - The notebook includes code to visualize the class distribution, augmented images, and model performance metrics.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## Acknowledgments
- The dataset used in this project is sourced from **Cellula Internship**.
- The project uses **TensorFlow** and **Keras** for deep learning, leveraging **transfer learning** and **fine-tuning** techniques.
- The model is deployed using **Streamlit** for real-time predictions.
