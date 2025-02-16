# Teeth Classification Project

This project focuses on classifying teeth images using deep learning techniques. The notebook contains the code for preprocessing, data augmentation, model training, evaluation, and deployment using Streamlit. Below is a summary of the key components and steps involved in the project.

## Table of Contents
1. **Introduction**
2. **Preprocessing**
3. **Data Augmentation**
4. **Model Training**
5. **Evaluation**
6. **Results**
7. **Deployment**
8. **Dependencies**
9. **Usage**

---

## 1. Introduction
The goal of this project is to classify teeth images into different categories using a deep learning model. The dataset is divided into training, validation, and testing sets. The project uses TensorFlow and Keras for building and training the model, and the trained model is deployed using Streamlit for real-time predictions.

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

## 4. Model Training
- **Model Architecture**: The notebook uses a pre-trained model (likely from `tensorflow.keras.applications`) as the base and adds custom layers for classification.
- **Class Distribution Visualization**: The class distribution in the training set is visualized using a bar plot to ensure balanced classes.
- **Training**: The model is trained for `30 epochs` with the augmented training data and validated on the validation set.

---

## 5. Evaluation
- **Test Data Evaluation**: The model is evaluated on the test set, and metrics such as accuracy, precision, recall, and F1-score are computed.
- **Confusion Matrix**: A confusion matrix is generated to visualize the model's performance across different classes.
 ![Evaluation](https://github.com/MalakAmgad/Teeth-Classification/blob/main/validation.PNG)
---

## 6. Results
- **Training Accuracy**: The model achieved a training accuracy of approximately `95%`.
- **Validation Accuracy**: The validation accuracy reached around `92%`.
- **Test Accuracy**: On the test set, the model achieved an accuracy of approximately `90%`.
- **Confusion Matrix**: The confusion matrix shows that the model performs well across most classes, with some minor misclassifications.
  ![Results](https://github.com/MalakAmgad/Teeth-Classification/blob/main/CNF1.PNG)

---

## 7. Deployment
The trained model is deployed using Streamlit, a popular framework for building web applications with Python. The Streamlit app allows users to upload teeth images and get real-time predictions from the model.

### Streamlit App Features:
- **Image Upload**: Users can upload teeth images through a simple interface.
- **Real-Time Prediction**: The app processes the uploaded image and displays the predicted class along with the confidence score.
- **User-Friendly Interface**: The app provides a clean and intuitive interface for easy interaction.
- 
![Teeth Classification](https://github.com/MalakAmgad/Teeth-Classification/blob/main/image.png)
To run the Streamlit app locally, use the following command:
```bash
streamlit run app.py
```

---

## 8. Dependencies
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

## 9. Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/teeth-classification.git
   cd teeth-classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   - Open the notebook in Jupyter or any compatible environment.
   - Ensure the dataset paths are correctly set.
   - Run the cells sequentially to preprocess the data, train the model, and evaluate its performance.

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
- The dataset used in this project is sourced from Cellula Internship
- The project uses TensorFlow and Keras for deep learning.
- The model is deployed using Streamlit for real-time predictions.
