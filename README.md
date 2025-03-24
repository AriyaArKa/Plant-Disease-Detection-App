# Plant Disease Detection using Deep Learning 🌱📸

## Overview
This project implements a **deep learning-based plant disease detection system** using **Convolutional Neural Networks (CNNs)**. The model classifies plant leaves into three categories:
- **Corn - Common Rust** 🌽
- **Potato - Early Blight** 🥔
- **Tomato - Bacterial Spot** 🍅

The dataset consists of labeled plant leaf images stored in **Google Drive**, and the model is trained using **Keras & TensorFlow** in **Google Colab**.

---

## 🚀 Features
✅ Load and preprocess plant leaf images from Google Drive
✅ Convert images into NumPy arrays for training
✅ Train a **CNN model** for classification
✅ **Save and load the model** for future predictions
✅ Evaluate model performance with **training history & accuracy plots**
✅ Test the model on unseen data and compare predictions

---

## 🏗️ Project Structure
```
📂 PlantDiseasesDetection
 ├── Corn_(maize)___Common_rust_
 ├── Potato___Early_blight
 ├── Tomato___Bacterial_spot
 ├── plant_disease.h5                 # Saved CNN model
 ├── plant_model.json                  # Model architecture
 ├── plant_model_weights.weights.h5    # Model weights
```

---

## 📌 Technologies Used
- **Python** 🐍
- **Google Colab** (for training & evaluation)
- **TensorFlow / Keras** (for deep learning model)
- **OpenCV & Matplotlib** (for image processing & visualization)
- **NumPy & Pandas** (for data handling)
- **SQLite** (optional for database integration)

---

## 📥 Dataset Loading
The dataset is stored in **Google Drive** and mounted using:
```python
from google.colab import drive
drive.mount('/content/drive')
```
Then, images are loaded and converted into arrays:
```python
def convert_image_to_array(image_dir):
    image = cv2.imread(image_dir)
    if image is not None:
        image = cv2.resize(image, (256,256))
        return img_to_array(image)
    return np.array([])
```

---

## 🏗️ Model Architecture
The CNN model consists of:
- **2 Convolutional Layers** with ReLU activation
- **MaxPooling Layers** to reduce dimensionality
- **Flatten Layer** to convert image matrix to vector
- **Fully Connected Dense Layers**

```python
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", activation="relu", input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(16, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(8, activation="relu"))
model.add(Dense(3, activation="softmax"))
```

---

## 🏋️ Model Training
- **Train/Test Split**: 80% training, 20% validation
- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Epochs**: 50

```python
model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0001), metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=50, validation_data=(x_val, y_val))
```

---

## 📊 Model Evaluation
Plot accuracy during training:
```python
plt.plot(history.history['accuracy'], color='r')
plt.plot(history.history['val_accuracy'], color='b')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['Train', 'Validation'])
plt.show()
```

Calculate test accuracy:
```python
scores = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {scores[1]*100}")
```

---

## 🔍 Model Prediction
Test the model on an image:
```python
img = array_to_img(x_test[10])
print("Actual Label:", all_labels[np.argmax(y_test[10])])
print("Predicted Label:", all_labels[np.argmax(y_pred[10])])
```

---

## 💾 Model Saving & Loading
Save the model:
```python
model.save("/content/drive/My Drive/PlantDiseasesDetection/plant_disease.h5")
```
Save architecture & weights separately:
```python
with open('/content/drive/My Drive/PlantDiseasesDetection/plant_model.json', 'w') as json_file:
    json_file.write(model.to_json())
model.save_weights('/content/drive/My Drive/PlantDiseasesDetection/plant_model_weights.weights.h5')
```

---

## 📌 Future Enhancements
🚀 Increase dataset size for better generalization  
🚀 Improve CNN architecture for higher accuracy  
🚀 Deploy the model as a **web app** using **Flask or Streamlit**  
🚀 Implement **real-time detection** using a mobile camera  

---

## 🤝 Contributing
Feel free to contribute by adding new features, optimizing the model, or improving documentation! 🛠️

---

## 📜 License
This project is open-source and available under the **MIT License**.

---

## 📩 Contact
For questions or collaboration, reach out via:
📧 Email: natharka55@gmail.com  
🔗 LinkedIn: [Arka Nath](https://www.linkedin.com/in/arka-nath55/)  
