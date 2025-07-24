


**Sign Language Detection Using MediaPipe and Machine Learning**

This project presents a real-time sign language detection system that utilizes the MediaPipe Hands framework for extracting skeletal hand landmarks and applies machine learning classifiers for gesture recognition. The system is designed to classify static hand signs of the English alphabet and form sentences by recognizing sequences of hand gestures.

---

**Project Overview**

The objective is to bridge communication gaps for individuals with hearing or speech impairments by enabling gesture-to-text conversion using hand landmarks. The system captures hand signs using a webcam, extracts 21 key landmarks using MediaPipe, and classifies them using Random Forest and Support Vector Machine (SVM) models. It focuses on static hand gestures (A–Z) and includes a sentence formation module.

---

**Technologies Used**

* MediaPipe Hands (Google) – for real-time detection of 21 hand landmarks
* Scikit-learn – for model training and evaluation
* OpenCV – for webcam input and visualization
* Python – overall development and data processing
* Custom Dataset – collected manually using webcam and landmark extraction

---

**Dataset and Feature Extraction**

Each recorded gesture is represented by 21 landmarks with (X, Y, Z) coordinates, resulting in 63 features per sample. These features are stored in CSV format with a corresponding gesture label (e.g., A, B, C, ...). The dataset was collected under varied lighting and background conditions to improve generalization.

---

**Model Training and Evaluation**

Two classifiers were trained and tested using the dataset:

* **Random Forest Classifier**: Achieved a peak accuracy of 99.13% with 90 estimators.
* **SVM Classifier**:

  * Linear Kernel (C = 100): 99.78% accuracy
  * RBF Kernel (C = 100): 99.78% accuracy
  * Polynomial Kernel (Degree > 3): Lower performance due to overfitting

Model evaluation was done using precision, recall, F1-score, and confusion matrices.

---

**Key Features**

* Real-time hand landmark detection using MediaPipe
* High-accuracy static gesture classification (A–Z)
* Sentence formation by sequencing recognized gestures
* Support for live testing via webcam
* Demonstration videos for:

  * Landmark point extraction
  * Sentence formation (e.g., "I AM SAM")

Simulation videos are included in the `videos/` directory.

---

**Results Summary**

* Overall accuracy (Random Forest): 99.13%
* Overall accuracy (SVM with RBF/Linear): 99.78%
* Most letters achieved precision and recall above 0.95
* Certain gestures like 'M' and 'E' were occasionally misclassified due to visual similarity
* Dynamic gestures (e.g., 'J', 'Z') are currently not supported

---

**Limitations**

* Only supports static gestures; dynamic gesture recognition is not implemented
* Limited variation in dataset (same users, backgrounds)
* Misclassification possible with visually similar signs or poor lighting

---

**Future Enhancements**

* Integrate support for dynamic gestures using LSTM or temporal modeling
* Extend dataset with diverse hand shapes, lighting, and environments
* Develop mobile or edge-optimized version (e.g., using TensorFlow Lite)
* Combine with speech synthesis for gesture-to-speech translation

---

