# **Fraud Detection with MLflow Tracking**  

## **Overview**  
This project focuses on building, training, and tracking a **fraud detection model** using MLflow. The model leverages a **RandomForest Classifier** to identify fraudulent transactions. You will use **Google Colab** for execution, manually upload the required datasets, and log model metrics and artifacts with **MLflow** to monitor performance.

## **Objectives**  
The key tasks in this project include:  
1. **Uploading datasets**:  
   - `Fraud_Data.csv`: Contains transactional data with fraud labels.  
   - `IpAddress_to_Country.csv`: Maps IP addresses to their respective countries.  
2. **Preprocessing the data**:  
   - Splitting data into training and test sets.  
3. **Training the model**:  
   - Using a **RandomForest Classifier** for fraud prediction.  
4. **Tracking experiments with MLflow**:  
   - Logging metrics and saving the trained model.  
5. **Downloading the trained model** as a `.pkl` file for future use.  
6. **Documenting performance metrics** using accuracy, confusion matrix, and classification report.

## **Project Structure**  
- **`mlflow.ipynb`**: The notebook containing all code for data loading, preprocessing, model training, and logging with MLflow.  
- **Data files**:  
  - `Fraud_Data.csv`: Transactional data.  
  - `IpAddress_to_Country.csv`: IP-to-country mapping for contextual analysis.  

## **How to Execute the Project in Google Colab**  
### **Step 1: Set Up Google Colab**  
1. Open [Google Colab](https://colab.research.google.com/).  
2. Upload the `mlflow.ipynb` notebook.  
3. In the notebook, **execute the cells step-by-step** to set up MLflow and import libraries.

### **Step 2: Upload the Required Datasets**  
- Run the code cell prompting you to upload files manually from your drive.  
- Upload both:  
  - `Fraud_Data.csv`  
  - `IpAddress_to_Country.csv`

### **Step 3: Preprocess the Data**  
- The `Fraud_Data.csv` will be loaded and split into **features (X)** and **target (y)** columns.  
- Use **train-test split** to divide the data (70% training, 30% testing).

### **Step 4: Train the Model and Log with MLflow**  
- A **RandomForest Classifier** will be trained using the training data.  
- The model will predict fraud on the test set, and **performance metrics** (like accuracy) will be computed.  
- MLflow will **log the model and metrics** (accuracy, confusion matrix, etc.) for tracking purposes.  

### **Step 5: Save and Download the Trained Model**  
- The trained model will be saved as `fraud_detection_model.pkl`.  
- Download the model to your local machine for further use.

## **Performance Metrics Logged in MLflow**  
1. **Accuracy**: Measures the percentage of correct predictions.  
2. **Confusion Matrix**: Shows the distribution of true positives, false positives, true negatives, and false negatives.  
3. **Classification Report**: Displays precision, recall, f1-score, and support for each class.  

## **MLflow Experiment Logs**  
- MLflow will save the experiment logs, metrics, and the trained model in a directory (`/content/mlruns`) on Colab.  
- You can **inspect experiment metrics** later using MLflow’s UI or commands.

## **Project Deliverables**  
1. **Notebook (`mlflow.ipynb`)**: Contains all the code and outputs.  
2. **Trained Model (`fraud_detection_model.pkl`)**: Downloaded after training.  
3. **MLflow Logs**: Available for tracking and version control.

## **How to Inspect MLflow Logs**  
- In the notebook, MLflow logs the experiment to `/content/mlruns`.  
- You can **download the logs** for further inspection or connect to an external tracking server to visualize the experiment run history.

## **Usage of the Trained Model**  
- The trained RandomForest model can be used to predict fraud on **new transactional data**.  
- Load the `.pkl` model in any Python environment using:  
  ```python
  import pickle
  with open('fraud_detection_model.pkl', 'rb') as f:
      model = pickle.load(f)
  predictions = model.predict(new_data)
