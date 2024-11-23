# Dangerous Content Detection and Text Classification with Machine Learning

Contact
=======

For any questions or feedback:

*   **Email**: saadhajari10@gmail.com
    
*   **Buy Me a coffee here --->**: **https://www.paypal.com/donate/?hosted_button_id=5URJR262Y77BQ**


![x](https://github.com/user-attachments/assets/6f76df4a-f6b6-477f-92e4-684caa575ce8) 

## Overview

This project provides a Python-based solution for two main use cases:

1. **Detect Dangerous Content**: Detect harmful or suspicious content in posts, such as:
   - Dangerous keywords (e.g., "attack", "violence").
   - Suspicious URLs from predefined blacklisted domains.
   - Excessive special characters or emojis.

2. **Text Classification using Machine Learning**:
   - Apply various Machine Learning algorithms to classify text data.
   - Compare two vectorization techniques: **TF-IDF** and **Bag of Words**.


---

## Features

- **Content Filtering**: Preprocess posts and flag those containing dangerous content or suspicious patterns.
- **Text Vectorization**: Transform textual data into numerical representations using:
  - **Bag of Words**
  - **TF-IDF**
- **Machine Learning Models**:
  - Logistic Regression
  - K-Nearest Neighbors (KNN)
  - Support Vector Machines (SVM)
  - Naive Bayes
  - Multi-Layer Perceptron (MLP)
  - Random Forest
  - Gradient Boosting
  - XGBoost
- **Model Evaluation**: Automatically computes metrics such as:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
- **Customizable Pipeline**: Add keywords, blacklisted domains, or new ML models with ease.

---

## Project Structure

```plaintext
x_classification/
├── data_x_posts.json      # Example dataset with posts
├── x_classification.py    # Main script for filtering and ML classification
├── README.md              # Project documentation
├── requirements.txt       # Python dependencies          

```

#### Installation

1. **git clone** https://github.com/your_username/classification_X.git
2.  **cd classification_X**
3.  **pip install -r requirements.txt**
4.  **python classification_X.py**



### Prerequisites

1. **Python Version**: Python 3.7 or higher.
2. **Libraries**: The script requires several Python libraries. Install them using:
   ```bash
   pip install -r requirements.txt



### The results include:

*   Metrics for each model.
    
*   Visualizations comparing model performance.
    

Results and Outputs
===================

Results Table
-------------

A sample table summarizing the results:



![image](https://github.com/user-attachments/assets/4680a6c0-07d3-4c23-9cbd-98ef39f38dac)


Visualizations
--------------

The script generates bar plots comparing model performances.

### Example Plot

Algorithms and Vectorization Techniques
---------------------------------------

The project leverages the following algorithms with two vectorization techniques:

*   **TF-IDF**
    
*   **Bag of Words**
    

### Illustration of the Workflow

Customization
=============

Update Dangerous Keywords
-------------------------

Modify the contains\_dangerous\_keywords function in x\_classification.py:

Plain

codedangerous_keywords = ['attack', 'violence', 'bomb', 'hacking', 'danger']   `

Add Suspicious Domains
----------------------

Update the contains\_suspicious\_links function:


Add a New ML Model
------------------

To add a new ML model:

1.  Define the model in the relevant section of the script.
    
2.  Update the evaluation pipeline to include the new model.
    

Contributing
============

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please:

*   Open an issue on GitHub.
    
*   Submit a pull request with detailed changes.
    

Future Features
============

To further enhance the project, consider adding the following feature:

**Generic Program or API with Endpoints** ✅
**Develop a generic program or RESTful API with endpoints to:** ✅

- Compare different machine learning models. ✅
- Switch between TF-IDF and Bag of Words vectorization techniques. ✅
- Serve predictions for new text data. ✅


**Example API Endpoints**

- **GET /models:** Retrieve a list of available ML models and their metrics. ✅
- **POST /predict:** Accept a text input and return predictions from a specified model and vector type. ✅
- **GET /compare:**  Generate a comparison chart of model performances for both vectorization techniques. ✅

  This addition would make the project highly extensible and allow users to interact programmatically with the machine learning pipeline.

License
=======

This project is licensed under the **Apache License 2.0**. See the LICENSE file for full details.
