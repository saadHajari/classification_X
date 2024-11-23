# Dangerous Content Detection and Text Classification with Machine Learning

Contact
=======

For any questions or feedback:

*   **Email**: saadhajari10@gmail.com
    
*   **Buy Me a coffe here --->**: **https://www.paypal.com/donate/?hosted_button_id=5URJR262Y77BQ**


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

Plain textANTLR4BashCC#CSSCoffeeScriptCMakeDartDjangoDockerEJSErlangGitGoGraphQLGroovyHTMLJavaJavaScriptJSONJSXKotlinLaTeXLessLuaMakefileMarkdownMATLABMarkupObjective-CPerlPHPPowerShell.propertiesProtocol BuffersPythonRRubySass (Sass)Sass (Scss)SchemeSQLShellSwiftSVGTSXTypeScriptWebAssemblyYAMLXML`   pythonCopy codedangerous_keywords = ['attack', 'violence', 'bomb', 'hacking', 'danger']   `

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
    

License
=======

This project is licensed under the **MIT License**. See the LICENSE file for full details.
