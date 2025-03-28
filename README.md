Predicting Restoration Failures in Primary and Permanent Teeth – A Machine Learning Approach

This repository contains the source code for developing and validating machine learning models to predict posterior dental restoration failures. The code is organized into multiple notebooks, each illustrating different aspects of the data analysis workflow—from data preprocessing and exploratory analysis to model training, hyperparameter tuning, evaluation, and interpretability.

The repository is intended as a reference for researchers and practitioners interested in applying similar techniques to their own datasets. The methods implemented here may be adapted or extended for use in different studies.

Overview

This project demonstrates how to build and evaluate predictive models using advanced machine learning algorithms such as Decision Trees, Random Forests, XGBoost, CatBoost, and Neural Networks. The analytical framework covers:

Data preprocessing (imputation, scaling, and encoding)
Cluster-based train-test splitting (to ensure data integrity)
Model training with cross-validation and nested hyperparameter tuning
Performance evaluation using accuracy, sensitivity, specificity, F1-score, and ROC AUC
Model interpretability through SHAP analysis
Repository Structure

notebooks/
Contains Jupyter/Colab notebooks showcasing the code workflow. The notebooks are grouped as follows:

Permanent Teeth Analysis (CaCIA)

Permanent Teeth Analysis (CaCIA) Models: Decision Tree, Random Forest, XGBoost
Final Fev 2025 Cacia.ipynb

Permanent Teeth Analysis (CaCIA) Models: CatBoost
Final Fev 2025 CatBoost Cacia.ipynb

Permanent Teeth Analysis (CaCIA) Models: Neural Network
Final Fev 2025 NN Cacia.ipynb

Primary Teeth Analysis (CARDEC-3)

Primary Teeth Analysis (CARDEC) Global (any failure) - Models: Catboost
Final Fev 2025 Cardec Global CatBoost.ipynb

Primary Teeth Analysis (CARDEC) Global (any failure) - Models: Decision Tree, Random Forest, XGBoost, Neural Network
Final Fev 2025 Cardec Global.ipynb

Primary Teeth Analysis (CARDEC) Replacement - Models: Catboost
Final Fev 2025 Cardec Replac Catboost.ipynb

Primary Teeth Analysis (CARDEC) Replacement - Models: Decision Tree, Rnadom Forest, XGBoost, Neural Network
Final Fev 2025 Cardec Replac.ipynb


src/
(Optional) Utility scripts for data preprocessing, model training, and evaluation that can be imported into your own projects.
requirements.txt
Lists the Python dependencies needed to run the notebooks, including:
Python 3.10/3.11
scikit-learn==1.2.2
numpy==1.25.2
pandas==2.0.3
scipy==1.11.2
joblib==1.2.0
threadpoolctl==3.1.0
cython==0.29.36
imbalanced-learn==0.12.0
keras==3.5.0 & tensorflow==2.17.1 (for neural network implementations)
Project Description

Analytical Approach
The code demonstrates a comprehensive machine learning pipeline, including:

Data Splitting:
A patient-level train-test split is used to maintain the integrity of the data clusters.
Preprocessing:
Missing values are imputed using appropriate strategies, numerical features are scaled (using z-score normalization), and categorical variables are encoded with one-hot or ordinal encoding methods. Techniques like SMOTE are applied to address class imbalance where needed.
Model Development:
Multiple algorithms are used to build predictive models. Each model is trained with cross-validation, with nested grid search for optimal hyperparameter selection, and evaluated using standard metrics (accuracy, sensitivity, specificity, F1-score, and ROC AUC).
Interpretability:
SHAP analysis is performed to provide insights into the impact of each predictor variable on model outcomes.
Adaptability
The shared code is designed to be modular and adaptable. Researchers can use these notebooks as a blueprint to apply similar methodologies to their own studies, adapting the preprocessing steps and modeling strategies to fit different types of clinical or experimental data.

Installation

Clone the Repository:
git clone https://github.com/your_username/your_repository.git
cd your_repository
Set Up a Virtual Environment (optional but recommended):
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
Install the Dependencies:
pip install -r requirements.txt
Running the Code

Google Colab:
The notebooks are optimized for Google Colab. Open the desired notebook directly in Colab to run the code in an interactive environment.
Local Execution:
Run the notebooks using Jupyter Notebook or JupyterLab:
jupyter notebook
Navigate to the notebooks/ folder and open your preferred notebook.
Reproducibility and Adaptation

The notebooks include detailed comments and documentation outlining the workflow, ensuring that others can follow and adapt the code to their specific needs. While the code illustrates the analytical approach used in our study, it is meant primarily as a reference framework for methodological insights and potential adaptations.

Citation

If you find the code useful for your research, please cite the study as follows:

[It will be updated when published]
Please also reference the trial registration numbers:

CaCIA Trial: ClinicalTrials.gov (NCT03108586)
CARDEC 3 Trial: ClinicalTrials.gov (NCT03520309)

Acknowledgements

We thank the research teams and all study participants whose efforts contributed to the development of these analytical methods.

License

This project is not lincensed

Contact

For questions, suggestions, or collaboration inquiries, please open an issue on GitHub or contact vitordigmayer@gmail.com.
