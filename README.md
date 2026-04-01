This repository contains the complete workflow and datasets used in the project titled "Uncovering prognostic factors in esophageal cancer through machine learning and time-series analysis."

⚠️ Note: The files are prefixed with numbers to indicate the order of analysis. Please remove these prefixes before running the code to avoid potential path or module issues.

🔍 Project Overview
This project uses data from the SEER database to identify prognostic factors influencing 5-year survival in esophageal cancer (EC). It combines traditional statistical techniques (Cox regression, Kaplan-Meier) with machine learning (ML) and time-series approaches for robust and interpretable analysis.

📁 Repository Contents
File	Description

0_EC_data.csv:	Raw dataset extracted from the SEER database based on predefined inclusion/exclusion criteria.

1_Preprocess_EC.Rmd:	R Markdown script for preprocessing the SEER dataset, including filtering, recoding, and formatting.

2_EC_df.csv:	Cleaned and preprocessed dataset output from the R preprocessing step.

3_Feature_select_classification_MinMax.py:	Python script for selecting relevant features for ML classification models using Min-Max normalization.

4_EC_df_median.csv:	Dataset containing selected features for classification model input.
(Note: 4_EC_df_median-revised.csv is an updated CSV with normalization after splitting)

5(0)_ML_comparison.py:	Script to train and compare multiple ML algorithms on the dataset. Detailed evaluations for each model are in files 5(A) to 5(E).

6_Feaure_select_classification_Series.py:	Python script for selecting variables tailored for Cox regression, Kaplan-Meier, and time-series analysis.

7_EC_df_Series.csv:	Final dataset of selected variables for survival and time-series modeling.

8_Cox_and_KM.py:	Python script performing Cox proportional hazards regression and Kaplan-Meier survival analysis.

9_Time-series_analysis.py:	Python script for time-series trend analysis of selected variables over time.


🧪 Tools and Technologies

R / RMarkdown: Data preprocessing

Python: Feature selection, ML modeling, survival analysis, and time-series analysis

Libraries: pandas, numpy, scikit-learn, lifelines, matplotlib, seaborn


📌 Usage Instructions
Clone the repository.

Remove file prefixes (e.g., 3_, 5(0)_) before running any scripts.

Ensure you have the necessary packages installed.

Follow the numbered order of files for the complete analysis pipeline.
