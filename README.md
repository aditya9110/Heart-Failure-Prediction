# Heart Failure Prediction

A. Predicting the Likelihood of Death of the Patient

The prediction of the probability of the death will be based on the different features mentioned above. Firstly, we will 
clean the data and add feature engineering such as label encoding, temporal substitution, sampling, and feature 
selection based on their importance. Later the pre-processed data is passed on to different models for training. Based 
on their result and evaluating based on the accuracy and recall values the best model will be finalized.

B. Survival Probability of the Patient for ‘n’ period (time)

The second solution we are presenting is about the survival chances of a patient on the nth observed day. Within the 
dataset we have a feature described as ‘time’ which states the follow-up period of the patient. The time feature varies 
from 4(min) to 275(max). Thus, the survival model created based on a few important features will provide the hazard 
score and survival score for the patients.

Hazard score indicates the risk of death due to heart failure. Higher the hazard score, more the risk of getting a heart
complication. Another parameter we mentioned is survival score. Survival Function calculates survival probabilities of 
the patient over the course of time. The value of survival score lies between 0 to 1. These results help the doctors to
prioritize the patients and make improvements in their medicines so as to increase the survival rate of the patients.

## Code Execution
1. Make sure you have Python 3.8 installed on the system.
2. Open cmd where the contents are unzipped.
3. Run the below command to install required packages.
> pip install -r requirements.txt
4. Once all packages are installed, run the following command:
> streamlit run app.py

If facing any issues, [contact me](mailto:mahimkaradi@gmail.com)
