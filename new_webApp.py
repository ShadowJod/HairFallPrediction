# Import necessary libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open("trained_modelNew.sav", "rb"))



# Streamlit app layout
def main():
    # Title of the app
    page = st.sidebar.radio("Navigation",["Predict","Precaution"])

    if page == "Home":
        st.title("Home")
        st.title("Hair Loss Prediction App")

    binary_mapping = {"Yes": 1, "No": 0}
    stress_mapping = {"Low":  0.486239, "Moderate": 0.518519, "High":  0.485981}
    medical_conditions_mapping = {
        "No Data":    0.427273, "Alopecia Areata": 0.570093, "Psoriasis": 0.500000,"Dermatosis": 0.488636 ,"Thyroid Problems": 0.434343,
        "Androgenetic Alopecia": 0.561224, "Dermatitis":  0.478261, "Seborrheic Dermatitis":  0.568182,
        "Scalp Infection":   0.481013, "Eczema": 0.478261, "Ringworm": 0.478261
    }
    medication_treatment_mapping = {
        "No Data":  0.500000, "Rogaine": 0.508621, "Antidepressants": 0.481818, "Steroids": 0.551402, 
        "Heart Medication":  0.509615, "Accutane": 0.490196, "Antibiotics": 0.531915, "Antifungal Cream": 0.468085,
        "Chemotherapy":  0.511111, "Blood Pressure Medication":  0.466667, "Immunomodulators": 0.444444
    }
    nutritional_deficiencies_mapping = {
        'No Data': 0.525000, 'Biotin Deficiency': 0.464646, 'Iron deficiency': 0.512821, 'Magnesium deficiency': 0.547619,
        'Omega-3 fatty acids': 0.456522, 'Protein deficiency': 0.522222, 'Selenium deficiency': 0.512195,
        'Vitamin A Deficiency': 0.515152, 'Vitamin D Deficiency':  0.500000, 'Vitamin E deficiency':  0.457831,
        'Zinc Deficiency':  0.472222
    }

    # User inputs for features
    Genetics =binary_mapping[st.selectbox("Genetics",["Yes","No"])]
    HormonalChanges =binary_mapping[st.selectbox("Hormonal Changes", ["Yes", "No"])]
    MedicalConditions =medical_conditions_mapping[st.selectbox("Medical Condition", list(medical_conditions_mapping.keys()))]
    MedicationAndTreatments = medication_treatment_mapping[st.selectbox("Medication and Treatments", list(medication_treatment_mapping.keys()))]
    NutritionalDeficiencies =nutritional_deficiencies_mapping[st.selectbox("Nutritional Deficiencies", list(nutritional_deficiencies_mapping.keys()))]
    Stress = stress_mapping[st.selectbox("Stress",["Low", "Moderate", "High"])]
    Age = st.slider("Age",0,100,25)
    PoorHairCareHabits = binary_mapping[st.selectbox("PoorHairCareHabits",["Yes","No"])]
    EnvironmentalFactors = binary_mapping[st.selectbox("EnvironmentalFactors",["Yes","No"])]
    Smoking = binary_mapping[st.selectbox("Smoking",["Yes","No"])]
    WeightLoss = binary_mapping[st.selectbox("WeightLoss",["Yes","No"])]


    input_data = [
        Genetics, HormonalChanges, MedicalConditions, MedicationAndTreatments,
        NutritionalDeficiencies, Stress, Age, PoorHairCareHabits, EnvironmentalFactors,
        Smoking, WeightLoss
    ]
    
    input_data_array = np.asarray(input_data, dtype=float).reshape(1, -1)

    # print("Data types of input data array elements:", input_data_array.dtype)

     

    if st.button("Predict Hair Loss"):
        prediction = loaded_model.predict(input_data_array)



        if prediction[0] == 0:
            st.success("Prediction: No Hair Loss")
        else:
            st.success("Prediction: Has Hair Loss")
           

            



if __name__ == "__main__":
    main()
