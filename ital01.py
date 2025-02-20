# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:12:10 2023

@author: HP
"""

import tensorflow as tf
import streamlit as st
import pandas as pd
import numpy as np
#import sklearn
#import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from collections import Counter
from PIL import Image, ImageOps
import cv2
import os
#%matplotlib inline
#sns.set(color_codes=True)

def main():
    
    def load_data():
        data_clean=pd.read_csv('baseP.csv', sep=',', encoding="latin-1")
        return data_clean  
    st.markdown('''## :blue-background[:blue[ITaL 0.1:]]  :blue[I]dentificateur de :blue[Ta]urin :blue[L]obi''')
    st.markdown(''' Version 0.1 :flag-bf:''')
    st.markdown("##### *Cette application permet de déterminer si un bovin est un :red[Taurin Lobi] pur à partir d'une image ou de ses mesures morphologiques*"
             )
    st.markdown(":grey-background[Auteur:] **Bembamba Fulbert**")
    
    #@st.cache_data     #(persist=True)
   
    # Modélisation
        
    #st.sidebar.markdown(":streamlit:")
    st.sidebar.subheader("Que voulez-vous faire?")
    tache = st.sidebar.selectbox(
        "Tâche à effectuer",
        ("Prédiction par l'image", "Prédiction par les mesures morpho") # (Open Source) la visualisation nécessite des privilèges
    )
    if tache == "Prédiction par l'image":
        file = st.sidebar.file_uploader("Chargez une image de bovin à prédire", type=["jpg", "png"])
        
        if file == None:
            st.sidebar.warning("Veuillez télécharger une image de bovin.")
        else:
            image = Image.open(file)
            st.image(image,                      
                     #use_column_width=True
                     width=400
                     )
            
            def import_img(image_data):
                size = (256, 256)
                image = ImageOps.fit(image_data, size)  # 🔧 Suppression de `Image.ANTIALIAS` (obsolète)
                image = np.array(image)  # Conversion en NumPy array

                if image.shape[-1] == 4:  # 🔧 Certains PNG ont un canal alpha (RGBA)
                    image = image[:, :, :3]

                img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # OpenCV attend un format BGR
                img_reshape = img[np.newaxis, ...] / 255.0  # Normalisation
                return img_reshape
            
            img = import_img(image)
            
            from tensorflow.keras.models import load_model
            dir_savedModel = 'C:\\Users\\DELL\\Documents\\THESE\\Manuscript these\\Codes Manuscrit\\Scripts_manuscrit'
            with st.spinner('Chargement du modèle de prédiction de vue'):
                view_predictor = load_model(os.path.join(dir_savedModel, 'taurinView_classif.h5'))
            view_pred = view_predictor.predict(img)
            score = tf.nn.softmax(view_pred[0])
            view_classes=['tête', 'arrière', 'profil']
            view = view_classes[np.argmax(score)]
            
            def load_predictor(vue):
                match(vue):
                    case 'tête':
                        with st.spinner('Chargement du modèle de prédiction de tête'):
                            model=load_model(os.path.join(dir_savedModel, 'taurinCNN_head.h5'))
                    case 'arrière':
                        with st.spinner('Chargement du modèle de prédiction arrière'):
                            model=load_model(os.path.join(dir_savedModel, 'taurinCNN_tail.h5'))   
                    case 'profil':
                        with st.spinner('Chargement du modèle de prédiction de profil'):
                            model=load_model(os.path.join(dir_savedModel, 'taurinCNN_side.h5'))
                return model
            
            predictor = load_predictor(view)
            predictions = predictor.predict(img)
            prediction = "Lobi" if predictions > 0.5 else "NoLobi"
            if st.sidebar.button("PREDIRE"):
                st.sidebar.write(f" Cette image vue de {view} est un **{prediction}**")
            
    if tache =="Prédiction par les mesures morpho":
        #st.sidebar.subheader("Mode d'entrée des données")
        
        mode = st.sidebar.selectbox(
            "Mode d'entrée des données",
            ("saisie manuelle", "fichier csv") # (Open Source) la visualisation nécessite des privilèges
        )
        st.sidebar.subheader("Caractéristiques du bovin à prédire")
        if mode=="fichier csv":
            uploaded_file = st.sidebar.file_uploader("Télécharger vos input", type=['csv'])
            if uploaded_file is not None:
                input_df = pd.read_csv(uploaded_file)
            else:
                st.sidebar.warning("Veuillez télécharger un fichier CSV.")
                input_df = pd.DataFrame()  # Initialiser input_df pour éviter l'erreur
        else:
            col1, col2 = st.sidebar.columns(2) 
            
       
            def user_input():
                
                sacrum_ht = col1.number_input(
                    "sacrum height",
                    30.0, 140.0, 70.0, step=8.0
                )
                
                horn_l = col2.number_input(
                    "horn length",
                     0.0, 60.0, 30.0, step=5.0
                )    
                withers_ht = col1.number_input(
                    "withers height",
                    15.0, 170.0, 80.0, step=10.0
                     
                )
                muzl_circumf = col2.number_input(
                    "muzzle circumference",
                    3.0, 50.0, 30.0, step=5.0
                )
                body_l = col2.number_input(
                    "body length",
                    30.0, 110.0, 50.0, step=10.0
                )
                tail_l = col1.number_input(
                    "tail length",
                    0.0, 140.0, 70.0, step=2.0
                )
                dist_base_h = col2.number_input(
                    "distance base horn",
                    9.0, 70.0, 40.0, step=5.0
                )
                head_l = col1.number_input(
                    "head length",
                    30.0, 55.0, 40.0, step=3.0
                )
                age = col2.number_input(
                    "age",
                    0.0, 17.0, 10.0, step=2.0
                )
                
                hump_pos = col1.radio(
                    "hump position",
                    ('Absent-hump', 'Cervicothoracic')
                )
                sex = col2.radio(
                    "Sexe",
                    ('Male', 'Female')
                )
                st_mullet = col1.radio(
                    "stripe mullet",
                    ('Absent-stripe', 'Inverse', 'Dark')
                )
                data={'sacrum height':sacrum_ht,
                      'hump position':hump_pos,
                      'horn length':horn_l,
                      'withers height':withers_ht,
                      'muzzle circumference':muzl_circumf,
                      'stripe mullet':st_mullet,
                      'body length':body_l,
                      'tail length':tail_l,
                      'distance base horn':dist_base_h,
                      'head length':head_l,
                      'sex':sex,
                      'age':age
                    }
                features=pd.DataFrame(data, index=[0])
                return features
            
            input_df = user_input()
            
        modele=st.sidebar.selectbox("Sélectionnez un modèle",
                                     ("prédiction combinée", "RandomForest", "SVM", "KNN", "LogisticRegression", "XGBoost")
        )    
        
        baseP=load_data()
        taurins=baseP.drop(columns=['genotype'])
        df=pd.concat([input_df, taurins], axis=0)
          
        def encodage(df, columns):
            code = {
                'Male': 1, 'Female': 0,
                'Absent-hump': 0, 'Cervicothoracic': 1,
                'Absent-stripe': 0, 'Inverse': 1, 'Dark': 2
                }
            for col in columns:
                df[col] = df[col].map(code)
            return df
        
        columns_to_encode = ['sex', 'stripe mullet', 'hump position']              
        df = encodage(df, columns_to_encode)
        
        
        # for col in encode:
        #     df[col] = df[col].apply(col_encode)
        
        # df['sex'] = df['sex'].apply(col_encode)
        # df['hump position'] = df['hump position'].apply(col_encode)
        
        #SCALING
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
        df=df[:1]
        df=df.drop(columns=['Unnamed: 0'])
        
        # for col in encode
        #     print(f"Unique values in '{col}': {df[col].unique()}")
        
        st.write('Les paramètres du bovin à prédire')
        st.write(input_df)
        classe={'Taurin lobi':1,
                'Crossbread':0
            }
        # labels=pd.DataFrame(classe, index=[0])
        # st.write("le codage des classes")
        # st.write(labels)    
        
        
        def load_predictor(modele):
            match(modele):
                case 'SVM':
                    with open("svmp-default_saved.pkl", "rb") as f:
                        model = pickle.load(f)
                case 'RandomForest':
                    with open("rfp-default_saved.pkl", "rb") as f:
                        model = pickle.load(f)
                case 'LogisticRegression':
                    with open("lrp-default_saved.pkl", "rb") as f:
                        model = pickle.load(f)
                case 'KNN':
                    with open("knnp-default_saved.pkl", "rb") as f:
                        model = pickle.load(f)
                case 'XGBoost':
                    with open("xgbp-default_saved.pkl", "rb") as f:
                        model = pickle.load(f)
            return model
        if modele == 'prédiction combinée':
            def classify_candidate(features):
                svm_model=load_predictor('SVM')
                rf_model=load_predictor('RandomForest')
                lr_model=load_predictor('LogisticRegression')
                knn_model=load_predictor('KNN')
                xgb_model=load_predictor('XGBoost')
                
                xgb_pred=xgb_model.predict(features)
                if xgb_pred == 1:
                    return 1
                else:
                    preds = [
                        xgb_pred,
                        svm_model.predict(features),
                        rf_model.predict(features),
                        lr_model.predict(features),
                        knn_model.predict(features)
                    ]
                    majority_vote = Counter(preds).most_common(1)[0][0]
                    return 1 if majority_vote == 1 else 0
                
            prediction = classify_candidate(df)
        else:
            saved_model=load_predictor(modele)
            prediction=saved_model.predict(df)
        #prediction_proba=saved_model.predict_proba(df)
        
        
        
                
    # Saisir les caractéristiques de l'individu à prédire
 
        if st.sidebar.button("PREDIRE"):
            st.write(f"##### Les résultats du modèle {modele}")
                        # Prédiction
            #st.write(prediction)
            
            
            
            if (prediction==0):
                st.markdown("""Cet individu est un :blue[**CROSSBREAD**]:taurus:""") 
            else:
                st.markdown("""Cet individu est un :red[**TAURIN**] :cow:""") 
                
                
                
                # Métriques de performance
                #precision=precision_score(y_test, y_pred)
                #recall=recall_score(y_test, y_pred)
                
                # Afficher les métriques dans l'application
            
    # Fonctions de traçage des graphiques


if __name__=='__main__':
    main()
    
#"""""""""""""""""""""""""""""""    
#serie tuto interessant sur streamlit
    #https://www.youtube.com/watch?v=8M20LyCZDOY&list=PLtqF5YXg7GLmCvTswG32NqQypOuYkPRUE&index=2
    #https://www.youtube.com/watch?v=D0D4Pa22iG0
# emojii shortcut
    #https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/   

# Readme:
        #Pour lancer l'appli:
    #     cmd
    #     cd C:\Users\HP\Documents\THESE\Papers\Paper3\paper3_project
    #     python -m streamlit run AppTaurinV0.py
    
#"""""""""""""""""""""""""""""""""""""""