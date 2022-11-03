# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 10:29:03 2022

@author: ronit
"""

# Contents of ~/my_app/streamlit_app.py
import streamlit as st


def TELECOM_CHURN():

    import numpy as np
    import pickle
    import streamlit as st
    from PIL import Image
    
    st.sidebar.markdown("Page 1")
    

    image1 = Image.open('D:/PROJECT TELECOMMUNICATION/FINAL TELECOM CHURN MODEL/Images/5b91ad6b.jpg')

    st.image(image1)
    
    st.sidebar.header("![Alt Text](https://media.tenor.com/images/82505a986c4b785e7ebdfe66d6bf0097/tenor.gif)")
    st.sidebar.header('User Input Parameters')

    #loading the model
    loaded_model = pickle.load(open('D:\PROJECT TELECOMMUNICATION/telecom_churn_trained_model.sav','rb'))
        

    def churn_prediction(input_data):
        
        input_data_1 = np.asarray(input_data)
        #st.write(input_data_1.shape)
        input_data_1_reshaped = input_data_1.reshape(1,-1)
        #st.write(input_data_1_reshaped.shape)
        
        #checking the prediction
        prediction_1 = loaded_model.predict(input_data_1_reshaped)
        
        if(prediction_1[0] == 0):
            return ('This person is not going to churn')
        else:
            return ('This person is going to Churn')
        
        
        
    #def main():
    st.title('Telecom Churn Prediction')



    account_length          = st.sidebar.number_input('Account length', min_value = 0)
    str_voice_mail_plan     = st.sidebar.radio('Voice Mail Plan', ['Yes','No'])

    if str_voice_mail_plan == 'Yes':
        voice_mail_plan     = 1
    else:
        voice_mail_plan     = 0
        
    #st.write(voice_mail_plan)

    voice_mail_messages     = st.sidebar.number_input('Voice Mail Messages',min_value=0)
    evening_minutes         = st.sidebar.number_input('Evening Minutes')
    night_minutes           = st.sidebar.number_input('Night Minutes')
    international_minutes   = st.sidebar.number_input('International Minutes')
    customer_service_calls  = st.sidebar.number_input('Customer Service Calls', min_value=0)

    str_international_plan      = st.sidebar.radio('International Plan', ['Yes','No'])

    if str_international_plan == 'Yes':
        international_plan     = 1
    else:
        international_plan     = 0

    #st.write(international_plan)
        
    day_calls               = st.sidebar.number_input('Day Calls', min_value=0)
    evening_calls           = st.sidebar.number_input('Evening Calls', min_value=0)
    night_calls             = st.sidebar.number_input('Night Calls', min_value=0)
    international_calls     = st.sidebar.number_input('International Calls', min_value=0)
    total_charge            = st.sidebar.number_input('Total Charge')

    # code for prediction
    churn_status = ''

    #creating submit button
    if st.button('Predict Churn Status'):
        churn_status= churn_prediction([account_length,voice_mail_plan,voice_mail_messages,evening_minutes,
                                        night_minutes,international_minutes,customer_service_calls,international_plan,
                                        day_calls,evening_calls,night_calls,international_calls,total_charge])


    st.success(churn_status)    
    

import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")
import seaborn as sns 


def VISUALISATION():
    st.title("DATA VISUALISATION ")
    st.sidebar.markdown("Page 2 ")
    st.sidebar.header("![Alt Text](https://media.tenor.com/images/977502b47439c190e8ca5475ba7b7822/tenor.gif)")
    st.title("![Alt Text](https://media1.tenor.com/images/18d1ad50a584f635b228241edd5f0ba0/tenor.gif?itemid=27008790)")
    st.set_option('deprecation.showPyplotGlobalUse', False)


    def main():
	    """Semi Automated ML App with Streamlit """

	    activities = ["EDA","Plots"]	
	    choice = st.sidebar.selectbox("Select Activities",activities)

	    if choice == 'EDA':
		    st.subheader("Exploratory Data Analysis")

		    data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		    if data is not None:
			     df = pd.read_csv(data)
			     st.dataframe(df.head())
			

			     if st.checkbox("Show Shape"):
				     st.write(df.shape)

			     if st.checkbox("Show Columns"):
				     all_columns = df.columns.to_list()
				     st.write(all_columns)

			     if st.checkbox("Summary"):
				     st.write(df.describe())

			     if st.checkbox("Show Selected Columns"):
				     selected_columns = st.multiselect("Select Columns",all_columns)
				     new_df = df[selected_columns]
				     st.dataframe(new_df)

			     if st.checkbox("Show Value Counts"):
				     st.write(df.iloc[:,-1].value_counts())

			     if st.checkbox("Correlation Plot(Matplotlib)"):
				     plt.matshow(df.corr())
				     st.pyplot()

			     if st.checkbox("Correlation Plot(Seaborn)"):
				     st.write(sns.heatmap(df.corr(),annot=True))
				     st.pyplot()


			     if st.checkbox("Pie Plot"):
				     all_columns = df.columns.to_list()
				     column_to_plot = st.selectbox("Select 1 Column",all_columns)
				     pie_plot = df[column_to_plot].value_counts().plot.pie(autopct="%1.1f%%")
				     st.write(pie_plot)
				     st.pyplot()



	    elif choice == 'Plots':
		    st.subheader("Data Visualization")
		    data = st.file_uploader("Upload a Dataset", type=["csv", "txt", "xlsx"])
		    if data is not None:
			    df = pd.read_csv(data)
			    st.dataframe(df.head())


			    if st.checkbox("Show Value Counts"):
				    st.write(df.iloc[:,-1].value_counts().plot(kind='bar'))
				    st.pyplot()
		
			    # Customizable Plot

			    all_columns_names = df.columns.tolist()
			    type_of_plot = st.selectbox("Select Type of Plot",["area","bar","line","hist","box","kde"])
			    selected_columns_names = st.multiselect("Select Columns To Plot",all_columns_names)

			    if st.button("Generate Plot"):
				    st.success("Generating Customizable Plot of {} for {}".format(type_of_plot,selected_columns_names))

				# Plot By Streamlit
				    if type_of_plot == 'area':
					    cust_data = df[selected_columns_names]
					    st.area_chart(cust_data)

				    elif type_of_plot == 'bar':
					    cust_data = df[selected_columns_names]
					    st.bar_chart(cust_data)

				    elif type_of_plot == 'line':
					    cust_data = df[selected_columns_names]
					    st.line_chart(cust_data)
                   
				# Custom Plot 
				    elif type_of_plot:
					    cust_plot= df[selected_columns_names].plot(kind=type_of_plot)
					    st.write(cust_plot)
					    st.pyplot()
  


    if __name__ == '__main__':
	    main()

from PIL import Image
def MODEL_EVALUATION():
    st.title("MODEL EVALUATION")
    st.sidebar.markdown("Page 3 ")
    st.sidebar.header("![Alt Text](https://media.tenor.com/images/82505a986c4b785e7ebdfe66d6bf0097/tenor.gif)")
    st.title("![Alt Text](https://media1.tenor.com/images/b56165c4e1024f208ce84f63fa41befe/tenor.gif?itemid=14794342)")
    
    
 
    def main():

     activities = ["ModelEvaluation","ModelOptimisation"]	
     choice = st.sidebar.selectbox("Select Activities",activities)

     if choice == 'ModelEvaluation':
         st.subheader("Model Evaluation")
         
         box1 = st.selectbox("Select Type of Model",["LR","NB","DT","RF","XGboost","SVM"])     
         
         if box1 == 'LR':
           st.subheader('Logistic Regression Result:')
           #create data
           Index = ['Model','Accuracy_tr','Accuracy_ts','F1-score_1']
           Values = ['Logistic Regression',0.86,0.85,0.17]
           List_LR = list(zip(Index,Values))
           
           df1 = pd.DataFrame(List_LR, columns=['Index', 'Values'])
           df1
           
         elif box1 == 'NB':
            st.subheader('Gaussian NB Result:')
            #create data
            Index = ['Model','Accuracy_tr','Accuracy_ts','F1-score_1']
            Values = ['Gaussian NB',0.86,0.85,0.00]
            List_NB = list(zip(Index,Values))
             
            df2 = pd.DataFrame(List_NB, columns=['Index', 'Values'])
            df2
            
         elif box1 == 'DT':
           st.subheader('Decision Tree Result:')
           #create data
           Index = ['Model','Accuracy_tr','Accuracy_ts','F1-score_1']
           Values = ['Decision Tree',0.87,0.87,0.15]
           List_DT = list(zip(Index,Values))
            
           df3 = pd.DataFrame(List_DT, columns=['Index', 'Values'])
           df3
           
         elif box1 == 'RF':
            st.subheader('Random Forest Result:')
            #create data
            Index = ['Model','Accuracy_tr','Accuracy_ts','F1-score_1']
            Values = ['Random Forest',0.86,0.85,0.41]
            List_RF = list(zip(Index,Values))
             
            df4 = pd.DataFrame(List_RF, columns=['Index', 'Values'])
            df4
            
         elif box1 == 'XGboost':
           st.subheader('XG Boosting Result:')
           #create data
           Index = ['Model','Accuracy_tr','Accuracy_ts','F1-score_1']
           Values = ['XG Boosting',1.00,0.96,0.85]
           List_XGB = list(zip(Index,Values))
            
           df5 = pd.DataFrame(List_XGB, columns=['Index', 'Values'])
           df5
           
         elif box1 == 'SVM':
           st.subheader('Support Vector Machine Result:')
           #create data
           Index = ['Model','Accuracy_tr','Accuracy_ts','F1-score_1']
           Values = ['Support Vector Machine',1.00,0.97,0.90]
           List_SVM = list(zip(Index,Values))
            
           df6 = pd.DataFrame(List_SVM, columns=['Index', 'Values'])
           df6

      
             
     elif choice == 'ModelOptimisation':
         st.subheader("Model Optimisation")
         box2 = st.selectbox("Select Type of Model",["SVM","RF","XGboost","Scoretable"])
         if box2 == 'SVM':
          st.subheader('GridSearchCV Support Vector Machine Result:')
          imageSVM = Image.open('D:/PROJECT TELECOMMUNICATION/FINAL TELECOM CHURN MODEL/Images/Screenshot (24).png')
          st.image(imageSVM)
          
         elif box2 == 'RF':
           st.subheader('GridSearchCV Random Forest Result:')
           imageRF = Image.open('D:/PROJECT TELECOMMUNICATION/FINAL TELECOM CHURN MODEL/Images/Screenshot (25).png')
           st.image(imageRF)
           
         elif box2 == 'XGboost':
           st.subheader('GridSearchCV XG Boosting Result:')
           imageXGB = Image.open('D:/PROJECT TELECOMMUNICATION/FINAL TELECOM CHURN MODEL/Images/Screenshot (26).png')
           st.image(imageXGB)
           
         elif box2 == 'Scoretable':
           st.subheader('GridSearchCV Score Table Result:')
           imageST = Image.open('D:/PROJECT TELECOMMUNICATION/FINAL TELECOM CHURN MODEL/Images/Screenshot (27).png')
           st.image(imageST)
   

    if __name__ == '__main__':
     main()

              
    


page_names_to_funcs = {
    "Telecom churn": TELECOM_CHURN,
    "Data Visualisation": VISUALISATION,
    "Model Evaluation": MODEL_EVALUATION,
}

selected_page = st.sidebar.selectbox("Select a page", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()