from flask import Flask,render_template, request, jsonify,redirect,url_for
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import io
import base64
app = Flask(__name__)

#the aim is to ,understand the flow of data, give it a direction,then take contributions
data=pd.read_csv(r'C:\Users\afsaa\OneDrive\Documents\RAIN\Python Class\project\cleaned_data.csv')
model = joblib.load(r'C:\Users\afsaa\OneDrive\Documents\RAIN\Python Class\project\model.pkl')
form=joblib.load(r'C:\Users\afsaa\OneDrive\Documents\RAIN\Python Class\project\form.pkl')
label_encoders = joblib.load(r'C:\Users\afsaa\OneDrive\Documents\RAIN\Python Class\project\label_encoders.pkl')
# # #Load the trained model, scaler, and label encoder
# label_encoder = joblib.load('label.pkl')  # Replace with your label encoder


    #logistics forms the basis on which communication works


def regenerate_dependent_jsons():
    try:
        # Load updated data
        data = pd.read_json("data.json", orient="records")

        # 1. Update logistics_df.json
        grouped = (
            data.groupby(['Fruit_x', 'Form', 'RetailPriceUnit_x'])
            .mean(numeric_only=True)
            .reset_index()
        )
        if not grouped.empty:
            grouped.to_json("logistics_df.json", orient="records", indent=2)

        # 2. Update correlation.json
        numeric_cols = data.select_dtypes(include='number')
        if not numeric_cols.empty:
            correlation = numeric_cols.corr()
            correlation.to_json("correlation.json", orient="split", indent=2)

        print("✔ Dependent JSONs updated.")

    except Exception as e:
        print(f"⚠ Error updating dependent JSONs: {e}")
data.to_json("data.json", orient="records", indent=2)
regenerate_dependent_jsons()
@app.route('/predict', methods=['GET','POST']) 
def predict():
    data = pd.read_json("data.json", orient="records")
    correlation = pd.read_json("correlation.json", orient="split")
    correlation = correlation.fillna(0)
    data = data.drop('Unnamed: 0', axis=1)

    Fruit = data['Fruit_x'].dropna().unique().tolist()
    Form = data['Form'].dropna().unique().tolist()
    amounts = data['CupEquivalentSize_x'].dropna().unique().tolist()
    sizes = data['RetailPriceUnit_y'].dropna().unique().tolist()

    prediction = None  # default
    error_message = None
    prediction_value=None

    # Query the help based on input
    if request.method == 'POST':
       
        form1 = int(request.form.get('Form'))
        form2 = int(request.form.get('Fruit_x'))
        form3 = float(request.form.get('CupEquivalentSize_x'))
        form4 = int(request.form.get('RetailPriceUnit_y'))
        # Matches the value stored in forms
        
        contribute = data[
            (data['Form'] == form1) &
            (data['Fruit_x'] == form2) &
            (data['CupEquivalentSize_x'] == form3) &
            (data['RetailPriceUnit_y'] == form4)
            ]
   
# Find relation between correlation and data ,use it to build contributions
       
        if not contribute.empty:
               #check the measure
            print(contribute.corr())
 
            print("Form inputs:", form1, form2, form3, form4)
            # Step 2: Filter the main dataset
            factor = contribute[
            (contribute['Fruit_x'].isin(correlation['Fruit_x'])) &
            (contribute['Form'].isin(correlation['Form'])) &
            (contribute['CupEquivalentSize_x'].isin(correlation['CupEquivalentSize_x'])) &
            (contribute['RetailPriceUnit_y'].isin(correlation['RetailPriceUnit_y']))
        ]
        #check factor where criteria is still true 
            contribution = factor.value_counts()
            
            if not contribution.empty:#understand the result
                print(contribution.describe() )

                # Make predictions
            try:
                X = contribute.drop(['RetailPrice_x'], axis=1)
                prediction = model.predict(X)
                prediction_value = round(prediction[0], 2) if isinstance(prediction, np.ndarray) else prediction
                data['predicted_price'] = prediction_value
                #prediction_value = contribution['predicted_price'].tolist()

            except Exception as e:
                error_message = f"Prediction Error: {str(e)}"
                print("Prediction Error:", e)
        else:
            print("Contribute empty?", contribute.empty)
        print(prediction_value)
        print(contribute.columns)
    return render_template("pro1.html",Form=Form,Fruit=Fruit,amounts=amounts, sizes=sizes,prediction=prediction_value,error_message=error_message)

# query the form then pass it through the promotion, allow them to make changes using html,leading to prediction           
@app.route('/investment',methods=['GET','POST']) #understand the flow of data
def helps():
    data=pd.read_json("data.json", orient="records")
    Fruit = data['Fruit_x'].dropna().unique().tolist()
    Form = data['Form'].dropna().unique().tolist()
    RetailPriceUnit = data['RetailPriceUnit_y'].dropna().unique().tolist()
    if request.method == 'POST': #query the features
        Fruit = request.form.get('Fruit_x')
        RetailPriceUnit = request.form.get('RetailPriceUnit_y')
        Form = request.form.get('Form')
        investment=data[
                (data['Fruit_x'] == Fruit) &
                (data['Form'] == Form) &
                (data['RetailPriceUnit_y'] == RetailPriceUnit) 
                ] 
        if not investment.empty:#create a relationship
            filtered_investment=investment.describe()
            print(filtered_investment.corr())
        #relationship help decide logistics shape and direction
            if 'mean' in filtered_investment.index:
                logistics=filtered_investment.loc['mean'].sort_values(ascending=False)
                logistics_df = logistics.reset_index()
                #logistics_df.columns = ['Fruit_x', 'Form','RetailPriceUnit_x']
                
    #check correlation of logistics
                if not logistics_df.empty:
                    print(logistics_df.corr())      
                    logistics_df.to_json("logistics_df.json", orient="records", indent=2)
                    print(logistics_df.columns)
    #logistics forms the basis on which communication works
        return redirect(url_for('helpfilter'))
    return render_template('logistics.html',Fruit=Fruit,Form=Form,RetailPriceUnit=RetailPriceUnit)

@app.route('/communication',methods=['GET','POST']) #store data,
def helpfilter():
    logistics_df=pd.read_json("logistics_df.json", orient="records") # save to json
    Fruit = logistics_df['Fruit_x'].dropna().unique().tolist()
    Form = logistics_df['Form'].dropna().unique().tolist()
    RetailPriceUnit = logistics_df['RetailPriceUnit_y'].dropna().unique().tolist()

   # data = pd.read_json("data.json", orient="records") 
    if request.method == 'POST':
# # Get values from the HTML form
        Fruit = request.form.get('Fruit_x')
        RetailPriceUnit = request.form.get('RetailPriceUnit_y')
        Form = request.form.get('Form')
    # Query the dataset using form inputs
        filtered = logistics_df[
            (logistics_df['Fruit_x'] == Fruit) &
            (logistics_df['RetailPriceUnit_y'] == RetailPriceUnit) &
            (logistics_df['Form'] == Form)
        ]
       
    #find correlations
        if not filtered.empty:
            #query correlation to define new basis determine the most relevant features
            correlation = filtered.select_dtypes(include='number').corr()
        else:
            correlation = logistics_df.select_dtypes(include='number').corr()
            #filtered_top = correlation[numeric_cols]
            print("Filtered_Top:",correlation)
        #new basis is the start of prediction
            if not correlation.empty:
                correlation.to_json("correlation.json", orient="split") # save to json
            else:
                print("Correlation matrix is empty. No numeric data or only one numeric column.")
        return redirect(url_for('predict')) 
        #return render_template('comunication.html')
    return render_template('comunication.html',Fruit=Fruit,Form=Form,RetailPriceUnit=RetailPriceUnit)
            # Convert to numpy array for prediction
 #  data['RetailPriceUnit'] = list(zip(data['RetailPriceUnit_x'], data['RetailPriceUnit_y']))
# data['Fruit'] = list(zip(data['Fruit_x'], data['Fruit_y']))
# data['RetailPrice'] = list(zip(data['RetailPrice_x'], data['RetailPrice_y']))
# data['CupEquivalentSize'] = list(zip(data['CupEquivalentSize_x'], data['CupEquivalentSize_y']))
# data['CupEquivalentPrice'] = list(zip(data['CupEquivalentPrice_x'], data['CupEquivalentPrice_y']))
# data['CupEquivalentUnit'] = list(zip(data['CupEquivalentUnit_x'], data['CupEquivalentUnit_y']))

if __name__ == '__main__':
    app.run(debug=True)
   # python app.py
   
    # def stimulation(data):
    #     """Simulate economic stimulus by adjusting key features."""
    #     if 'Yield' in data.columns and not data['Yield'].isnull().all():
    #         data['Yield'] = data['Yield'].astype(float) * 1.10  # Simulate 10% yield increase
    #     if 'Form' in data.columns and not data['Form'].isnull().all():
    #         data['Form'] = data['Form'].astype(float) * 0.9  # Simulate promo effect
    #     return data


     # else:
        #     contribute = pd.DataFrame([{
        #     'Form': form1,
        #     'Fruit': form2,
        #     'CupEquivalentUnit': form3,
        #     'RetailPriceUnit': form4
        #     'Yield': data['Yield'].dropna().mode().iloc[0] if 'Yield' in data.columns else 1,
        #     'Form': data['Form'].dropna().mean() if 'Form' in data.columns else 1,
        #     'RetailPrice': 0
        #     }]) 
            #contribute = stimulation(contribute)