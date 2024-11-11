from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load and preprocess data
data = pd.read_csv('Lotwize Data with New Features (1).csv')

# Preprocessing
numerical_columns = ['bathrooms', 'bedrooms', 'livingArea', 'lotSize', 'yearBuilt', 
                    'NearestSchoolDistance', 'NearestHospitalDistance', 'ShopsIn2Miles']
categorical_columns = ['city', 'region', 'homeType']

# Imputation
for col in numerical_columns:
    data[col].fillna(data[col].median(), inplace=True)
for col in categorical_columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

# Handle outliers
Q1 = data[numerical_columns].quantile(0.25)
Q3 = data[numerical_columns].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

for col in numerical_columns:
    data[col] = data[col].clip(lower=lower_bound[col], upper=upper_bound[col])

# Encode categorical features
data_encoded = pd.get_dummies(data.drop(columns=['monthSold', 'city']), 
                            columns=categorical_columns[1:], 
                            drop_first=True)

# Define target and features
X = data_encoded.drop(columns=['price'])
y = data_encoded['price']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Define the UI
app_ui = ui.page_fluid(
    ui.tags.style("""
        /* Global styles */
        body {
            background-color: #f5f7fa;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        .container-fluid {
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
        }
        .well {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
            margin-bottom: 20px;
        }
        .form-control {
            border-radius: 6px;
            border: 1px solid #ddd;
            padding: 8px 12px;
            margin-bottom: 15px;
            width: 100%;
        }
        .form-control:focus {
            border-color: #2c3e50;
            box-shadow: 0 0 0 0.2rem rgba(44, 62, 80, 0.25);
        }
        .btn-primary {
            background-color: #2c3e50;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            transition: all 0.3s;
            width: 100%;
            font-size: 16px;
            font-weight: 500;
        }
        .btn-primary:hover {
            background-color: #34495e;
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-weight: 600;
            font-size: 24px;
        }
        h4 {
            color: #34495e;
            margin-top: 25px;
            margin-bottom: 15px;
            font-weight: 500;
            font-size: 18px;
        }
        .prediction-panel {
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            min-height: 200px;
            text-align: center;
        }
        .input-label {
            font-weight: 500;
            color: #445566;
            margin-bottom: 5px;
        }
        .app-header {
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 10px;
            text-align: center;
        }
        .prediction-result {
            font-size: 24px;
            font-weight: 600;
            color: #2c3e50;
            margin: 20px 0;
        }
        .prediction-details {
            color: #666;
            font-size: 16px;
            margin-top: 10px;
        }
        .feature-icon {
            margin-right: 8px;
            color: #2c3e50;
        }
    """),
    
    ui.div(
        {"class": "app-header"},
        ui.h2("🏠 Real Estate Price Predictor", style="color: white; margin: 0;"),
        ui.p("Enter house details below to get an estimated price", style="margin: 10px 0 0 0;")
    ),
    
    ui.row(
        ui.column(4,
            ui.div(
                {"class": "well"},
                ui.h2("House Features"),
                
                ui.h4("📊 Basic Information"),
                ui.input_numeric("bedrooms", "Number of Bedrooms", value=3, min=1, max=10),
                ui.input_numeric("bathrooms", "Number of Bathrooms", value=2, min=1, max=10),
                ui.input_numeric("livingArea", "Living Area (sq ft)", value=2000, min=500, max=10000),
                ui.input_numeric("lotSize", "Lot Size (sq ft)", value=5000, min=1000, max=50000),
                
                ui.h4("🏠 Property Details"),
                ui.input_numeric("yearBuilt", "Year Built", value=2000, min=1900, max=2024),
                ui.input_select("region", "Region", choices=list(data['region'].unique())),
                ui.input_select("homeType", "Home Type", choices=list(data['homeType'].unique())),
                
                ui.h4("📍 Location Features"),
                ui.input_numeric("NearestSchoolDistance", "Distance to School (miles)", 
                               value=1, min=0, max=10),
                ui.input_numeric("NearestHospitalDistance", "Distance to Hospital (miles)", 
                               value=2, min=0, max=20),
                ui.input_numeric("ShopsIn2Miles", "Shops within 2 Miles", 
                               value=10, min=0, max=100),
                
                ui.div(
                    {"style": "margin-top: 30px;"},
                    ui.input_action_button(
                        "predict", "🔍 Calculate Price Prediction",
                        class_="btn-primary"
                    )
                )
            )
        ),
        ui.column(8,
            ui.div(
                {"class": "prediction-panel"},
                ui.h2("Prediction Results"),
                ui.div(
                    {"class": "prediction-result"},
                    ui.output_text("prediction")
                ),
                ui.div(
                    {"class": "prediction-details"},
                    ui.output_text("prediction_details")
                )
            )
        )
    )
)

def server(input, output, session):
    # Store model prediction and data for reactive use
    prediction_store = reactive.Value({'price': None, 'error': None})
    
    @reactive.Effect
    @reactive.event(input.predict)
    def predict():
        try:
            # Prepare input data
            input_data = {
                'bedrooms': input.bedrooms(),
                'bathrooms': input.bathrooms(),
                'livingArea': input.livingArea(),
                'lotSize': input.lotSize(),
                'yearBuilt': input.yearBuilt(),
                'NearestSchoolDistance': input.NearestSchoolDistance(),
                'NearestHospitalDistance': input.NearestHospitalDistance(),
                'ShopsIn2Miles': input.ShopsIn2Miles(),
            }
            
            # Add dummy variables
            for region in X_train.filter(regex='^region_').columns:
                region_name = region.replace('region_', '')
                input_data[region] = 1 if input.region() == region_name else 0
                
            for home_type in X_train.filter(regex='^homeType_').columns:
                type_name = home_type.replace('homeType_', '')
                input_data[home_type] = 1 if input.homeType() == type_name else 0
            
            # Create DataFrame with aligned features
            input_df = pd.DataFrame([{col: input_data.get(col, 0) for col in X_train.columns}])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_store.set({'price': prediction, 'error': None})
            
        except Exception as e:
            prediction_store.set({'price': None, 'error': str(e)})
    
    @output
    @render.text
    def prediction():
        pred = prediction_store.get()
        if pred['error']:
            return f"❌ Error: {pred['error']}"
        elif pred['price'] is not None:
            return f"💰 Estimated Price: ${pred['price']:,.2f}"
        return "👋 Enter house details and click 'Calculate Price Prediction'"
    
    @output
    @render.text
    def prediction_details():
        pred = prediction_store.get()
        if pred['price'] is not None:
            return """Based on current market conditions and comparable properties
                     Prediction confidence may vary based on market volatility"""
        return ""

app = App(app_ui, server)

if __name__ == "__main__":
    app.run(launch_browser=True)



