from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend connection

API_KEY = "4be818742436d1138ad21ddc4b9b5270"

# Function to get weather data
def get_weather(city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"q": city, "appid": API_KEY, "units": "metric"}
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if response.status_code == 200:
            return {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"]
            }
        else:
            return {"error": data.get("message", "Invalid city name.")}
    except:
        return {"error": "No internet connection."}

# AI Weather Prediction Model
class WeatherPredictionSystem:
    def __init__(self):
        self.model = BayesianNetwork([
            ('CloudCover', 'RainChance'),
            ('Humidity', 'RainChance'),
            ('Temperature', 'Sunlight')
        ])

        # CPDs (Conditional Probability Distributions)
        cpd_cloud_cover = TabularCPD(variable='CloudCover', variable_card=2, values=[[0.6], [0.4]])
        cpd_humidity = TabularCPD(variable='Humidity', variable_card=2, values=[[0.7], [0.3]])
        cpd_temperature = TabularCPD(variable='Temperature', variable_card=2, values=[[0.5], [0.5]])

        cpd_rain = TabularCPD(variable='RainChance', variable_card=2,
                              values=[[0.8, 0.6, 0.7, 0.4],  
                                      [0.2, 0.4, 0.3, 0.6]],
                              evidence=['CloudCover', 'Humidity'],
                              evidence_card=[2, 2])

        cpd_sunlight = TabularCPD(variable='Sunlight', variable_card=2,
                                  values=[[0.9, 0.4],
                                          [0.1, 0.6]],
                                  evidence=['Temperature'],
                                  evidence_card=[2])

        self.model.add_cpds(cpd_cloud_cover, cpd_humidity, cpd_temperature, cpd_rain, cpd_sunlight)
        self.inference = VariableElimination(self.model)

    def predict_weather(self, user_input):
        evidence = {
            'CloudCover': int(user_input['cloud_cover']),
            'Humidity': int(user_input['humidity']),
            'Temperature': int(user_input['temperature'])
        }
        rain = self.inference.query(variables=['RainChance'], evidence=evidence)
        sunlight = self.inference.query(variables=['Sunlight'], evidence=evidence)
        return {'RainChance': rain.values[1], 'Sunlight': sunlight.values[1]}

# Initialize AI Model
weather_system = WeatherPredictionSystem()

@app.route('/weather', methods=['POST'])
def weather():
    data = request.json
    city = data.get("city")

    # Get real-time weather
    weather_data = get_weather(city)

    if "error" in weather_data:
        return jsonify({"error": weather_data["error"]}), 400

    # Get user input (convert real values to binary)
    cloud_cover = 1 if "cloud" in weather_data['description'].lower() else 0
    humidity = 1 if weather_data['humidity'] > 60 else 0
    temperature = 1 if weather_data['temperature'] > 20 else 0

    # AI Prediction
    predictions = weather_system.predict_weather({
        'cloud_cover': cloud_cover,
        'humidity': humidity,
        'temperature': temperature
    })

    # Response
    return jsonify({
        "temperature": weather_data['temperature'],
        "humidity": weather_data['humidity'],
        "description": weather_data['description'],
        "rain_chance": predictions['RainChance'],
        "sunlight": predictions['Sunlight']
    })

if __name__ == "__main__":
    app.run(debug=True)
