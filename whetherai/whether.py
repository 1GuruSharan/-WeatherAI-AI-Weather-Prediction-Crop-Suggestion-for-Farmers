import requests
from pgmpy.models import BayesianNetwork  
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Function to fetch real-time weather data
def get_weather(city):
    api_key = "4be818742436d1138ad21ddc4b9b5270"  
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"
    }   

    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if response.status_code == 200:
            weather_info = {
                "temperature": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"]
            }
            return weather_info
        else:
            print(f"â ï¸ Error: {data['message']}. Please enter a correct city name.")
            return None
    except requests.exceptions.RequestException:
        print("ð¨ No internet connection. Please check and try again.")
        return None

# Weather Prediction System using Bayesian Network
class WeatherPredictionSystem:
    def __init__(self):
        self.model = BayesianNetwork([  # Changed to BayesianNetwork
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

        # Add CPDs to the model
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
        return {'RainChance': rain, 'Sunlight': sunlight}

# Main function to run the AI-powered weather system
def run_weather_prediction_system():
    system = WeatherPredictionSystem()
    print("\nð¾ Welcome to the AI Weather Guide for Farmers ð¾\n")

    city = input("ð Enter the name of your city or village: ").strip()
    real_time_data = get_weather(city)

    if real_time_data:
        print(f"\nð Weather in {city.capitalize()} Today:")
        print(f"ð¡ï¸ Temperature: {real_time_data['temperature']}Â°C")
        print(f"ð§ Humidity: {real_time_data['humidity']}%")
        print(f"ð¥ï¸ Sky Condition: {real_time_data['description'].capitalize()}\n")
    else:
        print("â ï¸ Unable to fetch real-time weather. Let's continue with manual inputs.\n")

    # Farmer-friendly questions
    cloud_cover = input("âï¸ Is the sky mostly covered with clouds? (yes/no): ").strip().lower() == 'yes'
    humidity = real_time_data['humidity'] > 60 if real_time_data else input("ð¦ Is the air humid (feels sticky)? (yes/no): ").strip().lower() == 'yes'
    temperature = real_time_data['temperature'] > 20 if real_time_data else input("ð Is the weather warm? (yes/no): ").strip().lower() == 'yes'

    user_input = {
        'cloud_cover': cloud_cover,
        'humidity': humidity,
        'temperature': temperature
    }

    predictions = system.predict_weather(user_input)

    print("\nð® AI-Based Farming Weather Advice:\n")

    rain_prob = predictions['RainChance'].values[1]
    if rain_prob > 0.5:
        print("ð§ **It may rain today.**")
        print("- ð¾ Keep your crops covered if needed.")
        print("- ð Avoid harvesting today.")
        print("- â Check drainage to avoid waterlogging.")
    else:
        print("âï¸ **No rain expected today.**")
        print("- ð Good day for farming work.")
        print("- ð¦ You may need to water your crops.")

    sun_prob = predictions['Sunlight'].values[1]
    if sun_prob > 0.5:
        print("\nð **Expect bright sunlight today.**")
        print("- ð Wear protection against heat.")
        print("- ð¦ Irrigate crops early in the morning.")
        print("- â¡ Solar-powered equipment will work well.")
    else:
        print("\nâï¸ **Cloudy weather expected.**")
        print("- ð¾ Less sunlight for crops.")
        print("- âï¸ It may be a cooler day.")

    print("\nð **Note:** This is an AI-based prediction. Always check local weather updates!\n")

# Run the system
if __name__ == "__main__":
    run_weather_prediction_system()
