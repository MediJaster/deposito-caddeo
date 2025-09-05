import asyncio
import ssl
from open_meteo import DailyParameters, OpenMeteo


def create_ssl_context():
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE


async def main():
    city_name = input("Enter city name: ")
    print(f"Fetching weather data for {city_name}...")

    async with OpenMeteo() as weather:
        city_results = await weather.geocoding(
            name=city_name,
            count=5,
        )

        if not city_results.results:
            print("No results found.")
            return

        for idx, city in enumerate(city_results.results, start=1):
            print(
                f"{idx}. {city.name}, {city.country} (Lat: {city.latitude}, Lon: {city.longitude})"
            )

        selection = int(input("Select a city by number: ")) - 1
        if selection < 0 or selection >= len(city_results.results):
            print("Invalid selection.")
            return

        selected_city = city_results.results[selection]
        weather_data = await weather.forecast(
            latitude=selected_city.latitude,
            longitude=selected_city.longitude,
            current_weather=True,
        )

        if weather_data.current_weather:
            # Ask user for forecast range and display options
            forecast_range = input(
                "Show weather for (1) today, (2) 3 days, or (3) 7 days? Enter 1, 2, or 3: "
            )
            show_wind = input("Show wind speed? (y/n): ").strip().lower() == "y"

            if forecast_range != "1":
                show_precip = (
                    input("Show precipitation percentage? (y/n): ").strip().lower()
                    == "y"
                )
            else:
                show_precip = False

            print("\n")

            if forecast_range == "1":
                cw = weather_data.current_weather

                print(f"Current weather in {selected_city.name}: {cw.temperature}°C")
                if show_wind:
                    print(
                        f"Wind Speed: {cw.wind_speed} km/h, Wind Direction: {cw.wind_direction}°"
                    )
            else:
                days = 3 if forecast_range == "2" else 7
                daily_weather = await weather.forecast(
                    latitude=selected_city.latitude,
                    longitude=selected_city.longitude,
                    daily=[
                        DailyParameters.TEMPERATURE_2M_MAX,
                        DailyParameters.TEMPERATURE_2M_MIN,
                        DailyParameters.PRECIPITATION_SUM,
                        DailyParameters.WIND_SPEED_10M_MAX,
                    ],
                    timezone="auto",
                )

                print(f"Weather forecast for next {days} days in {selected_city.name}:")

                if daily_weather.daily:
                    print(f"Daily forecast data for {selected_city.name}:")

                    for i in range(min(days, len(daily_weather.daily.time))):
                        date = daily_weather.daily.time[i]
                        temp_max = daily_weather.daily.temperature_2m_max[i]
                        temp_min = daily_weather.daily.temperature_2m_min[i]
                        precip = (
                            daily_weather.daily.precipitation_sum[i]
                            if show_precip
                            else None
                        )
                        wind_speed = (
                            daily_weather.daily.wind_speed_10m_max[i]
                            if show_wind
                            else None
                        )

                        print(f"Date: {date}")
                        print(f"  Max Temp: {temp_max}°C")
                        print(f"  Min Temp: {temp_min}°C")
                        if precip is not None:
                            print(f"  Precipitation: {precip} mm")
                        if wind_speed is not None:
                            print(f"  Max Wind Speed: {wind_speed} km/h")
                else:
                    print("No daily forecast data available.")
        else:
            print("No current weather data available.")


if __name__ == "__main__":
    create_ssl_context()
    asyncio.run(main())
