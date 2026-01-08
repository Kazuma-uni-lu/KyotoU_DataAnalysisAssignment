import pandas as pd
def parse():
    df0 = pd.read_csv('SeoulBikeData.csv', encoding="latin1")
    holiday_map = {
        "Holiday": 1,
        "No Holiday": 0,
    }
    df0["Holiday"] = df0["Holiday"].map(holiday_map)
    fd_map = {
        "Yes": 1,
        "No": 0,
    }
    df0["Functioning Day"] = df0["Functioning Day"].map(fd_map)
    df0["Month"] = pd.to_datetime(df0["Date"], format="%d/%m/%Y", errors="coerce").dt.month
    df = df0[  ['Rented Bike Count','Hour','Temperature(°C)','Humidity(%)','Wind speed (m/s)','Visibility (10m)','Solar Radiation (MJ/m2)','Rainfall(mm)','Snowfall (cm)','Holiday','Functioning Day', 'Month']  ]
    df = df.rename( columns={'Rented Bike Count':'Bike Count', 'Temperature(°C)':'Temperature','Humidity(%)':'Humidity','Wind speed (m/s)':'Wind Speed','Visibility (10m)':'Visibility','Dew point temperature(?C)':'Dew Point Temperature','Solar Radiation (MJ/m2)':'Solar Radiation','Rainfall(mm)':'Rainfall','Snowfall (cm)':'Snowfall'} )
    return df