import pandas as pd
df=pd.read_csv(r"https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv")
df["Time"]=pd.to_datetime(df.Time)
df["DayName"]=df.Time.dt.day_name()
df.DayName.value_counts().plot(kind="bar")