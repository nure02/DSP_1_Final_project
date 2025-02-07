import joblib
import numpy as np
import pandas as pd

lg = joblib.load('models/lg.pkl')
knn = joblib.load('models/knn.pkl')

precipitation = float(input('Enter the value of precipitation: '))
temp_max = float(input('Enter the value of maximum temperature: '))
temp_min = float(input('Enter the value of maximum temperature: '))
wind = float(input('Enter the value of wind: '))

test =np.array([[precipitation,temp_max,temp_min,wind]])
ot = lg.predict(test)
print("The weather is:")
if(ot==0):
    print("Drizzle")
elif(ot==1):
    print("Fog")
elif(ot==2):
    print("Rain")
elif(ot==3):
    print("Snow")
else:
    print("Sun")