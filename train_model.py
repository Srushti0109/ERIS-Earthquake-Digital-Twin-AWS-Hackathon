import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# 1. Create Data based on your research logic
data = {
    'seismic': np.random.randint(0, 100, 1000),
    'radon': np.random.randint(0, 100, 1000),
    'animal': np.random.randint(0, 100, 1000),
    'emf': np.random.randint(0, 100, 1000),
}
df = pd.DataFrame(data)

# Risk is High (1) if precursors are high (your innovation logic)
df['risk'] = ((df['radon'] > 70) & (df['animal'] > 70) | (df['seismic'] > 80)).astype(int)

# 2. Train Model
X = df.drop('risk', axis=1)
y = df['risk']
model = RandomForestClassifier()
model.fit(X, y)

# 3. Save it
with open('earthquake_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("✅ Brain (Model) generated successfully!")