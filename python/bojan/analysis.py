import pandas as pd
import os
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Load the data
data_path = os.path.join('C:\\', 'paul', 'bojan', 'CommData.csv')
df = pd.read_csv(data_path)
print(df.columns)

# Fit the model
model = smf.ols(formula='VO2max ~ C(CommToSch)*C(gender) + age + MVPAsqrt + C(CommToSch):DistLog2ToSch', data=df).fit()

# Print the summary
print(model.summary())

# Create new data frame for prediction
new = pd.DataFrame({
    'gender': ['female']*4 + ['male']*4,
    'CommToSch': ['car', 'public', 'walk', 'wheels']*2,
    'age': [df['age'].mean()]*8,
    'MVPAsqrt': [df['MVPAsqrt'].mean()]*8,
    'DistLog2ToSch': [df.groupby('CommToSch')['DistLog2ToSch'].median().iloc[i] for i in range(4)]*2
})

# Convert to categorical
new['gender'] = pd.Categorical(new['gender'])
new['CommToSch'] = pd.Categorical(new['CommToSch'])

# Predict
new['pred'] = model.predict(new)

# Get prediction intervals
_, new['lwr'], new['upr'] = wls_prediction_std(model, new)

# Plot
plt.figure(figsize=(14, 13))
for g in new['gender'].unique():
    data = new[new['gender'] == g]
    plt.errorbar(data['CommToSch'], data['pred'], yerr=[data['pred']-data['lwr'], data['upr']-data['pred']], fmt='o')
plt.xlabel('Commuting mode (from home to school)')
plt.ylabel('VO2max (predicted at mode median distance)')
plt.savefig('Fig2.png', dpi=600)