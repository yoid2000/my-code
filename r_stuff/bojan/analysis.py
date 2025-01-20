import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std

# Load the data
df = pd.read_csv("./FrancisP/CommData.csv")

# Fit the model
model = smf.ols(formula='VO2max ~ C(CommToSch)*C(gender) + age + MVPAsqrt + np.log2(DistToSch)*C(CommToSch)', data=df).fit()

# Print the summary
print(model.summary())

# Create new data frame for prediction
new = pd.DataFrame({
    'gender': [1]*4 + [2]*4,
    'CommToSch': list(range(4))*2,
    'age': [df['age'].mean()]*8,
    'MVPAsqrt': [df['MVPAsqrt'].mean()]*8,
    'DistToSch': [df.groupby('CommToSch')['DistToSch'].median()[i] for i in range(4)]*2
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