#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score


# In[11]:


df = pd.read_csv('insurance3r2.csv')

for col in ['sex', 'smoker', 'region']:
    df[col] = df[col].astype(str).str.strip().str.lower()

for col in ['sex', 'smoker', 'region']:
    print(f"Classes found for '{col}':", sorted(df[col].unique()))
df['claim'] = (df['charges'] > df['charges'].median()).astype(int)

le_sex = LabelEncoder()
df['sex'] = le_sex.fit_transform(df['sex'].astype(str).str.lower())

le_smoker = LabelEncoder(); 
df['smoker'] = le_smoker.fit_transform(df['smoker'])
le_region = LabelEncoder(); 
df['region'] = le_region.fit_transform(df['region'])

features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])


# In[12]:


X = df[features]
y = df['claim']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

print("--- Classification Model ---")
print(classification_report(y_test, clf.predict(X_test), target_names=['No Claim','Claim']))

df_claimed = df[df['claim'] == 1]
Xr = df_claimed[features]
yr = df_claimed['charges']

Xr_train, Xr_test, yr_train, yr_test = train_test_split(Xr, yr, test_size=0.2, random_state=42)
regr = RandomForestRegressor(random_state=42)
regr.fit(Xr_train, yr_train)

yr_pred = regr.predict(Xr_test)
print("--- Regression Model ---")
print(f"MAE (regression): {mean_absolute_error(yr_test, yr_pred):.2f}")
print(f"R2 (regression): {r2_score(yr_test, yr_pred):.2f}")


# In[14]:


def get_number(prompt, _type=int):
    while True:
        val = input(prompt).strip()
        try: return _type(val)
        except ValueError: print(f"Please enter a valid {_type.__name__}.")

def print_category_mapping(encoder, field_name):
    display_map = ", ".join([f"{i}-{cat}" for i, cat in enumerate(encoder.classes_)])
    print(f"{field_name} options: {display_map}")
sex_mapping = {0: 'male', 1: 'female'}
smoker_mapping = {0: 'no', 1: 'yes'}
region_mapping = {0: 'northeast', 1: 'northwest', 2: 'southeast', 3: 'southwest'}


print("\n=== INSURANCE CLAIM PREDICTION ===")
age = get_number("Age (years): ", int)
bmi = get_number("BMI: ", float)
children = get_number("Number of children/dependents: ", int)

def print_manual_mapping(mapping, field_name):
    display_map = ', '.join(f"{k}-{v}" for k, v in mapping.items())
    print(f"{field_name}({display_map})")


print_manual_mapping(sex_mapping, "Sex")
sex_code = get_number("Enter the code for Sex: ", int)

print_manual_mapping(smoker_mapping, "Smoker")
smoker_code = get_number("Enter the code for Smoker: ", int)

print_manual_mapping(region_mapping, "Region")
region_code = get_number("Enter the code for Region: ", int)

user_dict = {
    "age": age,
    "sex": sex_code,
    "bmi": bmi,
    "children": children,
    "smoker": smoker_code,
    "region": region_code
}

user_df = pd.DataFrame([user_dict])
user_df[features] = scaler.transform(user_df[features])
prob_claim = clf.predict_proba(user_df[features])[0][1]
is_claim = clf.predict(user_df[features])[0]

if is_claim:
    pred_amt = regr.predict(user_df[features])[0]
    print(f"\nPrediction: This person is LIKELY to claim insurance.")
    print(f"Estimated claim amount: ₹{pred_amt:.2f}")
else:
    print("\nPrediction: This person is UNLIKELY to claim insurance.")
print(f"Probability of claim: {prob_claim:.2%}")


# In[ ]:




