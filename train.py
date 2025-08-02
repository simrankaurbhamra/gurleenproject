import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import os

# Load and prepare data
df = pd.read_csv("CareerAdvisory.csv")
df.columns = df.columns.str.strip()

# Fill default columns if missing
if 'Soft_Skills' not in df.columns: df['Soft_Skills'] = 7
if 'Thinking_Ability' not in df.columns: df['Thinking_Ability'] = 6

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Stream_in_12th', 'Entrance_Exam', 'Subject_Strength',
                    'Scholarship_Eligibility', 'Study_Abroad_Plan', 'Target_College',
                    'Career_Interest', 'Backup_Course', 'Counselor_Recommendation',
                    'Interest_Domain', 'Target_Job_Role']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Features
features = ['Stream_in_12th', 'Entrance_Exam', 'Subject_Strength', '12th_Percentage',
            'Aptitude_Test_Score', 'Scholarship_Eligibility', 'Study_Abroad_Plan',
            'Interest_Domain', 'Soft_Skills', 'Thinking_Ability']

X = df[features]
targets = {
    'Target_College': df['Target_College'],
    'Career_Interest': df['Career_Interest'],
    'Backup_Course': df['Backup_Course'],
    'Counselor_Recommendation': df['Counselor_Recommendation'],
    'Target_Job_Role': df['Target_Job_Role']
}

# Train and save each model
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

for name, y in targets.items():
    model = DecisionTreeClassifier()
    model.fit(X, y)
    
    with open(f"{model_dir}/{name}.pkl", "wb") as f:
        pickle.dump(model, f)

# Save encoders
with open(f"{model_dir}/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

print("✅ All models and label encoders saved to 'models/' folder")
