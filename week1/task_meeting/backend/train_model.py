import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# Ensure models directory exists
os.makedirs('backend/models', exist_ok=True)

# 1. Mock Dataset
# Map tasks to specializations:
# - John: "Database Admin"
# - Sarah: "Finance Manager"
# - Mike: "Sales Representative"
# - Elena: "Marketing Specialist"
# - David: "Security Analyst"

data = [
    ("Update the SQL server", "Database Admin"),
    ("Optimize database queries", "Database Admin"),
    ("Backup the data warehouse", "Database Admin"),
    ("Fix the database connection error", "Database Admin"),
    ("Migrate to new SQL version", "Database Admin"),
    
    ("Prepare the budget for Q4", "Finance Manager"),
    ("Review the quarterly financial report", "Finance Manager"),
    ("Calculate taxes for the year", "Finance Manager"),
    ("Approve the expense reports", "Finance Manager"),
    ("Audit the payroll", "Finance Manager"),
    
    ("Call the client regarding the new contract", "Sales Representative"),
    ("Follow up with the lead from the conference", "Sales Representative"),
    ("Prepare the sales pitch deck", "Sales Representative"),
    ("Negotiate the deal terms", "Sales Representative"),
    ("Schedule a demo with the prospect", "Sales Representative"),
    
    ("Draft the press release for the product launch", "Marketing Specialist"),
    ("Update the social media campaign", "Marketing Specialist"),
    ("Design the new brochure", "Marketing Specialist"),
    ("Write a blog post about the update", "Marketing Specialist"),
    ("Analyze website traffic", "Marketing Specialist"),
    
    ("Check the security logs for suspicious activity", "Security Analyst"),
    ("Update the firewall rules", "Security Analyst"),
    ("Patch the security vulnerability", "Security Analyst"),
    ("Conduct a penetration test", "Security Analyst"),
    ("Review user access permissions", "Security Analyst"),
]

X = [item[0] for item in data]
y = [item[1] for item in data]

# 2. Create Pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 3. Train
print("Training model...")
model.fit(X, y)

# 4. Save
model_path = 'backend/models/task_classifier.pkl'
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
