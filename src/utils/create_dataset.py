import pandas as pd

data = {
    "Category": [
        "Data Science",
        "Data Science",
        "Java Developer",
        "Java Developer",
        "HR",
        "HR",
        "Python Developer",
        "Python Developer"
    ],
    "Resume": [
        "Python Machine Learning Deep Learning Pandas NumPy",
        "Data Analysis Statistics SQL Machine Learning",
        "Java Spring Boot Hibernate REST API",
        "Core Java JDBC OOP Spring Framework",
        "Recruitment Employee Relations HR Management",
        "Talent Acquisition Payroll Performance Management",
        "Python Django Flask API Development",
        "Python Automation Selenium Web Scraping"
    ]
}

df = pd.DataFrame(data)

df.to_csv("data/raw/UpdatedResumeDataSet.csv", index=False)

print("Dataset created successfully!")
print(df.head())