from src.utils.ats_matcher import calculate_ats_score

resume = """
Python
Django
Flask
SQL
Git
"""

job_description = """
Python Developer

Required Skills:
Python
Django
Flask
SQL
Git
Docker
AWS
"""

score = calculate_ats_score(
    resume,
    job_description
)

print("\nATS Match Score:")
print(f"{score}%")