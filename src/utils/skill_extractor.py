# Common skills database

SKILLS_DB = [
    "python",
    "java",
    "c++",
    "sql",
    "mysql",
    "mongodb",
    "pandas",
    "numpy",
    "tensorflow",
    "keras",
    "pytorch",
    "machine learning",
    "deep learning",
    "data science",
    "scikit-learn",
    "django",
    "flask",
    "fastapi",
    "html",
    "css",
    "javascript",
    "react",
    "nodejs",
    "git",
    "github",
    "docker",
    "kubernetes",
    "aws",
    "azure",
    "power bi",
    "tableau",
    "excel"
]


def extract_skills(text):
    """
    Extract skills from resume text
    """

    text = text.lower()

    found_skills = []

    for skill in SKILLS_DB:
        if skill.lower() in text:
            found_skills.append(skill)

    return sorted(list(set(found_skills)))