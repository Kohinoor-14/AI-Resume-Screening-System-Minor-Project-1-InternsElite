def calculate_resume_score(skills_found):
    """
    Calculate resume score based on skills count
    """

    max_skills = 15

    score = min(
        (len(skills_found) / max_skills) * 100,
        100
    )

    return round(score, 2)