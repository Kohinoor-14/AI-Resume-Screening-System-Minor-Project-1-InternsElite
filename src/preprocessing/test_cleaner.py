from text_cleaner import clean_resume

sample_resume = """
Python Machine Learning Deep Learning Pandas NumPy
Email: test@gmail.com
https://github.com/test
"""

cleaned_text = clean_resume(sample_resume)

print("\nOriginal Text:\n")
print(sample_resume)

print("\nCleaned Text:\n")
print(cleaned_text)