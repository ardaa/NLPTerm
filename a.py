from nltk.stem import WordNetLemmatizer



lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("baker").upper())

if lemmatizer.lemmatize("dances").upper() in "DANCES".upper():
    print("yes")