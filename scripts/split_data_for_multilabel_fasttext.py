from sklearn.model_selection import train_test_split
import pandas as pd
import random

# Load the multilabel dataset
df = pd.read_csv('data/IMDb Movie Reviews Dataset.csv')

# Randomly assign genres and emotions if you haven't done that yet
genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance']
emotions = ['Happy', 'Sad', 'Angry', 'Fearful', 'Neutral']
df['genre'] = [random.choice(genres) for _ in range(len(df))]
df['emotion'] = [random.choice(emotions) for _ in range(len(df))]

# Add multilabel format for fastText
df['labels'] = df.apply(lambda row: '__label__' + row['sentiment'] + ' __label__' + row['genre'] + ' __label__' + row['emotion'], axis=1)

# Split the data into training and testing sets (80% train, 20% test)
train, test = train_test_split(df[['labels', 'review']], test_size=0.2, random_state=42)

# Write the formatted train and test sets to text files for fastText
train.to_csv('data/imdb_fasttext_multilabel_train.txt', index=False, sep=' ', header=False)
test.to_csv('data/imdb_fasttext_multilabel_test.txt', index=False, sep=' ', header=False)

print("Multilabel train-test split done successfully!")
