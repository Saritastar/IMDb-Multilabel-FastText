import pandas as pd
import random

# Load the dataset (assuming 'review' and 'sentiment' columns are present)
df = pd.read_csv('data/IMDb Movie Reviews Dataset.csv')

# Create a list of genres and emotions to assign
genres = ['Action', 'Comedy', 'Drama', 'Horror', 'Romance']
emotions = ['Happy', 'Sad', 'Angry', 'Fearful', 'Neutral']

# Randomly assign a genre and emotion to each row
df['genre'] = [random.choice(genres) for _ in range(len(df))]
df['emotion'] = [random.choice(emotions) for _ in range(len(df))]

# Add a '__label__' prefix to sentiment, genre, and emotion to prepare for fastText
df['labels'] = df.apply(lambda row: '__label__' + row['sentiment'] + ' __label__' + row['genre'] + ' __label__' + row['emotion'], axis=1)

# Save the dataset in fastText format
with open('data/imdb_fasttext_multilabel.txt', 'w') as f:
    for row in df[['labels', 'review']].itertuples(index=False):
        f.write(f"{row.labels} {row.review}\n")

print("Multilabel dataset created successfully!")
