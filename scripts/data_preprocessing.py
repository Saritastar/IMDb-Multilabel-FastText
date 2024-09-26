import pandas as pd

# Load IMDb dataset (assuming it's a CSV with 'review' and 'sentiment' columns)
df = pd.read_csv('data/IMDb Movie Reviews Dataset.csv')

# Create a new column where we add __label__ prefix to sentiment
df['label'] = df['sentiment'].apply(lambda x: '__label__' + x)

# Create a file with the formatted data for fastText
with open('imdb_fasttext.txt', 'w') as f:
    for row in df[['label', 'review']].itertuples(index=False):
        f.write(f"{row.label} {row.review}\n")