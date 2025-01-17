import pandas as pd
df = pd.read_csv("./data/ml-latest-small/ratings.csv")

df.head()
df.describe()
df.isnull().sum()
df.userId.nunique(), df.movieId.nunique()
df.rating.value_counts()
