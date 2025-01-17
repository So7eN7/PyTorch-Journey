import matplotlib.pyplot as plt
from data_loading import df
plt.figure(figsize=(8, 6))
plt.hist(df.rating)
plt.xlabel("Rating")
plt.ylabel("Count")
plt.title("Count of Ratings")
plt.show()