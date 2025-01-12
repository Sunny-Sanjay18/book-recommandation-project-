import surprise
import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import SVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle


# Load data
data = pd.read_csv("final_dataset.csv")

# Collaborative Filtering
reader = Reader(rating_scale=(1, 10))
surprise_data = Dataset.load_from_df(data[['User-ID', 'ISBN', 'Book-Rating']], reader)
trainset, testset = train_test_split(surprise_data, test_size=0.2)
algo = SVD()
algo.fit(trainset)

# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
data['Book-Title'] = data['Book-Title'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['Book-Title'])

# Hybrid Model
def hybrid_recommendations(book_title, num_recommendations):
    # Content-Based Recommendations
    idx = data[data['Book-Title'] == book_title].index[0]
    cosine_similarities = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    content_based_scores = list(enumerate(cosine_similarities))
    content_based_scores = sorted(content_based_scores, key=lambda x: x[1], reverse=True)
    
    # Collaborative Filtering Recommendations
    book_id = data[data['Book-Title'] == book_title]['ISBN'].iloc[0]
    cf_recommendations = algo.predict(uid=None, iid=book_id, verbose=False).est
    
    # Combine Recommendations
    hybrid_recommendations = []
    already_recommended = set()
    for i, score in content_based_scores:
        title = data.iloc[i]['Book-Title']
        if title != book_title and title not in already_recommended:
            hybrid_recommendations.append((title, data.iloc[i]['Book-Author'], cf_recommendations))
            already_recommended.add(title)
            if len(hybrid_recommendations) == num_recommendations:
                break
                
    return hybrid_recommendations

# User Interface
def get_recommendations():
    book_title = input("Enter the book name: ")
    num_recommendations = int(input("Enter the number of recommendations: "))
    recommendations = hybrid_recommendations(book_title, num_recommendations)
    print("\nInput Book:")
    print(book_title)
    print("\nRecommended Books:")
    for title, author, rating in recommendations:
        print(f"{title} by {author} (Rating: {rating})")

# Run the interface
get_recommendations()

# Dump hybrid model to pickle file
with open('hybrid_model.pkl', 'wb') as f:
    pickle.dump(algo, f)
