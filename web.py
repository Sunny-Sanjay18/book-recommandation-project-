from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pickle

app = Flask(__name__)


# Load data
data = pd.read_csv("final_dataset.csv")

# Load Hybrid Model
with open('hybrid_model.pkl', 'rb') as f:
    algo = pickle.load(f)


# Content-Based Filtering
tfidf = TfidfVectorizer(stop_words='english')
data['Book-Title'] = data['Book-Title'].fillna('')
tfidf_matrix = tfidf.fit_transform(data['Book-Title'])


# Define hybrid_recommendations function
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

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        book_title = request.form['book_title']
        num_recommendations = int(request.form['num_recommendations'])
        recommendations = hybrid_recommendations(book_title, num_recommendations)
        return render_template('result.html', book_title=book_title, recommendations=recommendations)

if __name__ == '__main__':
    app.run(port=8000)



