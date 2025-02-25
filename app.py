import pandas as pd
import re
import string
import joblib
from flask import Flask, request, jsonify
from google_play_scraper import Sort, reviews
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import psycopg2
from psycopg2.extras import execute_values
import logging

# Initialize Flask App
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

class SentimentPredictor:
    def __init__(self, vectorizer_path, model_path):
        self.tfidf_vectorizer = joblib.load(vectorizer_path)
        self.logistic_model = joblib.load(model_path)
        self.stop_words = stopwords.words('indonesian')
        self.stemmer = StemmerFactory().create_stemmer()

    def preprocess_text(self, text):
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
        text = re.sub('\\d+', '', text)  # Remove digits
        text = ' '.join([word for word in text.split() if word not in self.stop_words])  # Remove stopwords
        text = self.stemmer.stem(text)  # Stemming
        return text

    def predict(self, texts):
        preprocessed_texts = [self.preprocess_text(text) for text in texts]
        vectorized_texts = self.tfidf_vectorizer.transform(preprocessed_texts)
        return self.logistic_model.predict(vectorized_texts)

# Initialize Sentiment Predictor
predictor = SentimentPredictor('tfidf_vectorizer.pkl', 'logistic_model.pkl')

@app.route('/fetch_reviews', methods=['POST'])
def fetch_reviews():
    try:
        data = request.json

        if 'text' in data:  # If single review is provided
            text = data['text']
            sentiment = predictor.predict([text])[0]

            return jsonify({'text': text, 'predicted_sentiment': sentiment})

        else:  # Fetch Google Play reviews
            result, _ = reviews(
                'co.id.bankbsi.superapp',
                lang='id',
                country='id',
                count=5000,
                sort=Sort.NEWEST,
                filter_score_with=None
            )

            df = pd.DataFrame(result)
            df = df[['reviewId', 'userName', 'content', 'score', 'at', 'thumbsUpCount', 'appVersion']]
            df['content'] = df['content'].fillna('')  # Handle missing values
            df['sentiment'] = predictor.predict(df['content'].astype(str))

            # Convert DataFrame to JSON
            reviews_json = df.to_json(orient='records')

            # Append to PostgreSQL database
            conn = psycopg2.connect(
                host="104.154.175.45",
                database="multimatics-backend",
                user="tukam",
                password="tukam"  # Replace with your actual password
            )
            cursor = conn.cursor()

            # Check for existing reviews and filter out duplicates
            existing_review_ids_query = 'SELECT "reviewId" FROM byond_review WHERE "reviewId" = ANY(%s)'
            cursor.execute(existing_review_ids_query, (df['reviewId'].tolist(),))
            existing_review_ids = {row[0] for row in cursor.fetchall()}

            new_reviews = df[~df['reviewId'].isin(existing_review_ids)]

            if not new_reviews.empty:
                insert_query = """
                INSERT INTO byond_review ("reviewId", "userName", "content", "score", "at", "thumbsUpCount", "appVersion", "sentiment")
                VALUES %s
                """
                execute_values(cursor, insert_query, new_reviews[['reviewId', 'userName', 'content', 'score', 'at', 'thumbsUpCount', 'appVersion', 'sentiment']].values.tolist())
                conn.commit()


            cursor.close()
            conn.close()

            return jsonify({'number of reviews inserted': len(new_reviews)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)