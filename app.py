import pandas as pd
import re
import string
import joblib
from flask import Flask, request, jsonify, send_file
from google_play_scraper import Sort, reviews
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Initialize Flask App
app = Flask(__name__)

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
                count=1000,
                sort=Sort.NEWEST,
                filter_score_with=None
            )

            df = pd.DataFrame(result)
            df = df[['reviewId', 'userName', 'content', 'score', 'at', 'thumbsUpCount']]
            df['predicted_sentiment'] = predictor.predict(df['content'].astype(str))

            # Save as CSV
            csv_filename = "reviews_sentiment.csv"
            df.to_csv(csv_filename, index=False)

            return send_file(csv_filename, mimetype='text/csv', as_attachment=True)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
