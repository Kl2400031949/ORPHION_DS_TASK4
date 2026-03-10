
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')   


data = pd.read_csv("twitter.csv", encoding="latin-1", header=None)
data.columns = ["target", "id", "date", "flag", "user", "text"]


data['sentiment'] = data['target'].map({0:"Negative", 2:"Neutral", 4:"Positive"})


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)  # remove URLs
    text = re.sub(r'@\w+|#','', text)                   # remove mentions/hashtags
    text = re.sub(r'[^A-Za-z\s]', '', text)             # remove punctuation/numbers
    tokens = nltk.word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)


data['clean_text'] = data['text'].apply(clean_text)


X = data['clean_text']
y = data['sentiment']

vectorizer = CountVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.3, random_state=42)


clf = MultinomialNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

for sentiment in ["Positive", "Negative", "Neutral"]:
    text_data = " ".join(data[data['sentiment']==sentiment]['clean_text'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    
    plt.figure(figsize=(10,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(f"{sentiment} Tweets WordCloud")
    plt.show()
    
    wordcloud.to_file(f"{sentiment.lower()}_wordcloud.png")
