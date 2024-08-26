import requests
import time
import pandas as pd
from newsplease import NewsPlease
api_key = 'your_api_key_here'
endpoint = 'https://newsapi.org/v2/everything'
news_df = pd.DataFrame()

data_schema = {
    'Source_ID': str,
    'Soruce_Name': str,
    'Author': str,
    'News_Title': str,
    'News_Description': str,
    'URL': str,
    'Image_URL': str,
    'Published_Date': pd.Timestamp,
    'Content':str
}

news_df = pd.DataFrame(columns=data_schema.keys())

country_names = {
    'ae': 'United Arab Emirates', 'ar': 'Argentina', 'at': 'Austria', 'au': 'Australia',
    'be': 'Belgium', 'bg': 'Bulgaria', 'br': 'Brazil', 'ca': 'Canada', 'ch': 'Switzerland',
    'cn': 'China', 'co': 'Colombia', 'cu': 'Cuba', 'cz': 'Czech Republic', 'de': 'Germany',
    'eg': 'Egypt', 'fr': 'France', 'gb': 'United Kingdom', 'gr': 'Greece', 'hk': 'Hong Kong',
    'hu': 'Hungary', 'id': 'Indonesia', 'ie': 'Ireland', 'il': 'Israel', 'in': 'India',
    'it': 'Italy', 'jp': 'Japan', 'kr': 'South Korea', 'lt': 'Lithuania', 'lv': 'Latvia',
    'ma': 'Morocco', 'mx': 'Mexico', 'my': 'Malaysia', 'ng': 'Nigeria', 'nl': 'Netherlands',
    'no': 'Norway', 'nz': 'New Zealand', 'ph': 'Philippines', 'pl': 'Poland', 'pt': 'Portugal',
    'ro': 'Romania', 'rs': 'Serbia', 'ru': 'Russia', 'sa': 'Saudi Arabia', 'se': 'Sweden',
    'sg': 'Singapore', 'si': 'Slovenia', 'sk': 'Slovakia', 'th': 'Thailand', 'tr': 'Turkey',
    'tw': 'Taiwan', 'ua': 'Ukraine', 'us': 'United States', 've': 'Venezuela', 'za': 'South Africa'
}

params = {
    'apiKey': api_key,
    'pageSize': 5,     # Limit to 5 articles per request
    'language': 'en'
}

for country_code, country_name in country_names.items():
    params['q'] = country_name
    response = requests.get(endpoint, params=params)

    if response.status_code == 200:
        data = response.json()
        articles = data['articles']
        print(f"\nTop 5 News from {country_name}:")
        for idx, article in enumerate(articles, 1):
            print(article['source']['id'])
            article_url = NewsPlease.from_url(article['url'])
            # print(article['url'])
            if article_url and article_url.text:
                news_df['Source_ID'] =  article['source']['id']
                news_df['Soruce_Name'] = article['source']['name']
                news_df['Author'] = article['author']
                news_df['News_Title'] = article['title']
                news_df['News_Description'] = article['description']
                news_df['URL'] = article['url']
                news_df['Image_URL'] = article['urlToImage']
                news_df['Published_Date'] =  article['publishedAt']
                news_df['Content'] = article_url.text
    else:
        print(f"Error fetching news for {country_name}: {response.status_code}")
    time.sleep(1)
