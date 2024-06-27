import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client
import time
import logging
from scipy.stats import norm
from datetime import datetime
from joblib import dump, load
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
from nsepy import get_history
from jugaad_data import nse
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob

# Configuration
EMAIL_USER = 'your_email@gmail.com'
EMAIL_PASSWORD = 'your_email_password'
TWILIO_SID = 'your_twilio_sid'
TWILIO_TOKEN = 'your_twilio_token'
TWILIO_PHONE_NUMBER = 'your_twilio_phone_number'
YOUR_PHONE_NUMBER = 'your_phone_number'
SYMBOLS = {
    'Nifty': '^NSEI',
    'Bank Nifty': '^NSEBANK',
    'Midcap Nifty': '^NSEMIDCAP',
    'Bankex': '^BANKNIFTY',
    'Sensex': '^BSESN',
    'Fin Nifty': '^NIFTYFIN'
}
ALERT_THRESHOLD = 0.95
CHECK_INTERVAL = 1  # Check every 1 second for real-time analysis
MODEL_FILE = 'gamma_blast_model.joblib'

# Initialize Twilio Client
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

# Configure logging
logging.basicConfig(level=logging.INFO, filename='gamma_blast.log', filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define market open and close times (for illustrative purposes, adjust as per actual market timings)
MARKET_OPEN = datetime.strptime('09:15', '%H:%M').time()
MARKET_CLOSE = datetime.strptime('15:30', '%H:%M').time()

# Additional configuration for expiry days and indices
INDEX_EXPIRY_DAYS = {
    'BANKEX': 'Monday',
    'SENSEX': 'Friday',
    'MIDCAP NIFTY': 'Monday',
    'FINNIFTY': 'Tuesday',
    'BANKNIFTY': 'Wednesday',
    'NIFTY': 'Thursday'
}

def send_email(subject, body):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = EMAIL_USER
    msg['To'] = YOUR_PHONE_NUMBER

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, YOUR_PHONE_NUMBER, msg.as_string())

def send_whatsapp_message(body):
    twilio_client.messages.create(
        body=body,
        from_=f'whatsapp:{TWILIO_PHONE_NUMBER}',
        to=f'whatsapp:{YOUR_PHONE_NUMBER}'
    )

def get_option_chain(ticker):
    stock = yf.Ticker(ticker)
    options_expiry_dates = stock.options
    options_data = []

    for expiry in options_expiry_dates:
        opt_chain = stock.option_chain(expiry)
        calls = opt_chain.calls
        puts = opt_chain.puts
        calls['type'] = 'call'
        puts['type'] = 'put'
        options_data.append(calls)
        options_data.append(puts)

    options_df = pd.concat(options_data)
    return options_df

def calculate_indicators(options_df):
    options_df['price_change'] = options_df['lastPrice'].pct_change()
    options_df['iv_rank'] = options_df['impliedVolatility'].rank(pct=True)
    options_df['volume_rank'] = options_df['volume'].rank(pct=True)
    options_df['open_interest_rank'] = options_df['openInterest'].rank(pct=True)
    options_df['price_zscore'] = (options_df['lastPrice'] - options_df['lastPrice'].mean()) / options_df['lastPrice'].std()
    options_df['bid_ask_spread'] = options_df['ask'] - options_df['bid']
    options_df['historical_volatility'] = options_df['lastPrice'].rolling(window=30).std() * np.sqrt(252)
    options_df['delta'] = options_df.apply(lambda row: calculate_delta(row['lastPrice'], row['strike'], row['impliedVolatility'], row['daysToExpiration'], row['riskFreeRate'], row['type']), axis=1)
    options_df['gamma'] = options_df.apply(lambda row: calculate_gamma(row['lastPrice'], row['strike'], row['impliedVolatility'], row['daysToExpiration'], row['riskFreeRate'], row['type']), axis=1)
    options_df['theta'] = options_df.apply(lambda row: calculate_theta(row['lastPrice'], row['strike'], row['impliedVolatility'], row['daysToExpiration'], row['riskFreeRate'], row['type']), axis=1)
    options_df['vega'] = options_df.apply(lambda row: calculate_vega(row['lastPrice'], row['strike'], row['impliedVolatility'], row['daysToExpiration'], row['riskFreeRate'], row['type']), axis=1)
    options_df['rsi'] = calculate_rsi(options_df['lastPrice'])
    options_df['skewness'] = options_df['lastPrice'].skew()
    options_df['kurtosis'] = options_df['lastPrice'].kurtosis()
    options_df['macd'] = calculate_macd(options_df['lastPrice'])
    options_df['bollinger_bands'] = calculate_bollinger_bands(options_df['lastPrice'])
    options_df['fibonacci_retracement'] = calculate_fibonacci_retracement(options_df['lastPrice'])
    return options_df

def calculate_delta(S, K, sigma, T, r, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    if option_type == 'call':
        return norm.cdf(d1)
    elif option_type == 'put':
        return -norm.cdf(-d1)

def calculate_gamma(S, K, sigma, T, r, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def calculate_theta(S, K, sigma, T, r, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    elif option_type == 'put':
        return (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

def calculate_vega(S, K, sigma, T, r, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100

def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, short_window=12, long_window=26, signal_window=9):
    short_ema = prices.ewm(span=short_window, adjust=False).mean()
    long_ema = prices.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal_line = macd.ewm(span=signal_window, adjust=False).mean()
    return macd - signal_line

def calculate_bollinger_bands(prices, window=20):
    sma = prices.rolling(window).mean()
    std_dev = prices.rolling(window).std()
    upper_band = sma + (std_dev * 2)
    lower_band = sma - (std_dev * 2)
    return upper_band, lower_band

def calculate_fibonacci_retracement(prices):
    max_price = prices.max()
    min_price = prices.min()
    diff = max_price - min_price
    retracement_levels = {
        '0.0%': max_price,
        '23.6%': max_price - diff * 0.236,
        '38.2%': max_price - diff * 0.382,
        '50.0%': max_price - diff * 0.5,
        '61.8%': max_price - diff * 0.618,
        '100.0%': min_price
    }
    return retracement_levels

def get_macro_economic_indicators():
    # Dummy implementation for macro indicators
    # In practice, fetch data from a reliable source
    return {
        'gdp_growth': 5.0,  # Example value
        'interest_rate': 4.5,  # Example value
        'inflation_rate': 3.0  # Example value
    }

def get_event_based_data():
    # Dummy implementation for event data
    # In practice, fetch data from a reliable source
    return {
        'corporate_earnings': 1,  # Example value
        'geopolitical_events': 0,  # Example value
        'regulatory_changes': 0  # Example value
    }

def get_sentiment_analysis(ticker):
    # Use MoneyControl or other sources to get news and analyze sentiment
    url = f'https://www.moneycontrol.com/financials/{ticker.lower()}'
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    headlines = [tag.get_text() for tag in soup.find_all('a', class_='arial11')]
    sentiment_scores = [TextBlob(headline).sentiment.polarity for headline in headlines]
    average_sentiment = np.mean(sentiment_scores)
    return average_sentiment

def fetch_and_prepare_data(ticker, start_date, end_date):
    data_yf = yf.download(ticker, start=start_date, end=end_date)
    data_nsepy = get_history(symbol=ticker, start=start_date, end=end_date)
    data_jugaad = nse.get_quote(ticker)

    combined_data = pd.concat([data_yf, data_nsepy, data_jugaad], axis=1).dropna()
    return combined_data

def build_and_train_model(X, y):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier())
    ])

    param_grid = {
        'classifier__n_estimators': [100, 200, 300],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_
    return best_model

def main():
    while True:
        now = datetime.now().time()
        if MARKET_OPEN <= now <= MARKET_CLOSE:
            for index_name, ticker in SYMBOLS.items():
                options_df = get_option_chain(ticker)
                options_df = calculate_indicators(options_df)

                # Fetch additional data
                macro_indicators = get_macro_economic_indicators()
                event_data = get_event_based_data()
                sentiment = get_sentiment_analysis(ticker)

                # Combine all data into a single DataFrame
                options_df['macro_gdp_growth'] = macro_indicators['gdp_growth']
                options_df['macro_interest_rate'] = macro_indicators['interest_rate']
                options_df['macro_inflation_rate'] = macro_indicators['inflation_rate']
                options_df['event_corporate_earnings'] = event_data['corporate_earnings']
                options_df['event_geopolitical_events'] = event_data['geopolitical_events']
                options_df['event_regulatory_changes'] = event_data['regulatory_changes']
                options_df['sentiment'] = sentiment

                # Fetch historical data for the ticker
                start_date = datetime.now() - pd.DateOffset(years=5)
                end_date = datetime.now()
                combined_data = fetch_and_prepare_data(ticker, start_date, end_date)

                # Train the model with the prepared data
                X = combined_data.drop(columns=['target'])
                y = combined_data['target']
                model = build_and_train_model(X, y)

                # Predict and analyze the options
                predictions = model.predict(options_df.drop(columns=['type']))
                options_df['prediction'] = predictions

                gamma_blasts = options_df[(options_df['prediction'] == 1) & (options_df['gamma'] > ALERT_THRESHOLD)]
                if not gamma_blasts.empty:
                    message = f"Gamma Blast Detected in {index_name} options:\n" + gamma_blasts.to_string()
                    send_email("Gamma Blast Alert", message)
                    send_whatsapp_message(message)

            time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

