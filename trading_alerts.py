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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
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

    # Synthetic data generation
    X_train, y_train = options_df[['impliedVolatility', 'volume', 'openInterest', 'price_zscore',
                                   'bid_ask_spread', 'historical_volatility', 'rsi', 'skewness', 'kurtosis',
                                   'delta', 'gamma', 'theta', 'vega']], options_df['price_change'].apply(lambda x: 1 if x > 0.5 else 0)
    options_df = generate_synthetic_data(X_train, y_train)

    # Sentiment analysis (example: using a placeholder sentiment score)
    options_df['sentiment_score'] = options_df['news_text'].apply(get_sentiment)

    # Additional technical indicators (example: Bollinger Bands)
    options_df['upper_band'], options_df['lower_band'] = calculate_bollinger_bands(options_df['lastPrice'])

    # Seasonality features
    options_df['is_near_expiry'] = options_df['expiry_date'].apply(lambda x: 1 if x is close to today else 0)

    return options_df

def generate_synthetic_data(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def get_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity
    return sentiment_score

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

def calculate_bollinger_bands(prices, window=20):
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    return upper_band, lower_band

def train_predictive_model(data):
    features = data[['impliedVolatility', 'volume', 'openInterest', 'price_zscore',
                     'bid_ask_spread', 'historical_volatility', 'rsi', 'skewness', 'kurtosis',
                     'delta', 'gamma', 'theta', 'vega', 'sentiment_score', 'upper_band', 'lower_band', 'is_near_expiry']]
    target = data['price_change'].apply(lambda x: 1 if x > 0.5 else 0)  # Example threshold for significant change

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    clf_svm = SVC(kernel='linear', probability=True, random_state=42)
    clf_knn = KNeighborsClassifier()
    clf_nn = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)

    pipeline_rf = Pipeline([('scaler', scaler), ('clf', clf_rf)])
    pipeline_gb = Pipeline([('scaler', scaler), ('clf', clf_gb)])
    pipeline_svm = Pipeline([('scaler', scaler), ('clf', clf_svm)])
    pipeline_knn = Pipeline([('scaler', scaler), ('clf', clf_knn)])
    pipeline_nn = Pipeline([('scaler', scaler), ('clf', clf_nn)])

    # Define the parameter grids for GridSearchCV
    param_grid_rf = {
        'clf__n_estimators': [50, 100, 200],
        'clf__max_depth': [None, 10, 20]
    }

    param_grid_gb = {
        'clf__n_estimators': [50, 100, 200],
        'clf__learning_rate': [0.05, 0.1, 0.2]
    }

    param_grid_svm = {
        'clf__C': [0.1, 1, 10],
        'clf__gamma': [0.1, 1, 10],
    }

    param_grid_knn = {
        'clf__n_neighbors': [3, 5, 7],
        'clf__weights': ['uniform', 'distance']
    }

    param_grid_nn = {
        'clf__alpha': [0.0001, 0.001, 0.01],
        'clf__learning_rate_init': [0.001, 0.01, 0.1]
    }

    # Train models using GridSearchCV for hyperparameter tuning
    grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, n_jobs=-1)
    grid_search_gb = GridSearchCV(pipeline_gb, param_grid_gb, cv=5, n_jobs=-1)
    grid_search_svm = GridSearchCV(pipeline_svm, param_grid_svm, cv=5, n_jobs=-1)
    grid_search_knn = GridSearchCV(pipeline_knn, param_grid_knn, cv=5, n_jobs=-1)
    grid_search_nn = GridSearchCV(pipeline_nn, param_grid_nn, cv=5, n_jobs=-1)

    # Fit models
    grid_search_rf.fit(X_train, y_train)
    grid_search_gb.fit(X_train, y_train)
    grid_search_svm.fit(X_train, y_train)
    grid_search_knn.fit(X_train, y_train)
    grid_search_nn.fit(X_train, y_train)

    # Evaluate models
    best_model_rf = grid_search_rf.best_estimator_
    best_model_gb = grid_search_gb.best_estimator_
    best_model_svm = grid_search_svm.best_estimator_
    best_model_knn = grid_search_knn.best_estimator_
    best_model_nn = grid_search_nn.best_estimator_

    logging.info(f"Random Forest - Best Parameters: {grid_search_rf.best_params_}")
    logging.info(f"Gradient Boosting - Best Parameters: {grid_search_gb.best_params_}")
    logging.info(f"SVM - Best Parameters: {grid_search_svm.best_params_}")
    logging.info(f"KNN - Best Parameters: {grid_search_knn.best_params_}")
    logging.info(f"Neural Network - Best Parameters: {grid_search_nn.best_params_}")

    # Voting Classifier
    voting_clf = VotingClassifier(estimators=[('rf', best_model_rf), ('gb', best_model_gb), ('svm', best_model_svm),
                                              ('knn', best_model_knn), ('nn', best_model_nn)], voting='soft')

    voting_clf.fit(X_train, y_train)

    # Evaluate the ensemble model
    y_pred_voting = voting_clf.predict(X_test)
    logging.info(f"Voting Classifier - Accuracy: {accuracy_score(y_test, y_pred_voting)}, "
                 f"Precision: {precision_score(y_test, y_pred_voting)}, "
                 f"Recall: {recall_score(y_test, y_pred_voting)}")

    # Save the best model
    dump(voting_clf, MODEL_FILE)

    return voting_clf

def load_predictive_model():
    return load(MODEL_FILE)

def make_predictions(options_df, model):
    features = options_df[['impliedVolatility', 'volume', 'openInterest', 'price_zscore',
                           'bid_ask_spread', 'historical_volatility', 'rsi', 'skewness', 'kurtosis',
                           'delta', 'gamma', 'theta', 'vega', 'sentiment_score', 'upper_band', 'lower_band', 'is_near_expiry']]
    options_df['prediction'] = model.predict_proba(features)[:, 1]

    return options_df

def check_for_gamma_blast(options_df, threshold=0.95):
    alerts = options_df[options_df['prediction'] > threshold]
    return alerts

def main():
    try:
        model = load_predictive_model()
        logging.info("Loaded existing model.")
    except FileNotFoundError:
        logging.info("Model file not found. Training new model.")
        historical_data = get_historical_option_chain(SYMBOL, start_date, end_date)
        historical_data = calculate_indicators(historical_data)
        model = train_predictive_model(historical_data)

    while True:
        try:
            current_time = datetime.now().time()

            # Check if market is open
            if MARKET_OPEN <= current_time <= MARKET_CLOSE:
                logging.info("Fetching real-time option chain data...")
                options_df = get_option_chain(SYMBOL)
                options_df = calculate_indicators(options_df)

                # Store real-time data for further analysis
                store_real_time_data(options_df)

            else:
                logging.info("Market closed. Fetching historical data for analysis...")
                options_df = fetch_stored_real_time_data()

                if options_df.empty:
                    logging.warning("No stored real-time data available.")
                    time.sleep(CHECK_INTERVAL)
                    continue  # Retry fetching data

            if not options_df.empty:
                options_df = make_predictions(options_df, model)
                alerts = check_for_gamma_blast(options_df, ALERT_THRESHOLD)

                for index, row in alerts.iterrows():
                    message = (f"Gamma Blast Alert: {row['type'].capitalize()} option for {SYMBOL} with strike {row['strike']} "
                               f"might see significant movement.\nLast Price: {row['lastPrice']}\nIV: {row['impliedVolatility']}\n"
                               f"Volume: {row['volume']}\nOpen Interest: {row['openInterest']}")

                    # Calculate Stop Loss based on historical data (example logic)
                    historical_mean_price = options_df.loc[index, 'lastPrice'].mean()
                    stop_loss = historical_mean_price * 0.98  # 2% below mean price as an example

                    message += f"\nStop Loss: {stop_loss}"

                    logging.info(message)
                    send_email('Gamma Blast Alert', message)
                    send_whatsapp_message(message)

        except Exception as e:
            logging.error(f"Error in main loop: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

