import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import SVC
from xgboost import XGBClassifier
import logging
from scipy.stats import norm
from datetime import datetime
from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Configuration
EMAIL_USER = 'officaldesk.officeofkb@gmail.com'
EMAIL_PASSWORD = 'krishna567@V'
TWILIO_SID = '00000000000'
TWILIO_TOKEN = '00000000000'
TWILIO_PHONE_NUMBER = '00000000000'
YOUR_PHONE_NUMBER = '0000000000'
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
def get_historical_prices(symbol, window):
    stock = yf.Ticker(symbol) # Adjust period based on your needs (e.g., '1d', '5d', '1m') 
    historical_data = stock.history(period="1m") # Assuming 'Close' is the closing price column
    closing_prices = historical_data['Close']
    return closing_prices[-window:] # Return the last 'window' days of data

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
    options_df['oi_volume_ratio'] = options_df['openInterest'] / options_df['volume'] # Handle division by zero (consider replacing with np.nan for missing values) 
    options_df.loc[options_df['volume'] == 0, 'oi_volume_ratio'] = np.NAN
    underlying_symbol = options_df['underlyingSymbol'].iloc[0] # Assuming first row has symbol
    historical_prices = get_historical_prices(underlying_symbol, window=20)
    options_df['historical_volatility'] = calculate_historical_volatility(historical_prices)
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
def calculate_oi_volume_ratio(data):
    oi = data['open_interest']
    volume = data['volume']
    oi_volume_ratio = oi / volume # Assuming data is a dictionary
    return oi_volume_ratio


def train_predictive_model(data):
    features = data[['impliedVolatility', 'volume', 'openInterest', 'price_zscore',
                     'bid_ask_spread', 'historical_volatility', 'rsi', 'skewness', 'kurtosis',
                     'delta', 'gamma', 'theta', 'vega']]
    target = data['price_change'].apply(lambda x: 1 if x > 0.5 else 0)  # Example threshold for significant change

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    clf_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    clf_svm = SVC(kernel='rbf', random_state=42)
    clf_xgb = XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

    model_nn = Sequential()
    model_nn.add(Dense(128, activation='relu', input_shape=(features.shape[1],)))
    model_nn.add(Dense(64, activation='relu'))
    model_nn.add(Dense(1, activation='sigmoid'))

    pipeline_rf = Pipeline([('scaler', scaler), ('clf', clf_rf)])
    pipeline_gb = Pipeline([('scaler', scaler), ('clf', clf_gb)])
    pipeline_svm = Pipeline([('scaler', scaler), ('clf', clf_svm)])
    pipeline_xgb = Pipeline([('scaler', scaler), ('clf', clf_xgb)])

    model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train models
    pipeline_rf.fit(X_train, y_train)
    pipeline_gb.fit(X_train, y_train)
    pipeline_svm.fit(X_train, y_train)
    pipeline_xgb.fit(X_train, y_train)

    model_nn.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate models
    y_pred_rf = pipeline_rf.predict(X_test)
    y_pred_gb = pipeline_gb.predict(X_test)
    y_pred_svm = pipeline_svm.predict(X_test)
    y_pred_xgb = pipeline_xgb.predict(X_test)

    y_pred_nn = model_nn.predict(X_test)
    nn_accuracy = accuracy_score(y_test, y_pred_nn)
    nn_precision = precision_score(y_test, y_pred_nn)
    nn_recall = recall_score(y_test, y_pred_nn)

    xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
    xgb_precision = precision_score(y_test, y_pred_xgb)
    xgb_recall = recall_score(y_test, y_pred_xgb)

    svm_accuracy = accuracy_score(y_test, y_pred_svm)
    svm_precision = precision_score(y_test, y_pred_svm)
    svm_recall = recall_score(y_test, y_pred_svm)

    logging.info(f"Random Forest - Accuracy: {accuracy_score(y_test, y_pred_rf)}, Precision: {precision_score(y_test, y_pred_rf)}, Recall: {recall_score(y_test, y_pred_rf)}")
    logging.info(f"Gradient Boosting - Accuracy: {accuracy_score(y_test, y_pred_gb)}, Precision: {precision_score(y_test, y_pred_gb)}, Recall: {recall_score(y_test, y_pred_rf)}")
    logging.info(f"Gradient Boosting - Accuracy: {accuracy_score(y_test, y_pred_gb)}, Precision: {precision_score(y_test, y_pred_gb)}, Recall: {recall_score(y_test, y_pred_gb)}")
    logging.info(f"Neural Network - Accuracy: {nn_accuracy}, Precision: {nn_precision}, Recall: {nn_recall}")
    logging.info(f"XGBoost - Accuracy: {xgb_accuracy}, Precision: {xgb_precision}, Recall: {xgb_recall}")
    logging.info(f"SVM - Accuracy: {svm_accuracy}, Precision: {svm_precision}, Recall: {svm_recall}")

    # Save the best model
    best_models = [pipeline_rf, pipeline_gb, pipeline_svm, pipeline_xgb, model_nn]
    best_model = max(best_models, key=lambda x: accuracy_score(y_test, x.predict(X_test))) # Save the best model dump(best_model, MODEL_FILE)
    dump(best_model, MODEL_FILE)

def load_predictive_model():
    return load(MODEL_FILE)

def make_predictions(options_df, model):
    features = options_df[['impliedVolatility', 'volume', 'openInterest', 'price_zscore',
                           'bid_ask_spread', 'historical_volatility', 'rsi', 'skewness', 'kurtosis',
                           'delta', 'gamma', 'theta', 'vega']]
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


        except Exception as e:
            logging.error(f"Error in main loop: {e}")

        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

