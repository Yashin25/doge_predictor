import requests
import pandas as pd
import ta

def get_doge_data(symbol="DOGEUSDT", limit=1000, interval="1h"):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        print(f"Błąd pobierania danych: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(data, columns=[
        'time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['time'] = pd.to_datetime(df['time'], unit='ms')
    df.set_index('time', inplace=True)
    # Zamień tylko wybrane kolumny na float
    for col in ['open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Wskaźniki techniczne
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=14).ema_indicator()
    df['sma'] = ta.trend.SMAIndicator(df['close'], window=14).sma_indicator()
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    # Dodatkowe wskaźniki
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=14).cci()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14).adx()

    # Rolling features
    df['rolling_mean_10'] = df['close'].rolling(window=10).mean()
    df['rolling_std_10'] = df['close'].rolling(window=10).std()
    df['rolling_min_10'] = df['close'].rolling(window=10).min()
    df['rolling_max_10'] = df['close'].rolling(window=10).max()
    df['rolling_vol_10'] = df['volume'].rolling(window=10).mean()
    df['rolling_volatility_10'] = df['close'].pct_change().rolling(window=10).std()

    # Cechy czasowe
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['is_weekend'] = (df.index.dayofweek >= 5).astype(int)

    # Formacje świecowe (przykład: bullish engulfing)
    df['bullish_engulfing'] = ((df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))).astype(int)

    df.dropna(inplace=True)
    return df

def get_twitter_sentiment(query="dogecoin", bearer_token="...", limit=100):
    import tweepy
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from transformers import pipeline

    try:
        client = tweepy.Client(bearer_token=bearer_token)
        tweets = client.search_recent_tweets(query=query, max_results=min(limit,100), tweet_fields=["created_at"])
    except tweepy.TooManyRequests:
        print("Przekroczono limit zapytań do Twitter API (429).")
        return pd.DataFrame()
    except Exception as e:
        print(f"Błąd pobierania tweetów: {e}")
        return pd.DataFrame()

    texts, times = [], []
    if tweets.data:
        for tweet in tweets.data:
            texts.append(tweet.text)
            times.append(tweet.created_at)
    if not times:
        return pd.DataFrame()

    # Analiza sentymentu FinBERT
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    sentiments = []
    for text in texts:
        result = nlp(text)[0]
        # Możesz zamienić na liczbę: 1=positive, 0=neutral, -1=negative
        if result['label'] == 'positive':
            sentiments.append(1)
        elif result['label'] == 'negative':
            sentiments.append(-1)
        else:
            sentiments.append(0)

    df = pd.DataFrame({"sentiment": sentiments}, index=pd.to_datetime(times))
    df = df.resample("1h").mean()
    return df

def add_onchain_and_fear_greed(df, lunarcrush_api_key="demo"):
    """
    Rozszerza DataFrame o:
    - btc_active_addresses (CoinMetrics, dzienne)
    - fear_greed (Alternative.me, dzienne)
    Dopasowuje po dacie (UTC, bez godziny).
    """
    # --- Pobierz aktywne adresy BTC z CoinMetrics ---
    url_cm = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
    params_cm = {
        "assets": "btc",
        "metrics": "AdrActCnt",
        "frequency": "1d",
        "page_size": 1000
    }
    r_cm = requests.get(url_cm, params=params_cm)
    if r_cm.status_code != 200:
        btc_df = pd.DataFrame()
    else:
        data = r_cm.json().get("data", [])
        btc_df = pd.DataFrame(data)
        if not btc_df.empty:
            btc_df['time'] = pd.to_datetime(btc_df['time']).dt.tz_localize(None)
            btc_df.set_index('time', inplace=True)
            btc_df['btc_active_addresses'] = pd.to_numeric(btc_df['AdrActCnt'], errors='coerce')
            btc_df = btc_df[['btc_active_addresses']]

    # --- Pobierz Fear & Greed Index z alternative.me ---
    url_fg = "https://api.alternative.me/fng/?limit=0&format=json"
    r_fg = requests.get(url_fg)
    if r_fg.status_code != 200:
        fg_df = pd.DataFrame()
    else:
        data = r_fg.json().get("data", [])
        fg_df = pd.DataFrame(data)
        if not fg_df.empty:
            fg_df['timestamp'] = pd.to_datetime(fg_df['timestamp'], unit='s').dt.tz_localize(None)
            fg_df.set_index('timestamp', inplace=True)
            fg_df['fear_greed'] = pd.to_numeric(fg_df['value'], errors='coerce')
            fg_df = fg_df[['fear_greed']]

    # --- Przygotuj indeks dzienny w df ---
    df_daily = df.copy()
    df_daily['date'] = df_daily.index.floor('D')
    df_daily.set_index('date', inplace=True)

    # --- Połącz z danymi on-chain i sentymentem ---
    if not btc_df.empty:
        df_daily = df_daily.merge(btc_df, left_index=True, right_index=True, how='left')
    else:
        df_daily['btc_active_addresses'] = None

    if not fg_df.empty:
        df_daily = df_daily.merge(fg_df, left_index=True, right_index=True, how='left')
    else:
        df_daily['fear_greed'] = None

    # --- Pobierz sentyment z LunarCrush ---
    lunar_df = get_lunarcrush_sentiment(symbol="doge", api_key=lunarcrush_api_key)
    if not lunar_df.empty:
        lunar_df = lunar_df.reindex(df_daily.index, method='ffill')
        df_daily['lunarcrush_sentiment'] = lunar_df['lunarcrush_sentiment']
    else:
        df_daily['lunarcrush_sentiment'] = None

    # Przywróć oryginalny indeks (czasowy)
    df_daily = df_daily.sort_index()
    df_daily = df_daily[~df_daily.index.duplicated(keep='first')]
    df_daily = df_daily.reindex(df.index, method='ffill')

    return df_daily

def get_lunarcrush_sentiment(symbol="doge", api_key="demo"):
    import requests
    import pandas as pd
    url = "https://api.lunarcrush.com/v2"
    params = {
        "data": "assets",
        "key": api_key,  # <-- użyj swojego klucza!
        "symbol": symbol.upper()
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Błąd pobierania LunarCrush: {e}")
        return pd.DataFrame()
    data = r.json().get("data", [])
    if not data:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "lunarcrush_sentiment": data[0].get("galaxy_score"),
        "date": pd.to_datetime("now")
    }])
    df.set_index("date", inplace=True)
    return df