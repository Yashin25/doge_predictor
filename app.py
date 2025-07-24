import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from data import get_doge_data, add_onchain_and_fear_greed, get_twitter_sentiment
from model import train_classifier, prepare_lstm_data, train_lstm, optimize_rf, train_ensemble, optimize_lstm
from backtest import backtest_plot
from db_utils import save_predictions_to_gsheet

st.set_page_config(page_title="Predykcja kryptowalut", layout="wide")

# --- Ustawienia użytkownika ---
st.sidebar.header("Ustawienia")
symbol = st.sidebar.selectbox(
    "Kryptowaluta", ["DOGEUSDC", "BTCUSDC", "ETHUSDC"]
)
interval = st.sidebar.selectbox("Interwał", ["15m", "30m", "1h", "4h", "1d"])
limit = st.sidebar.slider("Liczba świec", 200, 1000, 500)
features_all = [
    'rsi', 'macd', 'ema', 'sma', 'atr', 'bb_bbm', 'bb_bbh', 'bb_bbl', 'cci', 'adx',
    'btc_close', 'eth_close', 'btc_return', 'eth_return',
    'rolling_mean_10', 'rolling_std_10', 'rolling_min_10', 'rolling_max_10', 'rolling_vol_10', 'rolling_volatility_10',
    'hour', 'dayofweek', 'month', 'is_weekend', 'bullish_engulfing'
] + ['btc_active_addresses', 'fear_greed', 'lunarcrush_sentiment']
features = st.sidebar.multiselect(
    "Wskaźniki do modelu", features_all, default=features_all[:8] + ['btc_close', 'eth_close', 'btc_return', 'eth_return']
)
lstm_features = ['close'] + features

st.sidebar.subheader("RandomForest")
n_estimators = st.sidebar.slider("Liczba drzew", 50, 300, 150)
max_depth = st.sidebar.slider("Maksymalna głębokość", 3, 20, 10)
if st.sidebar.button("Optymalizuj RandomForest"):
    st.session_state['optimize_rf'] = True
else:
    st.session_state['optimize_rf'] = False

st.sidebar.subheader("LSTM")
lstm_steps = st.sidebar.slider("Okno czasowe", 12, 48, 24)
lstm_epochs = st.sidebar.slider("Epoki", 5, 50, 10)

model_type = st.sidebar.selectbox(
    "Model ML",
    ["RandomForest", "Ensemble", "XGBoost", "LightGBM", "CatBoost", "SVM"]
)

# W sidebarze, np. pod wyborem modelu:
st.sidebar.subheader("Parametry strategii")
stop_loss = st.sidebar.number_input("Stop Loss (%)", min_value=0.001, max_value=0.2, value=0.02, step=0.001, format="%.3f")
take_profit = st.sidebar.number_input("Take Profit (%)", min_value=0.001, max_value=0.5, value=0.04, step=0.001, format="%.3f")

# --- Pobieranie i cachowanie danych DOGE, BTC, ETH ---
@st.cache_data
def load_data(symbol, interval, limit):
    return get_doge_data(symbol=symbol, interval=interval, limit=limit)

# --- Pobieranie i przygotowanie danych ---
df = load_data("DOGEUSDC", interval, limit)
df_btc = load_data("BTCUSDC", interval, limit)
df_eth = load_data("ETHUSDC", interval, limit)

date_min = df.index.min()
date_max = df.index.max()
date_range = st.sidebar.slider(
    "Zakres dat",
    min_value=date_min,
    max_value=date_max,
    value=(date_min, date_max),
    format="YYYY-MM-DD HH:mm"
)
start, end = date_range
df = df.loc[start:end]
df_btc = df_btc.reindex(df.index, method='nearest')
df_eth = df_eth.reindex(df.index, method='nearest')

# Dodaj ceny BTC i ETH jako nowe cechy
df['btc_close'] = df_btc['close']
df['eth_close'] = df_eth['close']
df['btc_return'] = df_btc['close'].pct_change()
df['eth_return'] = df_eth['close'].pct_change()

LUNARCRUSH_API_KEY = "fr5wqxuqqru3c09u6hr8i7mrbkmdhgymyeu6jfmh"
# Dodaj dane on-chain, Fear&Greed i LunarCrush
df = add_onchain_and_fear_greed(df, lunarcrush_api_key=LUNARCRUSH_API_KEY)

# Dodaj dane sentymentu z Twittera
twitter_sentiment_df = get_twitter_sentiment(query="dogecoin", limit=100)
if not twitter_sentiment_df.empty:
    # Dopasuj po czasie (np. do indeksu df po godzinie)
    df['twitter_sentiment'] = twitter_sentiment_df.reindex(df.index, method='ffill')['sentiment']
else:
    df['twitter_sentiment'] = 0  # lub np. None

# USUŃ WIERSZE Z BRAKAMI DANYCH DLA WYBRANYCH CECH
df = df.dropna(subset=features)

# USUŃ DUPLIKATY WEDŁUG INDEKSU (CZASU)
df = df[~df.index.duplicated(keep='first')]

if df.empty:
    st.error("Brak danych. Sprawdź połączenie z internetem lub API Binance.")
    st.stop()

# --- Przygotuj dane LSTM po przygotowaniu df ---
X_lstm, y_lstm, scaler = prepare_lstm_data(df, lstm_features, steps=lstm_steps)

# --- Trenowanie modeli (przycisk) ---
if st.sidebar.button("Trenuj modele"):
    with st.spinner("Trwa trenowanie modeli..."):
        if st.session_state.get('optimize_rf', False):
            st.info("Optymalizacja RandomForest (GridSearch)...")
            X = df[features]
            target_type = st.sidebar.selectbox("Target", ["Kierunek", "Zwrot logarytmiczny"])
            if target_type == "Kierunek":
                y = (df['close'].shift(-1) > df['close']).astype(int)
            else:
                y = np.log(df['close'].shift(-1) / df['close'])
            model_cls, acc = optimize_rf(X, y)
        else:
            if model_type == "RandomForest":
                model_cls, acc = train_classifier(df, features, n_estimators, max_depth)
            elif model_type == "Ensemble":
                X_train = df[features]
                y_train = (df['close'].shift(-1) > df['close']).astype(int)
                model_cls = train_ensemble(X_train, y_train)
                acc = None
            elif model_type == "XGBoost":
                import xgboost as xgb
                X_train = df[features]
                y_train = (df['close'].shift(-1) > df['close']).astype(int)
                model_cls = xgb.XGBClassifier(eval_metric='logloss')
                model_cls.fit(X_train, y_train)
                acc = None
            elif model_type == "LightGBM":
                import lightgbm as lgb
                X_train = df[features]
                y_train = (df['close'].shift(-1) > df['close']).astype(int)
                model_cls = lgb.LGBMClassifier()
                model_cls.fit(X_train, y_train)
                acc = None
            elif model_type == "CatBoost":
                import catboost as cb
                X_train = df[features]
                y_train = (df['close'].shift(-1) > df['close']).astype(int)
                model_cls = cb.CatBoostClassifier(verbose=0)
                model_cls.fit(X_train, y_train)
                acc = None
            elif model_type == "SVM":
                from sklearn.svm import SVC
                X_train = df[features]
                y_train = (df['close'].shift(-1) > df['close']).astype(int)
                model_cls = SVC(probability=True)
                model_cls.fit(X_train, y_train)
                acc = None
        X_lstm, y_lstm, scaler = prepare_lstm_data(df, lstm_features, steps=lstm_steps)
        model_lstm, loss_history, val_loss_history = train_lstm(X_lstm, y_lstm, epochs=lstm_epochs)
        with open(f"{symbol}_rf_model.pkl", "wb") as f:
            pickle.dump((model_cls, acc, features), f)
        with open(f"{symbol}_lstm_model.pkl", "wb") as f:
            pickle.dump((model_lstm, scaler, lstm_features, lstm_steps, loss_history, val_loss_history), f)
        st.success("Modele wytrenowane i zapisane!")
else:
    try:
        with open(f"{symbol}_rf_model.pkl", "rb") as f:
            model_cls, acc, features = pickle.load(f)
        with open(f"{symbol}_lstm_model.pkl", "rb") as f:
           model_lstm, scaler, lstm_features, lstm_steps, loss_history, val_loss_history = pickle.load(f)
    except Exception:
        st.warning("Najpierw wytrenuj modele!")
        st.stop()

# --- Interfejs główny ---
st.title("📈 Predykcja kryptowalut")
tab1, tab2, tab3, tab4 = st.tabs(["Klasyfikacja", "LSTM", "Backtest", "Eksport"])

with tab1:
    st.subheader("📊 Klasyfikator: wzrost / spadek")
    if acc is not None:
        st.write(f"Skuteczność: **{acc:.2%}**")
    else:
        st.write("Skuteczność: niedostępna dla modelu Ensemble.")
    last = df[features].iloc[[-1]]
    pred = model_cls.predict(last)[0]
    # Obsługa predict_proba dla modeli, które ją mają
    if hasattr(model_cls, "predict_proba"):
        prob = model_cls.predict_proba(last)[0][pred]
    else:
        prob = np.nan
    st.metric("Predykcja:", "WZROST" if pred else "SPADEK", delta=f"{prob*100:.2f}%")
    if prob > 0.9:
        st.success("ALERT: Wysokie prawdopodobieństwo wzrostu!")
    # Wizualizacja predykcji historycznych
    preds = model_cls.predict(df[features])
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df.index, df['close'], label="Cena", linewidth=2)
    ax.scatter(df.index, df['close'], c=preds, cmap='bwr', alpha=0.3, label="Predykcja (0/1)")
    ax.scatter(df.index[-1], df['close'].iloc[-1], color='red', s=60, zorder=5, label="Aktualny kurs")
    ax.set_title("Predykcja modelu na tle ceny", fontsize=16)
    ax.set_xlabel("Data", fontsize=12)
    ax.set_ylabel("Cena", fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    fig.autofmt_xdate()
    plt.tight_layout()
    st.pyplot(fig)

    # Heatmapa korelacji cech
    st.subheader("🟧 Korelacja cech (heatmapa)")
    import seaborn as sns
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    corr = df[features].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax_corr, annot_kws={"size":10})
    ax_corr.set_title("Korelacja cech", fontsize=16)
    plt.tight_layout()
    st.pyplot(fig_corr)

    # Wykres świecowy (candlestick)
    st.subheader("🕯️ Wykres świecowy (candlestick)")
    import plotly.graph_objects as go
    fig_candle = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close']
    )])
    fig_candle.update_layout(
        xaxis_rangeslider_visible=False,
        title="Wykres świecowy",
        width=1000,
        height=500,
        font=dict(size=14),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    fig_candle.update_xaxes(tickangle=0, tickfont=dict(size=12), nticks=20)
    fig_candle.update_yaxes(tickfont=dict(size=12))
    st.plotly_chart(fig_candle, use_container_width=True)

    # Interpretacja predykcji (SHAP) i ważność cech - tylko dla modeli drzewiastych
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    import xgboost as xgb
    import lightgbm as lgb
    import catboost as cb

    is_tree_model = isinstance(model_cls, (RandomForestClassifier, xgb.XGBClassifier, lgb.LGBMClassifier, cb.CatBoostClassifier))
    is_stacking = isinstance(model_cls, StackingClassifier)

    if is_tree_model or is_stacking:
        st.subheader("🧠 Interpretacja predykcji (SHAP)")
        import shap

        # Wybierz model bazowy do SHAP (dla stacking - pierwszy estimator)
        if is_stacking:
            base_model = model_cls.estimators_[0]
        else:
            base_model = model_cls

        explainer = shap.TreeExplainer(base_model)
        shap_values = explainer.shap_values(df[features])

        # Obsługa różnych kształtów shap_values
        if isinstance(shap_values, list):
            shap_to_plot = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            shap_to_plot = shap_values

        if shap_to_plot.shape[1] == len(features):
            shap.summary_plot(shap_to_plot, df[features], plot_type="bar", show=False)
            st.pyplot(plt.gcf(), bbox_inches='tight', dpi=300, pad_inches=0.1)
            plt.clf()
        else:
            st.warning("Nie można wyświetlić wykresu SHAP – liczba cech nie zgadza się z modelem.")

        # Ważność cech
        st.subheader("🔎 Ważność cech (feature importance)")
        if hasattr(base_model, "feature_importances_"):
            importances = base_model.feature_importances_
            imp_df = pd.DataFrame({'cecha': features, 'ważność': importances}).sort_values('ważność', ascending=False)
            fig_imp, ax_imp = plt.subplots()
            ax_imp.barh(imp_df['cecha'], imp_df['ważność'], color='teal')
            ax_imp.set_xlabel("Ważność")
            ax_imp.set_title("Ważność cech")
            st.pyplot(fig_imp)
        else:
            st.info("Model nie udostępnia ważności cech.")
    else:
        st.info("SHAP i ważność cech dostępne tylko dla modeli drzewiastych (np. RandomForest, XGBoost, LightGBM, CatBoost, Ensemble).")

with tab2:
    st.subheader("🔮 LSTM - Prognoza ceny na kilka kroków do przodu")
    X_lstm, _, scaler = prepare_lstm_data(df, lstm_features, steps=lstm_steps)
    forecast_steps = st.number_input("Ile kroków do przodu przewidzieć?", min_value=1, max_value=20, value=5)
    if len(X_lstm) > 0:
        last_sequence = X_lstm[-1].copy()
        future_preds = []
        for _ in range(forecast_steps):
            pred = model_lstm.predict(last_sequence.reshape(1, -1, X_lstm.shape[2]))[0, 0]
            new_row = last_sequence[-1].copy()
            new_row[0] = pred
            last_sequence = np.vstack([last_sequence[1:], new_row])
            scaled = new_row.copy()
            inv = scaler.inverse_transform([scaled])
            future_preds.append(inv[0, 0])

        last_time = df.index[-1]
        step = df.index[-1] - df.index[-2]
        future_idx = [last_time + step * (i + 1) for i in range(forecast_steps)]

        fig2, ax2 = plt.subplots()
        ax2.plot(df.index, df['close'], label="Cena rzeczywista")
        ax2.plot(future_idx, future_preds, color='green', marker='o', linestyle='--', label="Prognoza LSTM")
        ax2.scatter(df.index[-1], df['close'].iloc[-1], color='red', s=40, zorder=5, label="Aktualny kurs")
        ax2.legend()
        st.pyplot(fig2)

        st.subheader("📉 Strata treningowa (loss) LSTM")
        epochs_range = range(1, len(loss_history) + 1)
        fig_loss, ax_loss = plt.subplots()
        ax_loss.plot(epochs_range, loss_history, marker='o', label="Train Loss")
        if val_loss_history is not None and len(val_loss_history) == len(loss_history):
            ax_loss.plot(epochs_range, val_loss_history, marker='s', linestyle='--', label="Validation Loss")
        ax_loss.set_xlabel("Epoka")
        ax_loss.set_ylabel("Strata (loss)")
        ax_loss.set_title("Postęp trenowania LSTM")
        ax_loss.legend()
        st.pyplot(fig_loss)

        st.write("Prognozowane ceny:", [f"{p:.4f} USDC" for p in future_preds])
    else:
        st.warning("Za mało danych do predykcji LSTM.")

    st.subheader("📊 Rozkład błędów predykcji LSTM")
    preds_lstm = []
    for i in range(len(X_lstm)):
        pred = model_lstm.predict(X_lstm[i].reshape(1, -1, X_lstm.shape[2]))[0, 0]
        last_row = df[lstm_features].iloc[i + lstm_steps].values
        scaled = scaler.transform([last_row])
        scaled[0, 0] = float(pred)
        inv = scaler.inverse_transform(scaled)
        preds_lstm.append(inv[0, 0])
    true_lstm = df['close'].iloc[lstm_steps: lstm_steps + len(preds_lstm)].values
    errors = np.array(preds_lstm) - np.array(true_lstm)
    fig_err, ax_err = plt.subplots(figsize=(10, 5))
    ax_err.hist(errors, bins=30, color='orange', alpha=0.7)
    ax_err.set_title("Rozkład błędów predykcji (LSTM)", fontsize=16)
    ax_err.set_xlabel("Błąd predykcji [USDC]", fontsize=12)
    ax_err.set_ylabel("Liczba przypadków", fontsize=12)
    ax_err.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    st.pyplot(fig_err)

    st.subheader("📊 Rozkład zwrotów")
    fig_ret, ax_ret = plt.subplots()
    ax_ret.hist(df['close'].pct_change().dropna(), bins=50, color='purple', alpha=0.7)
    ax_ret.set_title("Rozkład zwrotów DOGE")
    st.pyplot(fig_ret)

with tab3:
    st.subheader("📈 Backtest strategii")
    backtest_plot(df, model_cls, features=features, stop_loss=stop_loss, take_profit=take_profit)

with tab4:
    st.subheader("📤 Eksport danych")
    st.download_button("Pobierz dane jako CSV", df.to_csv().encode(), file_name=f"{symbol}_data.csv")
    preds = model_cls.predict(df[features])
    df_export = df.copy()
    df_export['pred_rf'] = preds

    # Dodaj kolumnę success: 1 jeśli predykcja trafiona, 0 jeśli nie
    # Zakładamy, że predykcja dotyczy wzrostu (1) lub spadku (0) ceny w kolejnym kroku
    df_export['close_next'] = df_export['close'].shift(-1)
    df_export['success'] = ((df_export['pred_rf'] == 1) & (df_export['close_next'] > df_export['close'])) | \
                           ((df_export['pred_rf'] == 0) & (df_export['close_next'] < df_export['close']))
    df_export['success'] = df_export['success'].astype(int)
    st.download_button("Pobierz predykcje RF", df_export.to_csv().encode(), file_name=f"{symbol}_rf_preds.csv")

    # --- ZAPIS DO GOOGLE SHEETS ---
    if st.button("Zapisz predykcje do Google Sheets"):
        save_predictions_to_gsheet(
            df_export[['pred_rf', 'close', 'close_next', 'success']],
            sheet_name="Predictions",
            worksheet_name="Sheet1",
            creds_path="predykcje-769f705f09f5.json"   # <-- TAK JAK NAZWY TWOJEGO PLIKU!
        )
        st.success("Predykcje zapisane do Google Sheets!")

st.sidebar.info("Projekt: predykcja kryptowalut z ML i Streamlit.")

st.write("df_btc.columns:", df_btc.columns)
st.write("df_btc.head():", df_btc.head())
if df_btc.empty:
    st.error("Brak danych BTC!")
    st.stop()
if 'close' not in df_btc.columns:
    st.error("Brak kolumny 'close' w df_btc! Kolumny to: " + str(df_btc.columns))
    st.stop()
df['btc_close'] = df_btc['close']
