from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import optuna

def train_classifier(df, features, n_estimators=150, max_depth=10):
    df = df.copy()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    X = df[features]
    y = df['target']

    tscv = TimeSeriesSplit(n_splits=5)
    splits = list(tscv.split(X))
    train_idx, test_idx = splits[-1]

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth))
    ])
    pipeline.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, pipeline.predict(X_test))
    return pipeline, accuracy

def prepare_lstm_data(df, features, steps=24):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features])

    X, y = [], []
    for i in range(steps, len(scaled)-1):
        X.append(scaled[i-steps:i])
        y.append(scaled[i+1][0])

    return np.array(X), np.array(y), scaler

def train_lstm(X, y, epochs=10):
    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(128, return_sequences=True),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X, y, epochs=epochs, batch_size=32, verbose=0, validation_split=0.2)
    return model, history.history['loss'], history.history['val_loss']

def optimize_rf(X, y):
    param_grid = {
        'model__n_estimators': [50, 100, 150],
        'model__max_depth': [5, 10, 15]
    }
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestClassifier())
    ])
    tscv = TimeSeriesSplit(n_splits=5)
    grid = GridSearchCV(pipeline, param_grid, cv=tscv)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_score_

def train_ensemble(X_train, y_train):
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=100)),
        ('xgb', xgb.XGBClassifier(eval_metric='logloss')),
        ('lgb', lgb.LGBMClassifier()),
        ('cat', cb.CatBoostClassifier(verbose=0)),
        ('svc', SVC(probability=True))
    ]
    ensemble = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())
    ensemble.fit(X_train, y_train)
    return ensemble

def objective(trial, X, y):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    units = trial.suggest_categorical('units', [32, 64, 128])
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    model = Sequential()
    model.add(Input(shape=(X.shape[1], X.shape[2])))
    for _ in range(n_layers):
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout))
    model.add(LSTM(units))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
    loss = model.evaluate(X, y, verbose=0)
    return loss

def optimize_lstm(X, y, n_trials=10):
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    return study.best_params