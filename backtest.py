import streamlit as st
import vectorbt as vbt
import numpy as np
import matplotlib.pyplot as plt

def backtest_plot(df, model, features=None, commission=0.001, slippage=0.001, stop_loss=0.02, take_profit=0.04):
    if df.empty:
        st.warning("Brak danych do backtestu.")
        return

    if features is None:
        features = df.columns.tolist()
    df = df.copy()
    df['pred'] = model.predict(df[features])

    # Sygnały long/short
    entries = df['pred'] == 1
    exits = df['pred'].shift(1, fill_value=0) == 1
    short_entries = df['pred'] == 0
    short_exits = df['pred'].shift(1, fill_value=1) == 0

    # Backtest vectorbt
    pf = vbt.Portfolio.from_signals(
        close=df['close'],
        entries=entries,
        exits=exits,
        short_entries=short_entries,
        short_exits=short_exits,
        fees=commission + slippage,
        sl_stop=stop_loss,
        tp_stop=take_profit,
        direction='both'
    )

    # Wykres pozycji (entry/exit)
    fig = pf.plot(subplots=['orders', 'trades'])
    st.plotly_chart(fig, use_container_width=True)

    # Metryki
    st.subheader("📊 Metryki strategii (vectorbt)")
    stats = pf.stats()
    # CAGR
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (pf.total_return() + 1) ** (1 / years) - 1 if years > 0 else np.nan
    # Max Drawdown
    max_dd = pf.max_drawdown()
    # MAR ratio
    mar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    st.write(stats)
    st.write(f"**CAGR:** {cagr:.2%}")
    st.write(f"**Max Drawdown:** {max_dd:.2%}")
    st.write(f"**MAR ratio (CAGR/MaxDD):** {mar:.2f}")

    # Equity curve
    st.subheader("📈 Krzywa kapitału")
    fig2 = pf.plot()
    st.plotly_chart(fig2, use_container_width=True)