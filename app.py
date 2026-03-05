import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import MACD, EMAIndicator, SMAIndicator
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════
st.set_page_config(
    page_title="Algo Trading Backtester",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Algorithmic Trading Backtester")
st.markdown("*Test trading strategies on real market data before risking real money*")
st.markdown("---")


# ════════════════════════════════
# SIDEBAR — USER INPUTS
# ════════════════════════════════
st.sidebar.header("⚙️ Settings")

STOCKS = {
    'SAIL': 'SAIL.NS',
    'Bandhan Bank': 'BANDHANBNK.NS',
    'Ashok Leyland': 'ASHOKLEY.NS',
    'Manappuram': 'MANAPPURAM.NS',
    'Voltas': 'VOLTAS.NS',
    'Deepak Nitrite': 'DEEPAKNTR.NS',
    'MRF': 'MRF.NS',
    'Biocon': 'BIOCON.NS',
    'Coforge': 'COFORGE.NS',
    'Crompton': 'CROMPTON.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'TCS': 'TCS.NS',
    'Reliance': 'RELIANCE.NS',
    'Infosys': 'INFY.NS',
    'ITC': 'ITC.NS',
}

STRATEGY_NAMES = [
    'SMA Crossover',
    'EMA Crossover',
    'RSI Mean Reversion',
    'Bollinger Breakout',
    'Bollinger Mean Reversion',
    'MACD Crossover',
    'Stochastic',
    'Volume Breakout',
    'Triple EMA',
    'Multi-Factor (RSI+MACD+EMA)',
]

selected_stock = st.sidebar.selectbox(
    "📊 Select Stock", list(STOCKS.keys()))
selected_ticker = STOCKS[selected_stock]

selected_strategy = st.sidebar.selectbox(
    "🎯 Select Strategy", STRATEGY_NAMES)

period = st.sidebar.selectbox(
    "📅 Backtest Period",
    ['1y', '2y', '3y', '5y'],
    index=2
)

initial_capital = st.sidebar.number_input(
    "💰 Initial Capital (Rs.)",
    min_value=100000,
    max_value=10000000,
    value=1000000,
    step=100000
)

commission = st.sidebar.slider(
    "📋 Commission (%)",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05
) / 100

compare_all = st.sidebar.checkbox(
    "📊 Compare All Strategies", value=False)

run_button = st.sidebar.button(
    "🚀 Run Backtest", type="primary",
    use_container_width=True)


# ════════════════════════════════
# FIX YFINANCE
# ════════════════════════════════
def fix_yf(df):
    if df is None or len(df) == 0:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


# ════════════════════════════════
# STRATEGY FUNCTIONS
# ════════════════════════════════
def sma_crossover(df, fast=20, slow=50):
    close = df['Close'].squeeze()
    sma_fast = SMAIndicator(close, fast).sma_indicator()
    sma_slow = SMAIndicator(close, slow).sma_indicator()
    signal = pd.Series(0, index=df.index)
    signal[sma_fast > sma_slow] = 1
    signal[sma_fast < sma_slow] = -1
    indicators = {'SMA Fast': sma_fast, 'SMA Slow': sma_slow}
    return signal, indicators

def ema_crossover(df, fast=12, slow=26):
    close = df['Close'].squeeze()
    ema_fast = EMAIndicator(close, fast).ema_indicator()
    ema_slow = EMAIndicator(close, slow).ema_indicator()
    signal = pd.Series(0, index=df.index)
    signal[ema_fast > ema_slow] = 1
    signal[ema_fast < ema_slow] = -1
    indicators = {'EMA Fast': ema_fast, 'EMA Slow': ema_slow}
    return signal, indicators

def rsi_mean_reversion(df, period=14, oversold=30, overbought=70):
    close = df['Close'].squeeze()
    rsi = RSIIndicator(close, period).rsi()
    signal = pd.Series(0, index=df.index)
    signal[rsi < oversold] = 1
    signal[rsi > overbought] = -1
    indicators = {'RSI': rsi}
    return signal, indicators

def bollinger_breakout(df, window=20, std=2):
    close = df['Close'].squeeze()
    bb = BollingerBands(close, window, std)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    signal = pd.Series(0, index=df.index)
    signal[close > upper] = 1
    signal[close < lower] = -1
    indicators = {'Upper Band': upper, 'Lower Band': lower,
                  'Middle Band': bb.bollinger_mavg()}
    return signal, indicators

def bollinger_mean_reversion(df, window=20, std=2):
    close = df['Close'].squeeze()
    bb = BollingerBands(close, window, std)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    signal = pd.Series(0, index=df.index)
    signal[close < lower] = 1
    signal[close > upper] = -1
    indicators = {'Upper Band': upper, 'Lower Band': lower,
                  'Middle Band': bb.bollinger_mavg()}
    return signal, indicators

def macd_strategy(df):
    close = df['Close'].squeeze()
    macd = MACD(close)
    macd_line = macd.macd()
    signal_line = macd.macd_signal()
    signal = pd.Series(0, index=df.index)
    signal[macd_line > signal_line] = 1
    signal[macd_line < signal_line] = -1
    indicators = {'MACD': macd_line, 'Signal': signal_line,
                  'Histogram': macd.macd_diff()}
    return signal, indicators

def stochastic_strategy(df):
    close = df['Close'].squeeze()
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    stoch = StochasticOscillator(high, low, close, 14, 3)
    k_line = stoch.stoch()
    d_line = stoch.stoch_signal()
    signal = pd.Series(0, index=df.index)
    signal[(k_line > d_line) & (k_line < 20)] = 1
    signal[(k_line < d_line) & (k_line > 80)] = -1
    indicators = {'%K': k_line, '%D': d_line}
    return signal, indicators

def volume_breakout(df):
    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    vol_avg = volume.rolling(20).mean()
    vol_spike = volume > (vol_avg * 2.0)
    ret = close.pct_change()
    signal = pd.Series(0, index=df.index)
    signal[vol_spike & (ret > 0.02)] = 1
    signal[vol_spike & (ret < -0.02)] = -1
    indicators = {'Volume': volume, 'Avg Volume': vol_avg}
    return signal, indicators

def triple_ema(df):
    close = df['Close'].squeeze()
    ema_f = EMAIndicator(close, 5).ema_indicator()
    ema_m = EMAIndicator(close, 20).ema_indicator()
    ema_s = EMAIndicator(close, 50).ema_indicator()
    signal = pd.Series(0, index=df.index)
    signal[(ema_f > ema_m) & (ema_m > ema_s)] = 1
    signal[(ema_f < ema_m) & (ema_m < ema_s)] = -1
    indicators = {'EMA 5': ema_f, 'EMA 20': ema_m,
                  'EMA 50': ema_s}
    return signal, indicators

def multi_factor(df):
    close = df['Close'].squeeze()
    rsi = RSIIndicator(close, 14).rsi()
    rsi_sig = pd.Series(0, index=df.index)
    rsi_sig[rsi < 35] = 1
    rsi_sig[rsi > 65] = -1

    macd = MACD(close)
    macd_hist = macd.macd_diff()
    macd_sig = pd.Series(0, index=df.index)
    macd_sig[macd_hist > 0] = 1
    macd_sig[macd_hist < 0] = -1

    ema20 = EMAIndicator(close, 20).ema_indicator()
    ema50 = EMAIndicator(close, 50).ema_indicator()
    ema_sig = pd.Series(0, index=df.index)
    ema_sig[ema20 > ema50] = 1
    ema_sig[ema20 < ema50] = -1

    total = rsi_sig + macd_sig + ema_sig
    signal = pd.Series(0, index=df.index)
    signal[total >= 2] = 1
    signal[total <= -2] = -1

    indicators = {'RSI': rsi, 'MACD Hist': macd_hist,
                  'EMA 20': ema20, 'EMA 50': ema50}
    return signal, indicators


STRATEGY_MAP = {
    'SMA Crossover': sma_crossover,
    'EMA Crossover': ema_crossover,
    'RSI Mean Reversion': rsi_mean_reversion,
    'Bollinger Breakout': bollinger_breakout,
    'Bollinger Mean Reversion': bollinger_mean_reversion,
    'MACD Crossover': macd_strategy,
    'Stochastic': stochastic_strategy,
    'Volume Breakout': volume_breakout,
    'Triple EMA': triple_ema,
    'Multi-Factor (RSI+MACD+EMA)': multi_factor,
}


# ════════════════════════════════
# BACKTESTING ENGINE
# ════════════════════════════════
def run_backtest(df, signals, capital, comm):
    bt = df.copy()
    bt['signal'] = signals.fillna(0)
    bt['position'] = bt['signal'].shift(1).fillna(0)
    bt['market_return'] = bt['Close'].squeeze().pct_change()
    bt['strategy_return'] = bt['position'] * bt['market_return']

    bt['trade'] = bt['position'].diff().abs()
    bt['cost'] = bt['trade'] * (comm + 0.0005)
    bt['strategy_return'] = bt['strategy_return'] - bt['cost']

    bt['equity'] = capital * (1 + bt['strategy_return']).cumprod()
    bt['buy_hold'] = capital * (1 + bt['market_return']).cumprod()

    bt['peak'] = bt['equity'].cummax()
    bt['drawdown'] = (bt['equity'] - bt['peak']) / bt['peak'] * 100

    return bt


def calculate_metrics(bt, capital):
    strat_ret = bt['strategy_return'].dropna()
    days = len(strat_ret)
    years = days / 252

    total_return = float(
        (bt['equity'].iloc[-1] / capital - 1) * 100)
    bh_return = float(
        (bt['buy_hold'].iloc[-1] / capital - 1) * 100)

    ann_return = float(
        ((1 + total_return/100)**(1/max(years, 0.1)) - 1) * 100)
    ann_vol = float(strat_ret.std() * np.sqrt(252) * 100)

    rf_daily = 6.5 / 252 / 100
    sharpe = float(
        (strat_ret.mean() - rf_daily) / strat_ret.std()
        * np.sqrt(252)
    ) if strat_ret.std() > 0 else 0

    downside = strat_ret[strat_ret < 0]
    sortino = float(
        (strat_ret.mean() - rf_daily) / downside.std()
        * np.sqrt(252)
    ) if len(downside) > 0 and downside.std() > 0 else 0

    max_dd = float(bt['drawdown'].min())
    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    winning = strat_ret[strat_ret > 0]
    losing = strat_ret[strat_ret < 0]
    win_rate = float(
        len(winning) / (len(winning) + len(losing)) * 100
    ) if (len(winning) + len(losing)) > 0 else 0

    gross_profit = float(winning.sum()) if len(winning) > 0 else 0
    gross_loss = float(abs(losing.sum())) if len(losing) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    num_trades = int(bt['trade'].sum() / 2)
    final = float(bt['equity'].iloc[-1])

    return {
        'total_return': round(total_return, 2),
        'buy_hold_return': round(bh_return, 2),
        'excess_return': round(total_return - bh_return, 2),
        'ann_return': round(ann_return, 2),
        'ann_volatility': round(ann_vol, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'calmar': round(calmar, 3),
        'max_drawdown': round(max_dd, 2),
        'win_rate': round(win_rate, 1),
        'profit_factor': round(profit_factor, 3),
        'num_trades': num_trades,
        'final_equity': round(final),
        'profit_loss': round(final - capital),
        'best_day': round(float(strat_ret.max()) * 100, 2),
        'worst_day': round(float(strat_ret.min()) * 100, 2),
    }


# ════════════════════════════════
# MAIN APP LOGIC
# ════════════════════════════════
if run_button:

    with st.spinner(f"Downloading {selected_stock} data..."):
        df = yf.download(selected_ticker, period=period,
                        progress=False)
        df = fix_yf(df)

    if df is None or len(df) < 50:
        st.error("❌ Not enough data. Try different stock/period.")
    else:
        st.success(f"✅ Loaded {len(df)} days of {selected_stock} data")

        if not compare_all:
            # ══════════════════════════════
            # SINGLE STRATEGY BACKTEST
            # ══════════════════════════════
            with st.spinner(f"Running {selected_strategy}..."):
                strat_func = STRATEGY_MAP[selected_strategy]
                signals, indicators = strat_func(df)
                bt = run_backtest(df, signals, initial_capital,
                                commission)
                metrics = calculate_metrics(bt, initial_capital)

            # ── HEADER METRICS ──
            st.markdown("---")
            st.subheader(f"📊 {selected_stock} — {selected_strategy}")

            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Total Return",
                       f"{metrics['total_return']:+.1f}%",
                       f"vs B&H: {metrics['excess_return']:+.1f}%")
            col2.metric("Sharpe Ratio",
                       f"{metrics['sharpe']:.3f}")
            col3.metric("Max Drawdown",
                       f"{metrics['max_drawdown']:.1f}%")
            col4.metric("Win Rate",
                       f"{metrics['win_rate']:.1f}%")
            col5.metric("Profit/Loss",
                       f"Rs.{metrics['profit_loss']:,}")

            # ── EQUITY CURVE ──
            st.markdown("---")
            st.subheader("💰 Equity Curve")

            fig_equity = go.Figure()
            fig_equity.add_trace(go.Scatter(
                x=bt.index, y=bt['equity'],
                name='Strategy',
                line=dict(color='#2196F3', width=2)
            ))
            fig_equity.add_trace(go.Scatter(
                x=bt.index, y=bt['buy_hold'],
                name='Buy and Hold',
                line=dict(color='#FF9800', width=2,
                         dash='dash')
            ))
            fig_equity.update_layout(
                title=f"{selected_stock} — {selected_strategy} vs Buy and Hold",
                xaxis_title="Date",
                yaxis_title="Portfolio Value (Rs.)",
                height=500,
                template="plotly_white",
                hovermode="x unified"
            )
            st.plotly_chart(fig_equity, use_container_width=True)

            # ── DRAWDOWN CHART ──
            st.subheader("📉 Drawdown")

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=bt.index, y=bt['drawdown'],
                fill='tozeroy',
                name='Drawdown',
                line=dict(color='#f44336', width=1),
                fillcolor='rgba(244, 67, 54, 0.3)'
            ))
            fig_dd.update_layout(
                title="Portfolio Drawdown",
                xaxis_title="Date",
                yaxis_title="Drawdown (%)",
                height=350,
                template="plotly_white"
            )
            st.plotly_chart(fig_dd, use_container_width=True)

            # ── MONTHLY RETURNS HEATMAP ──
            st.subheader("📅 Monthly Returns Heatmap")

            monthly_ret = bt['strategy_return'].resample('M').apply(
                lambda x: (1 + x).prod() - 1) * 100

            monthly_df = pd.DataFrame({
                'Year': monthly_ret.index.year,
                'Month': monthly_ret.index.month_name().str[:3],
                'Return': monthly_ret.values
            })

            pivot = monthly_df.pivot_table(
                index='Year', columns='Month',
                values='Return', aggfunc='first')

            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                          'Jun', 'Jul', 'Aug', 'Sep', 'Oct',
                          'Nov', 'Dec']
            available = [m for m in month_order
                        if m in pivot.columns]
            pivot = pivot[available]

            fig_heat = px.imshow(
                pivot.values,
                x=available,
                y=[str(y) for y in pivot.index],
                color_continuous_scale='RdYlGn',
                aspect='auto',
                labels=dict(color="Return %")
            )
            fig_heat.update_layout(
                title="Monthly Returns (%)",
                height=300
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # ── PRICE CHART WITH SIGNALS ──
            st.subheader("📊 Price Chart with Signals")

            close_price = df['Close'].squeeze()
            buy_signals = bt[bt['signal'] == 1]
            sell_signals = bt[bt['signal'] == -1]

            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=df.index, y=close_price,
                name='Price',
                line=dict(color='#333', width=1.5)
            ))

            # Add indicator overlays
            for ind_name, ind_data in indicators.items():
                if ind_name not in ['RSI', 'MACD', 'MACD Hist',
                                    'Histogram', 'Signal',
                                    '%K', '%D', 'Volume',
                                    'Avg Volume']:
                    fig_price.add_trace(go.Scatter(
                        x=df.index, y=ind_data,
                        name=ind_name,
                        line=dict(width=1, dash='dot')
                    ))

            fig_price.add_trace(go.Scatter(
                x=buy_signals.index,
                y=close_price.loc[buy_signals.index],
                mode='markers', name='BUY',
                marker=dict(symbol='triangle-up',
                           size=10, color='green')
            ))
            fig_price.add_trace(go.Scatter(
                x=sell_signals.index,
                y=close_price.loc[sell_signals.index],
                mode='markers', name='SELL',
                marker=dict(symbol='triangle-down',
                           size=10, color='red')
            ))

            fig_price.update_layout(
                title=f"{selected_stock} Price with {selected_strategy} Signals",
                xaxis_title="Date",
                yaxis_title="Price (Rs.)",
                height=500,
                template="plotly_white"
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # ── DETAILED METRICS TABLE ──
            st.subheader("📋 Performance Metrics")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown("**Returns**")
                st.write(f"Total Return: {metrics['total_return']:+.2f}%")
                st.write(f"Annualized: {metrics['ann_return']:+.2f}%")
                st.write(f"Buy and Hold: {metrics['buy_hold_return']:+.2f}%")
                st.write(f"Excess Return: {metrics['excess_return']:+.2f}%")
                st.write(f"Best Day: {metrics['best_day']:+.2f}%")
                st.write(f"Worst Day: {metrics['worst_day']:.2f}%")

            with col2:
                st.markdown("**Risk**")
                st.write(f"Volatility: {metrics['ann_volatility']:.2f}%")
                st.write(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
                st.write(f"Sharpe Ratio: {metrics['sharpe']:.3f}")
                st.write(f"Sortino Ratio: {metrics['sortino']:.3f}")
                st.write(f"Calmar Ratio: {metrics['calmar']:.3f}")

            with col3:
                st.markdown("**Trading**")
                st.write(f"Total Trades: {metrics['num_trades']}")
                st.write(f"Win Rate: {metrics['win_rate']:.1f}%")
                st.write(f"Profit Factor: {metrics['profit_factor']:.3f}")
                st.write(f"Initial: Rs.{initial_capital:,}")
                st.write(f"Final: Rs.{metrics['final_equity']:,}")
                st.write(f"P&L: Rs.{metrics['profit_loss']:,}")

        else:
            # ══════════════════════════════
            # COMPARE ALL STRATEGIES
            # ══════════════════════════════
            st.markdown("---")
            st.subheader(f"📊 All Strategies — {selected_stock}")

            all_metrics = []
            all_equities = {}

            progress = st.progress(0)
            for i, (name, func) in enumerate(STRATEGY_MAP.items()):
                try:
                    signals, _ = func(df)
                    bt = run_backtest(df, signals,
                                    initial_capital, commission)
                    metrics = calculate_metrics(bt, initial_capital)
                    metrics['strategy'] = name
                    all_metrics.append(metrics)
                    all_equities[name] = bt['equity']
                except:
                    pass
                progress.progress((i + 1) / len(STRATEGY_MAP))

            progress.empty()

            if all_metrics:
                # Sort by Sharpe
                all_metrics.sort(key=lambda x: x['sharpe'],
                               reverse=True)

                # ── COMPARISON TABLE ──
                comp_df = pd.DataFrame(all_metrics)
                comp_df = comp_df[[
                    'strategy', 'total_return', 'sharpe',
                    'max_drawdown', 'win_rate',
                    'profit_factor', 'num_trades',
                    'profit_loss'
                ]]
                comp_df.columns = [
                    'Strategy', 'Return %', 'Sharpe',
                    'Max DD %', 'Win Rate %',
                    'Profit Factor', 'Trades', 'P&L (Rs.)'
                ]

                st.dataframe(
                    comp_df.style.format({
                        'Return %': '{:+.1f}',
                        'Sharpe': '{:.3f}',
                        'Max DD %': '{:.1f}',
                        'Win Rate %': '{:.1f}',
                        'Profit Factor': '{:.3f}',
                        'P&L (Rs.)': '{:,.0f}'
                    }).background_gradient(
                        subset=['Sharpe'], cmap='RdYlGn'
                    ),
                    use_container_width=True,
                    hide_index=True
                )

                # ── ALL EQUITY CURVES ──
                st.subheader("💰 All Equity Curves")

                fig_all = go.Figure()
                colors = px.colors.qualitative.Set3

                for i, (name, equity) in enumerate(
                        all_equities.items()):
                    fig_all.add_trace(go.Scatter(
                        x=equity.index, y=equity,
                        name=name,
                        line=dict(
                            color=colors[i % len(colors)],
                            width=1.5)
                    ))

                # Add buy and hold
                bh = initial_capital * (
                    1 + df['Close'].squeeze().pct_change()
                ).cumprod()
                fig_all.add_trace(go.Scatter(
                    x=bh.index, y=bh,
                    name='Buy and Hold',
                    line=dict(color='black', width=3,
                             dash='dash')
                ))

                fig_all.update_layout(
                    title=f"All Strategies vs Buy and Hold — {selected_stock}",
                    xaxis_title="Date",
                    yaxis_title="Portfolio Value (Rs.)",
                    height=600,
                    template="plotly_white",
                    hovermode="x unified"
                )
                st.plotly_chart(fig_all, use_container_width=True)

                # ── BEST STRATEGY HIGHLIGHT ──
                best = all_metrics[0]
                worst = all_metrics[-1]

                st.subheader("🏆 Results")
                col1, col2 = st.columns(2)

                with col1:
                    st.success(
                        f"**Best Strategy: {best['strategy']}**\n\n"
                        f"Return: {best['total_return']:+.1f}% | "
                        f"Sharpe: {best['sharpe']:.3f} | "
                        f"P&L: Rs.{best['profit_loss']:,}"
                    )

                with col2:
                    st.error(
                        f"**Worst Strategy: {worst['strategy']}**\n\n"
                        f"Return: {worst['total_return']:+.1f}% | "
                        f"Sharpe: {worst['sharpe']:.3f} | "
                        f"P&L: Rs.{worst['profit_loss']:,}"
                    )

                # ── SHARPE BAR CHART ──
                st.subheader("📊 Sharpe Ratio Comparison")

                sharpe_df = pd.DataFrame(all_metrics)
                colors_bar = ['green' if s > 0 else 'red'
                             for s in sharpe_df['sharpe']]

                fig_sharpe = go.Figure(data=[
                    go.Bar(
                        x=sharpe_df['strategy'],
                        y=sharpe_df['sharpe'],
                        marker_color=colors_bar
                    )
                ])
                fig_sharpe.update_layout(
                    title="Sharpe Ratio by Strategy",
                    xaxis_title="Strategy",
                    yaxis_title="Sharpe Ratio",
                    height=400,
                    template="plotly_white"
                )
                fig_sharpe.add_hline(y=0, line_dash="dash",
                                    line_color="gray")
                st.plotly_chart(fig_sharpe,
                              use_container_width=True)

else:
    # ── LANDING PAGE ──
    st.info("👈 Select a stock and strategy from the sidebar, then click **Run Backtest**")

    st.markdown("""
    ### How to Use
    1. **Select a Stock** from the dropdown
    2. **Select a Strategy** to test
    3. **Set your capital** and commission
    4. Click **Run Backtest** to see results
    5. Check **Compare All Strategies** to test all 10 at once

    ### Available Strategies
    | Strategy | Type | Description |
    |----------|------|-------------|
    | SMA Crossover | Trend | Buy when fast SMA crosses above slow |
    | EMA Crossover | Trend | Faster version of SMA crossover |
    | RSI Mean Reversion | Mean Rev | Buy oversold, sell overbought |
    | Bollinger Breakout | Breakout | Buy on upper band break |
    | Bollinger Mean Reversion | Mean Rev | Buy at lower band |
    | MACD Crossover | Momentum | MACD vs Signal line |
    | Stochastic | Momentum | K/D crossover in extreme zones |
    | Volume Breakout | Volume | Trade on volume spikes |
    | Triple EMA | Trend | Three EMA alignment |
    | Multi-Factor | Combined | RSI + MACD + EMA together |
    """)


# ── FOOTER ──
st.markdown("---")
st.markdown(
    "*Built by [Your Name] | PGDM Finance | "
    "Not financial advice — educational only*"
)
