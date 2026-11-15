"""
RegimeRiskStrategy - stage 1 (rule-based, no ML yet)

Layers:
  1) Regime filter (1h): EMA50 vs EMA200 + ADX. No trades in chop.
  2) Signal (15m): RSI pullback re-cross in the direction of the regime.
  3) Risk: leverage 3x isolated, ATR-based trailing stop (hard cap),
     time-decayed ROI, liquidation-proximity guard, protections.

NOTE: untested template. Backtest + lookahead/recursive analysis first.
Stoploss / ROI values are "profit ratio on stake" => already include leverage.
"""
from typing import Optional

import talib.abstract as ta
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative


class RegimeRiskStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True
    timeframe = "15m"
    process_only_new_candles = True
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # 1h EMA200 = 200h = 800 candles of 15m (+ buffer)
    startup_candle_count = 850

    # ---- risk parameters -------------------------------------------------
    target_leverage = 3.0
    adx_min = 20            # regime: minimum 1h ADX to allow trades
    rsi_long = 35           # long: RSI crosses back above this
    rsi_short = 65          # short: RSI crosses back below this
    min_atr_pct = 0.0015    # skip dead markets (fees would dominate)
    atr_stop_mult = 2.0     # stop distance in ATRs (price terms)
    min_stop = 0.03         # min stop, profit ratio (3% = ~1% price at 3x)
    liq_buffer = 0.05       # exit if price within 5% of liquidation price

    # Hard maximum loss (profit ratio). 3x => ~3.3% price move.
    # Liquidation at 3x is ~33% away, so this is far inside it.
    stoploss = -0.10
    use_custom_stoploss = True
    trailing_stop = False

    # Time-decayed take profit (profit ratio, minutes)
    minimal_roi = {"0": 0.12, "360": 0.06, "1080": 0.0}

    @property
    def protections(self):
        # Only active in backtests with --enable-protections
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 4},
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 96,
                "trade_limit": 3,
                "stop_duration_candles": 96,
                "only_per_pair": False,
            },
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 288,
                "trade_limit": 10,
                "stop_duration_candles": 192,
                "max_allowed_drawdown": 0.10,
            },
        ]

    # ---- indicators ------------------------------------------------------
    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema_fast"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema_slow"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        return dataframe

    # ---- signals ---------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        regime_up = (dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"]) & (
            dataframe["adx_1h"] > self.adx_min
        )
        regime_down = (dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"]) & (
            dataframe["adx_1h"] > self.adx_min
        )
        active = (dataframe["atr_pct"] > self.min_atr_pct) & (dataframe["volume"] > 0)

        long_cross = (dataframe["rsi"] > self.rsi_long) & (
            dataframe["rsi"].shift(1) <= self.rsi_long
        )
        short_cross = (dataframe["rsi"] < self.rsi_short) & (
            dataframe["rsi"].shift(1) >= self.rsi_short
        )

        dataframe.loc[regime_up & active & long_cross, ["enter_long", "enter_tag"]] = (
            1,
            "pullback_long",
        )
        dataframe.loc[regime_down & active & short_cross, ["enter_short", "enter_tag"]] = (
            1,
            "pullback_short",
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        trend_up = dataframe["ema_fast_1h"] > dataframe["ema_slow_1h"]
        trend_down = dataframe["ema_fast_1h"] < dataframe["ema_slow_1h"]

        dataframe.loc[(~trend_up) | (dataframe["rsi"] > 70), ["exit_long", "exit_tag"]] = (
            1,
            "regime_flip_or_rsi_high",
        )
        dataframe.loc[(~trend_down) | (dataframe["rsi"] < 30), ["exit_short", "exit_tag"]] = (
            1,
            "regime_flip_or_rsi_low",
        )
        return dataframe

    # ---- risk ------------------------------------------------------------
    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return min(self.target_leverage, max_leverage)

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> Optional[float]:
        """ATR-based stop that trails price (can only tighten), capped by self.stoploss."""
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None
        atr_pct = float(df.iloc[-1]["atr_pct"])
        dist = self.atr_stop_mult * atr_pct * trade.leverage
        dist = max(dist, self.min_stop)
        dist = min(dist, abs(self.stoploss))
        return -dist

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """Backup: close if price gets near liquidation (gaps / stop failure)."""
        liq = getattr(trade, "liquidation_price", None)
        if liq and current_rate:
            if abs(current_rate - liq) / current_rate < self.liq_buffer:
                return "liq_guard"
        return None
