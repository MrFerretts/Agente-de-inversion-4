"""
╔══════════════════════════════════════════════════════════════════╗
║         PATO QUANT — MOTOR DE TRADING AUTÓNOMO                  ║
║                                                                  ║
║  El agente decide SOLO cuándo entrar y salir.                   ║
║  Usa Alpaca Paper Trading (dinero ficticio).                    ║
║                                                                  ║
║  Lógica de decisión:                                            ║
║    → Score técnico + ML + régimen de mercado                    ║
║    → Tamaño de posición: % del portafolio según volatilidad     ║
║    → Stop loss / Take profit: dinámicos basados en ATR          ║
║    → Gestión de riesgo: drawdown máximo, exposición total       ║
║                                                                  ║
║  Se integra en scheduler.py como job adicional.                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import logging
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import pytz
import requests

from market_data import MarketDataFetcher

logger = logging.getLogger("PatoQuant.Trader")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE RIESGO
# Todos los parámetros ajustables sin tocar la lógica
# ─────────────────────────────────────────────────────────────────────────────

RISK_CONFIG = {
    # ── Tamaño de posición (riesgo por trade, no % fijo de equity) ────────────
    # FIX 2026: "position_pct" (antes 8%) presupuestaba el riesgo en dólares,
    # pero combinado con stops típicos de ATR*1.5 (~1.5-3% del precio) el qty
    # resultante superaba max_position_pct casi siempre → en la práctica TODAS
    # las posiciones terminaban en el tope fijo del 20%, sin importar volatilidad.
    # Bajamos el riesgo objetivo por trade para que el tamaño real escale con
    # la distancia al stop (más volátil / stop más ancho → posición más chica).
    "risk_pct_per_trade":  0.005,   # 0.5% del equity arriesgado por trade
    "max_position_pct":    0.20,    # Techo de concentración: máx 20% en un activo
    "min_position_usd":    50.0,    # Mínimo $50 por trade (evitar comisiones)

    # ── Stop Loss / Take Profit dinámicos (basados en ATR) ────────────────────
    "stop_loss_atr_mult":  1.5,     # Stop = precio_entrada - (ATR * 1.5) — más ajustado
    "take_profit_atr_mult": 3.0,    # TP   = precio_entrada + (ATR * 3.0) → R:R 1:2
    "trailing_stop":       True,    # Activar trailing stop
    "trailing_atr_mult":   1.2,     # Trailing = precio_max - (ATR * 1.2)

    # ── Reward:Risk escalado por convicción (Prioridad 4 — EXPLORATORIO) ──────
    # Desactivado por defecto: mientras no haya más muestra, el ratio se
    # mantiene fijo en 2:1 tal como se pidió. Activar solo para testear.
    "conviction_scaled_rr": False,
    "conviction_rr_high_mult": 1.5,   # score>=70 y ML>=70% → TP a 3:1
    "conviction_rr_low_mult":  0.75,  # score cerca del mínimo → TP a 1.5:1

    # ── Filtros de entrada ────────────────────────────────────────────────────
    "min_score":           35,      # Score técnico mínimo (antes 55 — casi nada pasaba)
    "min_adx":             18,      # Tendencia mínima confirmada
    "max_rsi_entry":       75,      # No comprar en sobrecompra extrema
    "min_rsi_entry":       22,      # No comprar en caída libre
    "require_market_open": True,    # Solo operar en horario regular

    # ── Gestión de riesgo del portafolio ──────────────────────────────────────
    "max_drawdown_pct":    0.15,    # Pausar si portafolio cae 15% desde pico
    "max_daily_loss_pct":  0.05,    # Pausar si pierde 5% en el día
    "max_total_exposure":  0.90,    # Máximo 90% del portafolio invertido

    # ── Time stop (Prioridad 2) ─────────────────────────────────────────────
    # Evita capital estancado en posiciones sin momentum (el problema real
    # observado en v3: ~5 meses sin tocar TP/SL por ADX y RVOL bajos).
    "time_stop_days":      21,      # A partir de aquí se evalúa cierre anticipado
    "time_stop_max_adx":   18,      # ...si además ADX sigue débil (igual a min_adx de entrada)
    "time_stop_max_rvol":  1.0,     # ...y sin volumen relativo
    "time_stop_min_gain_pct": 5.0,  # ...y sin ganancia relevante todavía
    "max_holding_days":    45,      # Cierre forzado pase lo que pase (tope duro)

    # ── Cooldown ─────────────────────────────────────────────────────────────
    "trade_cooldown_hours": 2,      # 2h cooldown (antes 4h — perdía re-entradas)
    "max_trades_per_day":  15,      # Máximo 15 operaciones por día

    # ── Correlación sectorial ────────────────────────────────────────────────
    "max_positions_per_sector": 3,  # Máximo 3 posiciones en el mismo sector
}

# ─────────────────────────────────────────────────────────────────────────────
# SIZING POR VIX (Prioridad 3) — mismos umbrales que consensus_analyzer.py
# para mantener consistencia en todo el sistema sobre qué es "VIX alto".
# ─────────────────────────────────────────────────────────────────────────────

VIX_SIZE_MULTIPLIERS = {
    "RISK_ON":  1.00,   # VIX < 19  — mercado tranquilo, tamaño normal
    "NEUTRAL":  0.75,   # VIX 19-25 — algo de cautela
    "RISK_OFF": 0.50,   # VIX 25-35 — reducir posiciones a la mitad
    "CRISIS":   0.25,   # VIX > 35  — muy conservador, solo posiciones pequeñas
}


# ─────────────────────────────────────────────────────────────────────────────
# MAPA DE SECTORES — Evita concentración en activos correlacionados
# ─────────────────────────────────────────────────────────────────────────────

SECTOR_MAP = {
    # Tech
    "AAPL": "tech", "MSFT": "tech", "GOOGL": "tech", "GOOG": "tech",
    "META": "tech", "NVDA": "tech", "AMD": "tech", "INTC": "tech",
    "CRM": "tech", "ORCL": "tech", "ADBE": "tech", "CSCO": "tech",
    "AVGO": "tech", "QCOM": "tech", "MU": "tech", "AMAT": "tech",
    "PLTR": "tech", "SNOW": "tech", "NET": "tech", "DDOG": "tech",
    # E-commerce / Internet
    "AMZN": "ecommerce", "SHOP": "ecommerce", "MELI": "ecommerce",
    "BABA": "ecommerce", "JD": "ecommerce", "EBAY": "ecommerce",
    # Fintech / Finanzas
    "V": "fintech", "MA": "fintech", "PYPL": "fintech", "SQ": "fintech",
    "COIN": "fintech", "JPM": "fintech", "GS": "fintech", "BAC": "fintech",
    "WFC": "fintech", "C": "fintech", "SOFI": "fintech",
    # Auto / EV
    "TSLA": "auto", "F": "auto", "GM": "auto", "RIVN": "auto",
    "LCID": "auto", "NIO": "auto",
    # Media / Entertainment
    "DIS": "media", "NFLX": "media", "CMCSA": "media", "WBD": "media",
    "SPOT": "media", "ROKU": "media",
    # Healthcare
    "JNJ": "health", "UNH": "health", "PFE": "health", "ABBV": "health",
    "MRK": "health", "LLY": "health", "TMO": "health",
    # Energy
    "XOM": "energy", "CVX": "energy", "COP": "energy", "OXY": "energy",
    "USO": "energy", "XLE": "energy",
    # Crypto
    "BTC-USD": "crypto", "ETH-USD": "crypto", "SOL-USD": "crypto",
    "XRP-USD": "crypto", "ADA-USD": "crypto", "DOGE-USD": "crypto",
    "AVAX-USD": "crypto", "DOT-USD": "crypto", "MATIC-USD": "crypto",
    "MSTR": "crypto",
    # Commodities
    "GC=F": "commodity", "SI=F": "commodity", "CL=F": "commodity",
    "GLD": "commodity", "SLV": "commodity",
    # Aerospace / Defense
    "RKLB": "aerospace", "BA": "aerospace", "LMT": "aerospace",
    "RTX": "aerospace", "NOC": "aerospace",
    # ETFs sectoriales
    "SPY": "index_etf", "QQQ": "index_etf", "IWM": "index_etf",
    "DIA": "index_etf", "VOO": "index_etf",
    "XLF": "sector_etf", "XLV": "sector_etf", "XLU": "sector_etf",
    "XLK": "sector_etf", "XLE": "sector_etf", "XLI": "sector_etf",
    # Cybersecurity
    "OKTA": "cybersec", "CRWD": "cybersec", "ZS": "cybersec",
    "PANW": "cybersec", "FTNT": "cybersec",
}


def get_sector(ticker: str) -> str:
    """Retorna el sector del ticker. Default: 'other'."""
    return SECTOR_MAP.get(ticker, "other")


# ─────────────────────────────────────────────────────────────────────────────
# CLIENTE ALPACA
# ─────────────────────────────────────────────────────────────────────────────

class AlpacaClient:
    """
    Wrapper simple para la API de Alpaca.
    Usa paper trading por defecto.
    """

    BASE_URL = "https://paper-api.alpaca.markets"

    def __init__(self):
        self.api_key    = os.getenv("ALPACA_API_KEY", "")
        self.api_secret = os.getenv("ALPACA_API_SECRET", "")

        if not self.api_key or not self.api_secret:
            raise ValueError("ALPACA_API_KEY y ALPACA_API_SECRET requeridos")

        self.headers = {
            "APCA-API-KEY-ID":     self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type":        "application/json",
        }

    def _get(self, endpoint: str) -> Dict:
        r = requests.get(f"{self.BASE_URL}{endpoint}",
                         headers=self.headers, timeout=10)
        r.raise_for_status()
        return r.json()

    def _post(self, endpoint: str, data: Dict) -> Dict:
        r = requests.post(f"{self.BASE_URL}{endpoint}",
                          headers=self.headers, json=data, timeout=10)
        r.raise_for_status()
        return r.json()

    def _delete(self, endpoint: str) -> bool:
        r = requests.delete(f"{self.BASE_URL}{endpoint}",
                            headers=self.headers, timeout=10)
        return r.status_code in (200, 204)

    # ── Cuenta ────────────────────────────────────────────────────────────────

    def get_account(self) -> Dict:
        return self._get("/v2/account")

    def get_portfolio_value(self) -> float:
        acc = self.get_account()
        return float(acc.get("portfolio_value", 0))

    def get_buying_power(self) -> float:
        acc = self.get_account()
        return float(acc.get("buying_power", 0))

    def get_equity(self) -> float:
        acc = self.get_account()
        return float(acc.get("equity", 0))

    # ── Posiciones ────────────────────────────────────────────────────────────

    def get_positions(self) -> List[Dict]:
        return self._get("/v2/positions")

    def get_position(self, symbol: str) -> Optional[Dict]:
        try:
            return self._get(f"/v2/positions/{symbol}")
        except Exception:
            return None

    def close_position(self, symbol: str) -> bool:
        try:
            self._delete(f"/v2/positions/{symbol}")
            logger.info(f"✅ Posición cerrada: {symbol}")
            return True
        except Exception as e:
            logger.error(f"❌ Error cerrando {symbol}: {e}")
            return False

    def close_all_positions(self) -> bool:
        try:
            self._delete("/v2/positions")
            logger.info("✅ Todas las posiciones cerradas")
            return True
        except Exception as e:
            logger.error(f"❌ Error cerrando todas: {e}")
            return False

    # ── Órdenes ───────────────────────────────────────────────────────────────

    def submit_order(self, symbol: str, qty: float,
                     side: str, order_type: str = "market",
                     time_in_force: str = "day",
                     limit_price: float = None,
                     stop_price: float = None) -> Optional[Dict]:
        """
        Envía una orden simple a Alpaca.
        side: 'buy' | 'sell'
        order_type: 'market' | 'limit' | 'stop' | 'stop_limit'
        """
        data = {
            "symbol":        symbol,
            "qty":           str(round(qty, 4)),
            "side":          side,
            "type":          order_type,
            "time_in_force": time_in_force,
        }
        if limit_price:
            data["limit_price"] = str(round(limit_price, 2))
        if stop_price:
            data["stop_price"] = str(round(stop_price, 2))

        try:
            order = self._post("/v2/orders", data)
            logger.info(
                f"📋 Orden enviada: {side.upper()} {qty:.4f} {symbol} "
                f"@ {order_type} | ID: {order.get('id','?')[:8]}"
            )
            return order
        except Exception as e:
            logger.error(f"❌ Error enviando orden {side} {symbol}: {e}")
            return None

    def submit_bracket_order(self, symbol: str, qty: float,
                              stop_loss: float,
                              take_profit: float) -> Optional[Dict]:
        """
        Envía un bracket order (OTO) a Alpaca:
          - Orden principal: Market BUY
          - Stop loss: sell stop automático
          - Take profit: sell limit automático

        Si Railway se cae, Alpaca mantiene las órdenes de protección activas.
        """
        data = {
            "symbol":        symbol,
            "qty":           str(round(qty, 4)),
            "side":          "buy",
            "type":          "market",
            "time_in_force": "gtc",
            "order_class":   "bracket",
            "stop_loss":     {"stop_price": str(round(stop_loss, 2))},
            "take_profit":   {"limit_price": str(round(take_profit, 2))},
        }

        try:
            order = self._post("/v2/orders", data)
            logger.info(
                f"📋 Bracket order: BUY {qty:.4f} {symbol} | "
                f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f} | "
                f"ID: {order.get('id','?')[:8]}"
            )
            return order
        except Exception as e:
            logger.error(f"❌ Error bracket order {symbol}: {e}")
            # Fallback: orden simple sin protección
            logger.warning(f"⚠️ Fallback a orden simple para {symbol}")
            return self.submit_order(symbol, qty, "buy")

    def get_orders(self, status: str = "open") -> List[Dict]:
        return self._get(f"/v2/orders?status={status}")

    def cancel_order(self, order_id: str) -> bool:
        try:
            self._delete(f"/v2/orders/{order_id}")
            return True
        except Exception:
            return False

    def cancel_all_orders(self):
        try:
            self._delete("/v2/orders")
        except Exception:
            pass

    # ── Precio actual ─────────────────────────────────────────────────────────

    def get_latest_price(self, symbol: str) -> Optional[float]:
        try:
            data = self._get(f"/v2/stocks/{symbol}/quotes/latest")
            quote = data.get("quote", {})
            # Usar mid-price si está disponible
            ask = float(quote.get("ap", 0))
            bid = float(quote.get("bp", 0))
            if ask > 0 and bid > 0:
                return (ask + bid) / 2
            return ask or bid or None
        except Exception:
            try:
                data = self._get(f"/v2/stocks/{symbol}/trades/latest")
                return float(data["trade"]["p"])
            except Exception:
                return None


# ─────────────────────────────────────────────────────────────────────────────
# CEREBRO DE DECISIÓN
# ─────────────────────────────────────────────────────────────────────────────

class TradingBrain:
    """
    Decide si comprar, mantener o vender basándose en:
    - Score técnico del scanner
    - Predicción ML (si disponible)
    - Régimen de mercado
    - Gestión de riesgo del portafolio
    """

    def __init__(self, alpaca: AlpacaClient):
        self.alpaca        = alpaca
        self.trade_history: List[Dict] = []
        self.daily_pnl     = 0.0
        self.peak_equity   = None
        self.last_trade_time: Dict[str, datetime] = {}
        self.position_opened_at: Dict[str, datetime] = {}   # ← time stop (P2)

        # Régimen de mercado / VIX, refrescado 1x por ciclo (P3)
        self.market_fetcher = MarketDataFetcher({})
        self.current_regime: Dict = {"regime": "NEUTRAL", "vix": None}

        # Cargar estado persistente (sobrevive restarts de Railway)
        self._load_state()

    def _state_path(self) -> str:
        return "data/trader_state.json"

    def _load_state(self):
        """Carga estado desde disco (sobrevive restarts de Railway)."""
        try:
            import json
            path = self._state_path()
            if os.path.exists(path):
                with open(path, "r") as f:
                    state = json.load(f)
                self.peak_equity = state.get("peak_equity")
                self.daily_pnl = state.get("daily_pnl", 0.0)
                # Restaurar trade history del día actual
                today = datetime.now().strftime("%Y-%m-%d")
                self.trade_history = [
                    t for t in state.get("trade_history", [])
                    if t.get("date") == today
                ]
                # Restaurar cooldowns
                for ticker, ts in state.get("last_trade_time", {}).items():
                    try:
                        self.last_trade_time[ticker] = datetime.fromisoformat(ts)
                    except Exception:
                        pass
                # Restaurar fechas de apertura de posición (para el time stop)
                for ticker, ts in state.get("position_opened_at", {}).items():
                    try:
                        self.position_opened_at[ticker] = datetime.fromisoformat(ts)
                    except Exception:
                        pass
                logger.info(f"📂 Estado del trader restaurado (peak: ${self.peak_equity or 0:,.2f})")
        except Exception as e:
            logger.debug(f"No se pudo cargar estado: {e}")

    def _save_state(self):
        """Persiste estado crítico a disco."""
        try:
            import json
            Path("data").mkdir(exist_ok=True)
            state = {
                "peak_equity": self.peak_equity,
                "daily_pnl": self.daily_pnl,
                "trade_history": self.trade_history,
                "last_trade_time": {
                    k: v.isoformat() for k, v in self.last_trade_time.items()
                },
                "position_opened_at": {
                    k: v.isoformat() for k, v in self.position_opened_at.items()
                },
                "saved_at": datetime.now().isoformat(),
            }
            with open(self._state_path(), "w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            logger.debug(f"No se pudo guardar estado: {e}")

    # ── Decisión de entrada ───────────────────────────────────────────────────

    def should_buy(self, ticker: str, scan_result: Dict,
                   ml_result: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Evalúa si el sistema debe comprar este ticker.
        Retorna (True/False, razón).
        """
        score   = float(scan_result.get("score", 0))
        rsi     = float(scan_result.get("rsi", 50))
        adx     = float(scan_result.get("adx", 0))
        rvol    = float(scan_result.get("rvol", 1))
        rec     = scan_result.get("recommendation", "")

        # ── 1. Filtros básicos (threshold dinámico según tendencia) ──────────

        # Usar SMA cross del scan para ajustar threshold
        trend = scan_result.get("trend", "NEUTRAL")
        sma_cross = scan_result.get("sma_cross", "")
        if trend == "BULLISH" or "GOLDEN" in str(sma_cross).upper():
            min_score = RISK_CONFIG["min_score"] - 5   # Más permisivo en alcista
        else:
            min_score = RISK_CONFIG["min_score"] + 5   # Más estricto sin tendencia

        if score < min_score:
            return False, f"Score insuficiente ({score:.0f} < {min_score})"

        if rsi > RISK_CONFIG["max_rsi_entry"]:
            return False, f"RSI sobrecomprado ({rsi:.1f})"

        if rsi < RISK_CONFIG["min_rsi_entry"]:
            return False, f"RSI en caída libre ({rsi:.1f})"

        if adx < RISK_CONFIG["min_adx"]:
            return False, f"Tendencia débil (ADX {adx:.1f})"

        if rec not in ("COMPRA", "COMPRA FUERTE", "NEUTRAL ALCISTA"):
            return False, f"Señal no alcista ({rec})"

        # ── 2. Cooldown por ticker ────────────────────────────────────────────

        if ticker in self.last_trade_time:
            hours_since = (datetime.now() - self.last_trade_time[ticker]).seconds / 3600
            if hours_since < RISK_CONFIG["trade_cooldown_hours"]:
                return False, f"Cooldown activo ({hours_since:.1f}h < {RISK_CONFIG['trade_cooldown_hours']}h)"

        # ── 3. ¿Ya tenemos posición en este ticker? ───────────────────────────

        pos = self.alpaca.get_position(ticker)
        if pos:
            return False, "Ya hay posición abierta"

        # ── 4. Límite de trades diarios ───────────────────────────────────────

        today_trades = [t for t in self.trade_history
                        if t.get("date") == datetime.now().strftime("%Y-%m-%d")]
        if len(today_trades) >= RISK_CONFIG["max_trades_per_day"]:
            return False, f"Límite diario alcanzado ({RISK_CONFIG['max_trades_per_day']})"

        # ── 5. Verificar drawdown del portafolio ──────────────────────────────

        ok, reason = self._check_portfolio_health()
        if not ok:
            return False, reason

        # ── 5b. Límite de posiciones por sector (anti-correlación) ───────────

        sector = get_sector(ticker)
        positions = self.alpaca.get_positions()
        sector_count = sum(
            1 for p in positions
            if get_sector(p.get("symbol", "")) == sector
        )
        max_per_sector = RISK_CONFIG["max_positions_per_sector"]
        if sector_count >= max_per_sector:
            return False, (
                f"Sector '{sector}' saturado ({sector_count}/{max_per_sector} posiciones)"
            )

        # ── 6. Filtro ML (bonus de confianza si disponible) ───────────────────

        ml_confidence = 0.0
        if ml_result and ml_result.get("probability_up"):
            ml_confidence = float(ml_result["probability_up"])
            # Si ML predice bajada con alta confianza, bloqueamos la entrada
            if ml_confidence < 0.35:
                return False, f"ML predice bajada ({ml_confidence*100:.0f}% prob subida)"

        # ── 7. Score compuesto final ──────────────────────────────────────────

        composite = score
        if ml_confidence > 0.6:
            composite += 10  # Bonus si ML confirma
        if rvol > 2.0:
            composite += 5   # Bonus si hay volumen institucional

        if composite < RISK_CONFIG["min_score"]:
            return False, f"Score compuesto insuficiente ({composite:.0f})"

        return True, f"Score={composite:.0f} | RSI={rsi:.1f} | ADX={adx:.1f} | ML={ml_confidence*100:.0f}%"

    # ── Régimen de mercado / VIX (Prioridad 3) ────────────────────────────────

    def refresh_market_regime(self):
        """Refresca el régimen de mercado (VIX) una vez por ciclo de trading."""
        try:
            self.current_regime = self.market_fetcher.get_market_regime()
        except Exception as e:
            logger.warning(f"⚠️ No se pudo refrescar régimen de mercado: {e}")

    def get_vix_size_multiplier(self) -> float:
        """Multiplicador de tamaño de posición según el régimen de VIX actual."""
        regime = self.current_regime.get("regime", "NEUTRAL")
        return VIX_SIZE_MULTIPLIERS.get(regime, 0.75)

    # ── Tamaño de posición ────────────────────────────────────────────────────

    def calculate_position_size(self, ticker: str, price: float,
                                  atr: float,
                                  conviction_rr_mult: float = 1.0
                                  ) -> Tuple[float, float, float]:
        """
        Calcula tamaño de posición usando volatilidad (ATR) — riesgo por trade,
        no % fijo de equity.

        Método: Risk-based position sizing
          - Arriesgar N% del portafolio por trade (ajustado por régimen de VIX)
          - Stop loss = ATR * multiplicador
          - Qty = (portafolio * risk_pct * vix_mult) / stop_distance

        A mayor distancia al stop (más volatilidad), MENOR el tamaño en dólares,
        para arriesgar aproximadamente el mismo % de la cuenta en cada trade.

        Retorna: (qty, stop_loss_price, take_profit_price)
        """
        portfolio_value = self.alpaca.get_portfolio_value()

        vix_mult = self.get_vix_size_multiplier()

        # Riesgo en dólares = risk_pct_per_trade del portafolio, ajustado por VIX
        risk_dollars = portfolio_value * RISK_CONFIG["risk_pct_per_trade"] * vix_mult

        # Stop distance basado en ATR
        stop_distance   = atr * RISK_CONFIG["stop_loss_atr_mult"]
        profit_distance = atr * RISK_CONFIG["take_profit_atr_mult"]

        # Reward:Risk escalado por convicción (P4 — exploratorio, off por defecto)
        if RISK_CONFIG["conviction_scaled_rr"]:
            profit_distance *= conviction_rr_mult

        # Evitar divisiones por cero
        if stop_distance < 0.01:
            stop_distance = price * 0.03  # Fallback: 3% del precio

        # Número de acciones que podemos comprar arriesgando risk_dollars
        qty = risk_dollars / stop_distance

        # Verificar que no exceda max_position_pct del portafolio (techo de
        # concentración, no el driver principal del tamaño — ver RISK_CONFIG)
        max_value = portfolio_value * RISK_CONFIG["max_position_pct"]
        qty = min(qty, max_value / price)

        # Verificar mínimo
        if qty * price < RISK_CONFIG["min_position_usd"]:
            qty = RISK_CONFIG["min_position_usd"] / price

        # Verificar buying power
        buying_power = self.alpaca.get_buying_power()
        if qty * price > buying_power * 0.95:
            qty = (buying_power * 0.95) / price

        # Redondear a 4 decimales para fractional shares
        qty = max(round(qty, 4), 0.0001)

        stop_loss   = round(price - stop_distance,   2)
        take_profit = round(price + profit_distance, 2)

        return qty, stop_loss, take_profit

    # ── Decisión de salida ────────────────────────────────────────────────────

    def should_sell(self, position: Dict, scan_result: Dict) -> Tuple[bool, str]:
        """
        Evalúa si cerrar una posición existente.
        Usa trailing stop dinámico + señales técnicas.
        """
        symbol      = position.get("symbol", "")
        entry_price = float(position.get("avg_entry_price", 0))
        current_val = float(position.get("current_price", 0))
        unrealized  = float(position.get("unrealized_plpc", 0)) * 100  # en %
        qty         = float(position.get("qty", 0))

        score = float(scan_result.get("score", 0))
        rsi   = float(scan_result.get("rsi", 50))
        rec   = scan_result.get("recommendation", "")

        # Recuperar ATR del scan para trailing stop
        atr = float(scan_result.get("atr", current_val * 0.02))

        # ── Stop loss fijo (desde entrada) ────────────────────────────────────
        stop_distance = atr * RISK_CONFIG["stop_loss_atr_mult"]
        stop_price    = entry_price - stop_distance

        if current_val <= stop_price:
            return True, f"Stop loss activado (precio {current_val:.2f} ≤ stop {stop_price:.2f})"

        # ── Trailing stop dinámico ────────────────────────────────────────────
        # El trailing stop se activa cuando ya estamos en ganancia
        if RISK_CONFIG["trailing_stop"] and current_val > entry_price:
            # Usar el máximo observado (Alpaca no da high_water, usamos unrealized)
            high_water = entry_price * (1 + max(unrealized, 0) / 100)
            trailing_distance = atr * RISK_CONFIG["trailing_atr_mult"]
            trailing_stop = high_water - trailing_distance

            # Solo aplica trailing si es más alto que el stop fijo
            if trailing_stop > stop_price and current_val <= trailing_stop:
                return True, (
                    f"Trailing stop activado (precio {current_val:.2f} "
                    f"≤ trail {trailing_stop:.2f} | ganancia máx +{unrealized:.1f}%)"
                )

        # ── Take profit ───────────────────────────────────────────────────────
        tp_distance = atr * RISK_CONFIG["take_profit_atr_mult"]
        tp_price    = entry_price + tp_distance

        if current_val >= tp_price:
            return True, f"Take profit alcanzado (+{unrealized:.1f}% | TP {tp_price:.2f})"

        # ── Time stop (Prioridad 2) ───────────────────────────────────────────
        # Evita capital estancado en posiciones sin momentum, ni TP ni SL.
        adx  = float(scan_result.get("adx", 0))
        rvol = float(scan_result.get("rvol", 1))
        opened_at = self.position_opened_at.get(symbol)

        if opened_at is not None:
            days_held = (datetime.now() - opened_at).days

            # Tope duro: cierre forzado sin importar nada más
            if days_held >= RISK_CONFIG["max_holding_days"]:
                return True, (
                    f"Time stop máximo alcanzado ({days_held}d ≥ "
                    f"{RISK_CONFIG['max_holding_days']}d) — cierre forzado, "
                    f"{unrealized:+.1f}%"
                )

            # Tope suave: solo si además no hay momentum (el patrón real de v3:
            # ADX bajo + RVOL bajo) y la ganancia todavía no es relevante
            if (days_held >= RISK_CONFIG["time_stop_days"]
                    and adx < RISK_CONFIG["time_stop_max_adx"]
                    and rvol < RISK_CONFIG["time_stop_max_rvol"]
                    and unrealized < RISK_CONFIG["time_stop_min_gain_pct"]):
                return True, (
                    f"Time stop: {days_held}d sin momentum "
                    f"(ADX {adx:.1f} | RVOL {rvol:.2f}x | {unrealized:+.1f}%) "
                    f"— capital estancado, replanteando"
                )

        # ── Señal técnica bajista fuerte ──────────────────────────────────────
        if rec in ("VENTA", "VENTA FUERTE") and score <= -30:
            return True, f"Señal bajista fuerte (Score {score:.0f})"

        # ── RSI sobrecomprado extremo ─────────────────────────────────────────
        if rsi > 80 and unrealized > 8:
            return True, f"RSI sobrecomprado ({rsi:.1f}) con ganancia ({unrealized:.1f}%)"

        # ── Pérdida máxima tolerada ───────────────────────────────────────────
        max_loss_pct = -RISK_CONFIG["stop_loss_atr_mult"] * 100 * (atr / entry_price)
        if unrealized < max_loss_pct * 1.5:  # 50% peor que el stop calculado
            return True, f"Pérdida excesiva ({unrealized:.1f}%)"

        return False, f"Mantener (Score {score:.0f} | {unrealized:+.1f}%)"

    # ── Salud del portafolio ──────────────────────────────────────────────────

    def _check_portfolio_health(self) -> Tuple[bool, str]:
        """Verifica que el portafolio no esté en drawdown excesivo."""
        try:
            equity = self.alpaca.get_equity()

            # Inicializar pico si es la primera vez
            if self.peak_equity is None:
                self.peak_equity = equity
            else:
                old_peak = self.peak_equity
                self.peak_equity = max(self.peak_equity, equity)
                if self.peak_equity != old_peak:
                    self._save_state()  # Persistir nuevo pico

            # Drawdown actual
            if self.peak_equity > 0:
                drawdown = (self.peak_equity - equity) / self.peak_equity
                if drawdown > RISK_CONFIG["max_drawdown_pct"]:
                    return False, f"Drawdown máximo alcanzado ({drawdown*100:.1f}%)"

            # Exposición total
            positions   = self.alpaca.get_positions()
            total_invested = sum(
                float(p.get("market_value", 0)) for p in positions
            )
            exposure = total_invested / equity if equity > 0 else 0

            if exposure > RISK_CONFIG["max_total_exposure"]:
                return False, f"Exposición máxima alcanzada ({exposure*100:.1f}%)"

            return True, "OK"

        except Exception as e:
            logger.warning(f"⚠️ No se pudo verificar salud del portafolio: {e}")
            return True, "OK (sin datos)"

    def log_trade(self, ticker: str, action: str, qty: float,
                  price: float, reason: str):
        """Registra operación en el historial y persiste a disco."""
        self.last_trade_time[ticker] = datetime.now()

        # Trackear apertura/cierre de posición para el time stop (P2)
        if action == "BUY":
            self.position_opened_at[ticker] = datetime.now()
        elif action == "SELL":
            self.position_opened_at.pop(ticker, None)

        self.trade_history.append({
            "date":   datetime.now().strftime("%Y-%m-%d"),
            "time":   datetime.now().strftime("%H:%M:%S"),
            "ticker": ticker,
            "action": action,
            "qty":    qty,
            "price":  price,
            "reason": reason,
        })
        self._save_state()


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR PRINCIPAL DE TRADING
# ─────────────────────────────────────────────────────────────────────────────

class AutonomousTrader:
    """
    Motor de trading autónomo.
    Se llama desde scheduler.py después de cada scan.

    Flujo:
      1. Revisar posiciones abiertas → ¿cerrar?
      2. Revisar resultados del scan → ¿abrir nuevas?
      3. Log de todo en Supabase
    """

    def __init__(self, db, notifier, perf_tracker=None):
        self.db            = db
        self.notifier      = notifier
        self.perf_tracker  = perf_tracker

        try:
            self.alpaca = AlpacaClient()
            self.brain  = TradingBrain(self.alpaca)
            self.active = True
            logger.info("💰 Motor de Trading Autónomo inicializado (Paper Trading)")
            logger.info(f"   Riesgo por trade: {RISK_CONFIG['risk_pct_per_trade']*100:.2f}% del equity (x VIX mult)")
            logger.info(f"   Stop/TP: {RISK_CONFIG['stop_loss_atr_mult']}x ATR / {RISK_CONFIG['take_profit_atr_mult']}x ATR")
            logger.info(f"   Time stop: {RISK_CONFIG['time_stop_days']}d (soft) / {RISK_CONFIG['max_holding_days']}d (hard)")
        except Exception as e:
            self.active = False
            logger.error(f"❌ No se pudo inicializar Alpaca: {e}")

    # ── Job principal ─────────────────────────────────────────────────────────

    def run(self, scan_results: List[Dict], ml_models: Dict = None):
        """
        Ejecuta el ciclo completo de trading.
        Llamar después de cada scan del scheduler.

        Args:
            scan_results: Lista de resultados del scan actual
            ml_models:    Dict {ticker: model} para predicciones
        """
        if not self.active:
            logger.debug("🔒 Trader inactivo (Alpaca no configurado)")
            return

        logger.info("💰 Iniciando ciclo de trading autónomo...")

        try:
            # ── Paso 0: Refrescar régimen de mercado (VIX) para sizing ────────
            self.brain.refresh_market_regime()
            regime = self.brain.current_regime
            logger.info(
                f"   🌡️ Régimen: {regime.get('regime', 'N/A')} "
                f"(VIX {regime.get('vix', 0):.1f} | "
                f"tamaño x{self.brain.get_vix_size_multiplier():.2f})"
            )

            # ── Paso 1: Gestionar posiciones existentes ───────────────────────
            self._manage_open_positions(scan_results)

            # ── Paso 2: Buscar nuevas entradas ────────────────────────────────
            self._find_new_entries(scan_results, ml_models or {})

            # ── Paso 3: Resumen ───────────────────────────────────────────────
            self._log_portfolio_summary()

        except Exception as e:
            logger.error(f"❌ Error en ciclo de trading: {e}")
            logger.debug(traceback.format_exc())

    def _manage_open_positions(self, scan_results: List[Dict]):
        """Revisa posiciones abiertas y decide si cerrarlas."""
        positions = self.alpaca.get_positions()

        if not positions:
            return

        logger.info(f"   Revisando {len(positions)} posiciones abiertas...")

        # Crear dict de scan_results por ticker para búsqueda rápida
        scan_map = {r["ticker"]: r for r in scan_results}

        for position in positions:
            ticker  = position.get("symbol", "")
            qty     = float(position.get("qty", 0))
            pnl_pct = float(position.get("unrealized_plpc", 0)) * 100
            price   = float(position.get("current_price", 0))

            # Posición sin fecha de apertura registrada (p.ej. abierta antes de
            # existir el time stop, o tras un restart sin estado persistido):
            # sembrar "ahora" para no dispararlo de inmediato.
            if ticker not in self.brain.position_opened_at:
                self.brain.position_opened_at[ticker] = datetime.now()

            scan = scan_map.get(ticker)
            if not scan:
                logger.debug(f"   {ticker}: sin datos de scan, manteniendo")
                continue

            should_close, reason = self.brain.should_sell(position, scan)

            if should_close:
                logger.info(f"   🔴 CERRAR {ticker}: {reason}")
                order = self.alpaca.submit_order(
                    symbol=ticker, qty=qty, side="sell"
                )
                if order:
                    self.brain.log_trade(ticker, "SELL", qty, price, reason)
                    self._notify_trade(ticker, "VENTA", qty, price, pnl_pct, reason)
                    self._save_trade_db(ticker, "SELL", qty, price, pnl_pct, reason)

                    # Registrar en performance tracker
                    if self.perf_tracker:
                        entry = float(position.get("avg_entry_price", 0))
                        pnl_usd = (price - entry) * qty
                        self.perf_tracker.record_trade(
                            ticker=ticker, action="SELL", qty=qty,
                            entry_price=entry, exit_price=price,
                            pnl=pnl_usd, reason=reason,
                        )
            else:
                logger.info(f"   🟡 MANTENER {ticker}: {reason}")

    def _find_new_entries(self, scan_results: List[Dict], ml_models: Dict):
        """Busca oportunidades de entrada en los resultados del scan."""

        # Ordenar por score descendente
        candidates = sorted(
            scan_results,
            key=lambda x: float(x.get("score", 0)),
            reverse=True
        )

        entries_this_cycle = 0

        for result in candidates:
            ticker = result.get("ticker", "")
            score  = float(result.get("score", 0))
            price  = float(result.get("price", 0))
            atr    = float(result.get("atr", price * 0.02))

            if score < RISK_CONFIG["min_score"]:
                break  # Ordenados por score, podemos parar

            if price <= 0:
                continue

            # Predicción ML: ya viene calculada correctamente desde
            # scheduler.py::_analyze_ticker (usa el modelo real .predict() con
            # el DataFrame completo). FIX: antes se intentaba llamar un método
            # predict_latest() que no existe en AdvancedTradingMLModel — el
            # error quedaba silenciado por un except genérico y ml_result
            # siempre era None, por lo que ML nunca contribuía a la decisión
            # (de ahí el "ML=0%" en todos los logs de trades).
            ml_result = None
            if result.get("ml_prob_up") is not None:
                ml_result = {
                    "probability_up": result["ml_prob_up"],
                    "recommendation": result.get("ml_rec"),
                }

            # ¿Debemos comprar?
            should_buy, reason = self.brain.should_buy(ticker, result, ml_result)

            if should_buy:
                # Convicción compuesta para el reward:risk exploratorio (P4)
                ml_conf = float(ml_result.get("probability_up", 0.5)) if ml_result else 0.5
                if score >= 70 and ml_conf >= 0.70:
                    conviction_mult = RISK_CONFIG["conviction_rr_high_mult"]
                elif score <= RISK_CONFIG["min_score"] + 10:
                    conviction_mult = RISK_CONFIG["conviction_rr_low_mult"]
                else:
                    conviction_mult = 1.0

                qty, stop_loss, take_profit = self.brain.calculate_position_size(
                    ticker, price, atr, conviction_rr_mult=conviction_mult
                )

                logger.info(
                    f"   🟢 COMPRAR {ticker}: {reason}\n"
                    f"      Qty: {qty:.4f} | Precio: ${price:.2f} | "
                    f"SL: ${stop_loss:.2f} | TP: ${take_profit:.2f}"
                )

                # Bracket order: stop loss + take profit como órdenes reales
                # Si Railway se cae, Alpaca mantiene la protección activa
                order = self.alpaca.submit_bracket_order(
                    symbol=ticker, qty=qty,
                    stop_loss=stop_loss, take_profit=take_profit,
                )

                if order:
                    self.brain.log_trade(ticker, "BUY", qty, price, reason)
                    self._notify_trade(ticker, "COMPRA", qty, price, 0, reason,
                                       stop_loss=stop_loss, take_profit=take_profit)
                    self._save_trade_db(ticker, "BUY", qty, price, 0, reason,
                                        stop_loss=stop_loss, take_profit=take_profit)
                    entries_this_cycle += 1

        if entries_this_cycle == 0:
            logger.info("   Sin nuevas entradas en este ciclo")
        else:
            logger.info(f"   ✅ {entries_this_cycle} nuevas posiciones abiertas")

    def _log_portfolio_summary(self):
        """Log del estado del portafolio."""
        try:
            account   = self.alpaca.get_account()
            equity    = float(account.get("equity", 0))
            cash      = float(account.get("cash", 0))
            positions = self.alpaca.get_positions()

            total_pnl = sum(float(p.get("unrealized_pl", 0)) for p in positions)
            total_pnl_pct = sum(
                float(p.get("unrealized_plpc", 0)) * 100 for p in positions
            ) / max(len(positions), 1)

            logger.info(
                f"\n{'─'*50}\n"
                f"💼 PORTAFOLIO PAPER TRADING\n"
                f"   Equity:     ${equity:>10.2f}\n"
                f"   Cash:       ${cash:>10.2f}\n"
                f"   Posiciones: {len(positions)}\n"
                f"   PnL abierto: ${total_pnl:>+8.2f} ({total_pnl_pct:+.1f}%)\n"
                f"{'─'*50}"
            )
        except Exception as e:
            logger.warning(f"⚠️ No se pudo obtener resumen: {e}")

    # ── Notificaciones ────────────────────────────────────────────────────────

    def _notify_trade(self, ticker: str, action: str, qty: float,
                       price: float, pnl_pct: float, reason: str,
                       stop_loss: float = None, take_profit: float = None):
        """Envía notificación Telegram de la operación."""
        emoji = "🟢" if action == "COMPRA" else "🔴"
        pnl_str = f" | PnL: {pnl_pct:+.1f}%" if action == "VENTA" else ""

        msg = (
            f"{emoji} *PAPER TRADE — {action}*\n"
            f"📈 Ticker: *{ticker}*\n"
            f"💲 Precio: ${price:.2f}\n"
            f"📦 Qty: {qty:.4f} acciones\n"
            f"💰 Valor: ${qty*price:.2f}{pnl_str}\n"
        )

        if stop_loss and take_profit:
            msg += (
                f"🛡️ Stop Loss: ${stop_loss:.2f}\n"
                f"🎯 Take Profit: ${take_profit:.2f}\n"
            )

        msg += f"\n📊 Razón: {reason}"

        try:
            self.notifier.send_telegram(msg, ticker, f"trade_{action.lower()}")
        except Exception:
            pass

    # ── Persistencia ──────────────────────────────────────────────────────────

    def _save_trade_db(self, ticker: str, action: str, qty: float,
                        price: float, pnl_pct: float, reason: str,
                        stop_loss: float = None, take_profit: float = None):
        """Guarda la operación en Supabase."""
        try:
            import json
            message = json.dumps({
                "action":      action,
                "qty":         qty,
                "price":       price,
                "pnl_pct":     pnl_pct,
                "stop_loss":   stop_loss,
                "take_profit": take_profit,
                "reason":      reason,
            })
            self.db.save_alert(ticker, f"trade_{action.lower()}", message, "alpaca")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo guardar trade en BD: {e}")

    # ── Estado del trader ─────────────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Retorna estado actual para el dashboard."""
        if not self.active:
            return {"active": False, "error": "Alpaca no configurado"}

        try:
            account   = self.alpaca.get_account()
            positions = self.alpaca.get_positions()

            return {
                "active":           True,
                "mode":             "PAPER TRADING",
                "equity":           float(account.get("equity", 0)),
                "cash":             float(account.get("cash", 0)),
                "buying_power":     float(account.get("buying_power", 0)),
                "open_positions":   len(positions),
                "positions":        positions,
                "trades_today":     len([
                    t for t in self.brain.trade_history
                    if t.get("date") == datetime.now().strftime("%Y-%m-%d")
                ]),
                "trade_history":    self.brain.trade_history[-20:],
            }
        except Exception as e:
            return {"active": False, "error": str(e)}
