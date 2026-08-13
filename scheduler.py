"""
╔══════════════════════════════════════════════════════════════════╗
║           PATO QUANT — SCHEDULER AUTÓNOMO                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import logging
import smtplib
import traceback

from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional
from pathlib import Path

import pandas as pd
import numpy as np
import pytz
import schedule

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

import database as db_supabase

from market_data import MarketDataFetcher
from technical_analysis import TechnicalAnalyzer
from core.state_manager import DataProcessor
from proactive_agent import ProactiveAgent
from autonomous_trader import AutonomousTrader          # ← CAMBIO 1
from core.performance_tracker import PerformanceTracker

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    "scan_interval_minutes": 30,    # FIX: Con datos diarios, escanear cada 30 min es suficiente
    "ml_retrain_hours":      24,    # (antes: 5 min — generaba ruido innecesario con velas diarias)
    "market_hours_only":     True,
    "agent_interval_hours":  4,
    "alert_score_threshold": 60,
    "alert_ml_probability":  0.72,
    "alert_rvol_threshold":  2.0,
    "db_path":               "data/scheduler.db",
    "results_retention_days": 30,
    "email_enabled":   bool(os.getenv("EMAIL_SENDER")),
    "email_sender":    os.getenv("EMAIL_SENDER", ""),
    "email_password":  os.getenv("EMAIL_PASSWORD", ""),
    "email_recipient": os.getenv("EMAIL_RECIPIENT", ""),
    "smtp_server":     "smtp.gmail.com",
    "smtp_port":       587,
    "telegram_enabled":  bool(os.getenv("TELEGRAM_TOKEN")),
    "telegram_token":    os.getenv("TELEGRAM_TOKEN", ""),
    "telegram_chat_id":  os.getenv("TELEGRAM_CHAT_ID", ""),
    "timezone":          "America/New_York",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

Path("logs").mkdir(exist_ok=True)
Path("data").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/scheduler.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("PatoQuant")


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE ADAPTER
# ─────────────────────────────────────────────────────────────────────────────

class Database:
    """Adaptador que conecta scheduler.py con database.py (Supabase)."""

    def __init__(self, db_path=None):
        logger.info("Conectando a Supabase PostgreSQL...")
        db_supabase.init_tables()

    def save_scan_result(self, ticker: str, result: Dict):
        db_supabase.save_scan_result(ticker, result)

    def get_watchlist(self) -> List[Dict]:
        return db_supabase.get_watchlist()

    def sync_watchlist_from_json(self, json_path: str = "data/watchlist.json"):
        db_supabase.sync_watchlist_from_json(json_path)

    def save_alert(self, ticker: str, alert_type: str, message: str, channel: str):
        db_supabase.save_alert(ticker, alert_type, message, channel)

    def was_alert_sent_recently(self, ticker: str, alert_type: str,
                                 cooldown_minutes: int = 30) -> bool:
        return not db_supabase.alert_cooldown_ok(ticker, alert_type, cooldown_minutes)

    def cleanup_old_data(self, days: int):
        db_supabase.cleanup_old_data(days)

    def get_latest_results(self, limit: int = 50) -> pd.DataFrame:
        return db_supabase.get_top_picks(n=limit)

    def get_ticker_history(self, ticker: str, days: int = 7) -> pd.DataFrame:
        return db_supabase.get_ticker_history(ticker, days)

    def save_ml_signal(self, ticker: str, prediction: Dict):
        if not prediction:
            return
        db_supabase.save_ml_signal(
            ticker=ticker,
            probability=prediction.get("probability_up"),
            signal=prediction.get("recommendation"),
            model_type="ensemble",
            features={
                "confidence":      prediction.get("confidence"),
                "confidence_level": prediction.get("confidence_level"),
                "threshold":       prediction.get("threshold"),
                "model_accuracy":  prediction.get("model_accuracy"),
                "model_agreement": prediction.get("model_agreement"),
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
# NOTIFICACIONES
# ─────────────────────────────────────────────────────────────────────────────

class Notifier:

    def __init__(self, db: Database):
        self.db = db

    def send_email(self, subject: str, body_html: str, ticker: str, alert_type: str):
        if not CONFIG["email_enabled"]:
            return
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = subject
            msg["From"]    = CONFIG["email_sender"]
            msg["To"]      = CONFIG["email_recipient"]
            msg.attach(MIMEText(body_html, "html"))
            with smtplib.SMTP(CONFIG["smtp_server"], CONFIG["smtp_port"]) as server:
                server.starttls()
                server.login(CONFIG["email_sender"], CONFIG["email_password"])
                server.send_message(msg)
            self.db.save_alert(ticker, alert_type, subject, "email")
            logger.info(f"📧 Email enviado: {subject}")
        except Exception as e:
            logger.error(f"❌ Error enviando email: {e}")

    def send_telegram(self, message: str, ticker: str, alert_type: str):
        if not CONFIG["telegram_enabled"]:
            return
        try:
            import requests
            url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendMessage"
            response = requests.post(url, json={
                "chat_id":    CONFIG["telegram_chat_id"],
                "text":       message,
                "parse_mode": "Markdown",
            }, timeout=10)
            if response.ok:
                self.db.save_alert(ticker, alert_type, message, "telegram")
                logger.info(f"📱 Telegram: {ticker} — {alert_type}")
            else:
                logger.warning(f"⚠️ Telegram falló: {response.text}")
        except Exception as e:
            logger.error(f"❌ Error Telegram: {e}")

    def notify(self, ticker: str, result: Dict, alert_type: str):
        if self.db.was_alert_sent_recently(ticker, alert_type, cooldown_minutes=30):
            logger.debug(f"🔕 Cooldown: {ticker} — {alert_type}")
            return

        tz      = pytz.timezone(CONFIG["timezone"])
        now_str = datetime.now(tz).strftime("%H:%M:%S %Z")
        score   = result.get("score", 0)
        rec     = result.get("recommendation", "—")
        price   = result.get("price", 0)
        rsi     = result.get("rsi", 0)
        rvol    = result.get("rvol", 0)

        emoji  = "🟢" if score > 0 else "🔴"
        tg_msg = (
            f"{emoji} *{ticker}* — {alert_type.upper()}\n"
            f"⏰ {now_str}\n"
            f"💲 Precio: ${price:.2f}\n"
            f"📊 Score: {score}/100 — {rec}\n"
            f"📈 RSI: {rsi:.1f} | RVOL: {rvol:.2f}x"
        )
        self.send_telegram(tg_msg, ticker, alert_type)

        subject = f"🦆 Pato Quant | {ticker} — {rec} (Score: {score})"
        html = f"""
        <html><body style="font-family:Arial,sans-serif;background:#111;color:#eee;padding:20px">
        <h2 style="color:#00e676">🦆 Pato Quant — Alerta Automática</h2>
        <table style="border-collapse:collapse;width:100%">
          <tr><td style="padding:8px;color:#aaa">Ticker</td>
              <td style="padding:8px;font-weight:bold;font-size:1.4em">{ticker}</td></tr>
          <tr><td style="padding:8px;color:#aaa">Precio</td>
              <td style="padding:8px">${price:.2f}</td></tr>
          <tr><td style="padding:8px;color:#aaa">Score</td>
              <td style="padding:8px;color:{'#00e676' if score>0 else '#ff5252'};font-weight:bold">
              {score}/100</td></tr>
          <tr><td style="padding:8px;color:#aaa">Señal</td>
              <td style="padding:8px;font-weight:bold">{rec}</td></tr>
          <tr><td style="padding:8px;color:#aaa">RSI</td>
              <td style="padding:8px">{rsi:.1f}</td></tr>
          <tr><td style="padding:8px;color:#aaa">RVOL</td>
              <td style="padding:8px">{rvol:.2f}x</td></tr>
          <tr><td style="padding:8px;color:#aaa">Hora</td>
              <td style="padding:8px">{now_str}</td></tr>
        </table>
        <p style="color:#555;font-size:0.8em;margin-top:20px">
          Enviado automáticamente por Pato Quant Scheduler.<br>
          No constituye asesoría financiera.
        </p>
        </body></html>
        """
        self.send_email(subject, html, ticker, alert_type)


# ─────────────────────────────────────────────────────────────────────────────
# MARKET HOURS
# ─────────────────────────────────────────────────────────────────────────────

def is_market_open() -> bool:
    tz  = pytz.timezone("America/New_York")
    now = datetime.now(tz)
    if now.weekday() >= 5:
        return False
    open_  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_ = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_ <= now <= close_


def is_pre_market() -> bool:
    tz  = pytz.timezone("America/New_York")
    now = datetime.now(tz)
    if now.weekday() >= 5:
        return False
    pre_open  = now.replace(hour=4, minute=0,  second=0, microsecond=0)
    pre_close = now.replace(hour=9, minute=30, second=0, microsecond=0)
    return pre_open <= now < pre_close


# ─────────────────────────────────────────────────────────────────────────────
# SCHEDULER PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class QuantScheduler:

    def __init__(self):
        self._load_app_config()

        self.db       = Database(CONFIG["db_path"])
        self.notifier = Notifier(self.db)
        self.fetcher  = MarketDataFetcher(self.api_config)
        self.analyzer = TechnicalAnalyzer(self.technical_config)
        self.ml_models: Dict = {}

        self.db.sync_watchlist_from_json()

        self.stats = {
            "scans_completed": 0,
            "alerts_sent":     0,
            "errors":          0,
            "started_at":      datetime.now().isoformat(),
        }

        logger.info("🦆 Pato Quant Scheduler inicializado")
        logger.info(f"   Intervalo: cada {CONFIG['scan_interval_minutes']} min")
        logger.info(f"   Solo horario de mercado: {CONFIG['market_hours_only']}")

        self.agent = ProactiveAgent(
            watchlist_path="data/watchlist.json",
            max_watchlist_size=50,
            groq_api_key=self.api_config.get("groq_api_key", ""),
        )
        logger.info("🤖 Agente Proactivo listo")

        # ── Performance Tracker: métricas reales (no backtest) ────────────────
        self.perf_tracker = PerformanceTracker()

        # ── CAMBIO 2: Inicializar trader autónomo ─────────────────────────────
        self.trader = AutonomousTrader(
            db=self.db,
            notifier=self.notifier,
            perf_tracker=self.perf_tracker,
        )

    def _load_app_config(self):
        try:
            from config import API_CONFIG, PORTFOLIO_CONFIG, TECHNICAL_INDICATORS
            self.api_config       = API_CONFIG
            self.portfolio_config = PORTFOLIO_CONFIG
            self.technical_config = TECHNICAL_INDICATORS
        except ImportError:
            logger.warning("⚠️  config.py no encontrado. Usando defaults.")
            self.api_config       = {}
            self.portfolio_config = {"stocks": ["AAPL", "MSFT"], "crypto": ["BTC-USD"]}
            self.technical_config = {}

    # ── ML ────────────────────────────────────────────────────────────────────

    def load_ml_models(self):
        try:
            import pickle
            models_dir = Path("data/models")
            if not models_dir.exists():
                logger.info("📂 No hay modelos guardados. Se entrenarán en el primer ciclo.")
                return
            for model_file in models_dir.glob("*.pkl"):
                ticker = model_file.stem
                with open(model_file, "rb") as f:
                    self.ml_models[ticker] = pickle.load(f)
                logger.info(f"🤖 Modelo cargado: {ticker}")
        except Exception as e:
            logger.error(f"❌ Error cargando modelos: {e}")

    def ensure_ml_models(self):
        """
        Entrena de inmediato los modelos que falten al arrancar.

        FIX: antes el primer entrenamiento solo ocurría vía el job programado
        cada `ml_retrain_hours` (24h) — en un deploy fresco de Railway (o tras
        cualquier restart si data/models no está en un volumen persistente),
        self.ml_models quedaba vacío hasta un día después. Durante esa ventana
        el ML no aportaba nada a las decisiones ("ML=0%" en todos los trades).
        Llamar esto una vez al arrancar, antes del primer scan, para que el
        ML esté disponible desde el ciclo 1.
        """
        watchlist = self.db.get_watchlist()
        faltantes = [item["ticker"] for item in watchlist
                     if item["ticker"] not in self.ml_models]

        if not faltantes:
            logger.info("🤖 Todos los modelos ML ya están cargados.")
            return

        logger.info(f"🎓 Entrenando {len(faltantes)} modelos ML faltantes al arrancar...")
        for ticker in faltantes:
            try:
                data = self.fetcher.get_stock_data(ticker, period="1y")
                if data is not None and len(data) >= 200:
                    data_p = DataProcessor.prepare_full_analysis(data, self.analyzer)
                    self.train_and_save_model(ticker, data_p)
                else:
                    logger.warning(f"⚠️  {ticker}: datos insuficientes para entrenar ML al arrancar")
            except Exception as e:
                logger.error(f"❌ Entrenamiento inicial falló para {ticker}: {e}")

    def train_and_save_model(self, ticker: str, data: pd.DataFrame):
        try:
            import pickle
            from ml_model import AdvancedTradingMLModel
            Path("data/models").mkdir(parents=True, exist_ok=True)
            logger.info(f"🎓 Entrenando modelo ML para {ticker}...")
            model   = AdvancedTradingMLModel(prediction_days=5, threshold=2.0)
            metrics = model.train(data, test_size=0.2)
            if model.is_trained:
                self.ml_models[ticker] = model
                with open(f"data/models/{ticker}.pkl", "wb") as f:
                    pickle.dump(model, f)
                logger.info(f"✅ Modelo {ticker} — Accuracy: {metrics.get('accuracy',0)*100:.1f}%")
        except Exception as e:
            logger.warning(f"⚠️  No se pudo entrenar modelo para {ticker}: {e}")

    # ── SCAN ─────────────────────────────────────────────────────────────────

    def run_scan(self):
        if CONFIG["market_hours_only"] and not is_market_open() and not is_pre_market():
            logger.debug("🔒 Mercado cerrado. Scan omitido.")
            return

        watchlist = self.db.get_watchlist()
        if not watchlist:
            logger.warning("⚠️  Watchlist vacía.")
            return

        now_str = datetime.now(pytz.timezone(CONFIG["timezone"])).strftime("%H:%M:%S %Z")
        logger.info(f"🔍 Iniciando scan — {now_str} — {len(watchlist)} activos")

        results = []
        errors  = 0

        for item in watchlist:
            ticker   = item["ticker"]
            category = item["category"]

            try:
                result = self._analyze_ticker(ticker, category)
                if result:
                    results.append(result)
                    self.db.save_scan_result(ticker, result)
                    self._evaluate_alerts(ticker, result)
            except Exception as e:
                errors += 1
                self.stats["errors"] += 1
                logger.error(f"❌ Error en {ticker}: {e}")
                logger.debug(traceback.format_exc())

        if results:
            df   = pd.DataFrame(results).sort_values("score", ascending=False)
            top3 = df.head(3)[["ticker", "score", "recommendation", "price"]].to_string(index=False)
            logger.info(f"\n{'─'*50}\n📊 TOP 3 PICKS:\n{top3}\n{'─'*50}")

            # ── CAMBIO 3: Ejecutar ciclo de trading con resultados del scan ───
            # Solo durante horario regular de mercado (no pre-market)
            if is_market_open():
                self.trader.run(
                    scan_results=results,
                    ml_models=self.ml_models,
                )

                # Registrar equity para performance tracking
                try:
                    trader_status = self.trader.get_status()
                    if trader_status.get("active"):
                        self.perf_tracker.record_equity(
                            equity=trader_status.get("equity", 0),
                            cash=trader_status.get("cash", 0),
                            n_positions=trader_status.get("open_positions", 0),
                        )
                except Exception as e:
                    logger.debug(f"Performance tracking: {e}")

        self.stats["scans_completed"] += 1
        logger.info(
            f"✅ Scan #{self.stats['scans_completed']} completado — "
            f"{len(results)} analizados, {errors} errores"
        )

        if self.stats["scans_completed"] % 100 == 0:
            self.db.cleanup_old_data(CONFIG["results_retention_days"])

    def _analyze_ticker(self, ticker: str, category: str) -> Optional[Dict]:
        period   = "3mo" if category == "stock" else "3mo"   # fix BTC datos insuficientes
        raw_data = self.fetcher.get_stock_data(ticker, period=period)

        if raw_data is None or raw_data.empty or len(raw_data) < 30:
            logger.warning(f"⚠️  Datos insuficientes para {ticker}")
            return None

        data_processed = DataProcessor.prepare_full_analysis(raw_data, self.analyzer)
        analysis       = self.analyzer.analyze_asset(data_processed, ticker)

        if not analysis:
            return None

        signals   = DataProcessor.get_latest_signals(data_processed)
        ml_result = self._get_ml_prediction(ticker, data_processed)

        # Persistir la señal ML en Supabase (ml_signals) — antes se calculaba
        # pero nunca se guardaba, por eso la tabla estaba siempre vacía.
        if ml_result:
            self.db.save_ml_signal(ticker, ml_result)

        result = {
            "ticker":         ticker,
            "category":       category,
            "price":          round(float(signals.get("price", 0)), 4),
            "change_pct":     round(float(signals.get("price_change_pct", 0)), 4),
            "score":          int(analysis["signals"].get("score", 0)),
            "recommendation": analysis["signals"].get("recommendation", "MANTENER"),
            "rsi":            round(float(signals.get("rsi", 0)), 2),
            "adx":            round(float(signals.get("adx", 0)), 2),
            "macd_hist":      round(float(signals.get("macd_hist", 0)), 6),
            "rvol":           round(float(signals.get("rvol", 0)), 2),
            "atr":            round(float(signals.get("atr", 0)), 4),
            "regime":         signals.get("trend", "UNKNOWN"),
            "trend_strength": signals.get("trend_strength", "WEAK"),
            "ml_prob_up":     ml_result.get("probability_up") if ml_result else None,
            "ml_rec":         ml_result.get("recommendation") if ml_result else None,
            "scanned_at":     datetime.now(pytz.timezone(CONFIG["timezone"])).isoformat(),
            # FIX: antes no se pasaba "extra", así que raw_data en Supabase
            # quedaba siempre vacío pese a tener la predicción ML disponible.
            "extra": {
                "ml_prob_up": ml_result.get("probability_up") if ml_result else None,
                "ml_rec":     ml_result.get("recommendation") if ml_result else None,
            },
        }

        logger.info(
            f"   {ticker:8s} | ${result['price']:>8.2f} | "
            f"Score: {result['score']:>4d} | {result['recommendation']:<15s} | "
            f"RSI: {result['rsi']:>5.1f} | RVOL: {result['rvol']:>4.2f}x"
        )
        return result

    def _get_ml_prediction(self, ticker: str, data: pd.DataFrame) -> Optional[Dict]:
        if ticker not in self.ml_models:
            return None
        try:
            return self.ml_models[ticker].predict(data)
        except Exception as e:
            logger.debug(f"ML predict falló para {ticker}: {e}")
            return None

    # ── ALERTAS ───────────────────────────────────────────────────────────────

    def _evaluate_alerts(self, ticker: str, result: Dict):
        score     = result.get("score", 0)
        rvol      = result.get("rvol", 0)
        ml_up     = result.get("ml_prob_up")
        threshold = CONFIG["alert_score_threshold"]

        if score >= threshold:
            self.notifier.notify(ticker, result, "COMPRA_TECNICA")
            self.stats["alerts_sent"] += 1
        elif score <= -threshold:
            self.notifier.notify(ticker, result, "VENTA_TECNICA")
            self.stats["alerts_sent"] += 1

        if ml_up is not None:
            ml_thr = CONFIG["alert_ml_probability"]
            if ml_up >= ml_thr:
                self.notifier.notify(ticker, result, "ML_COMPRA")
            elif ml_up <= (1 - ml_thr):
                self.notifier.notify(ticker, result, "ML_VENTA")

        if rvol >= CONFIG["alert_rvol_threshold"] and abs(score) >= 30:
            self.notifier.notify(ticker, result, "VOLUMEN_ANOMALO")

    # ── ML RETRAIN ────────────────────────────────────────────────────────────

    def run_ml_retrain(self):
        logger.info("🎓 Iniciando re-entrenamiento de modelos ML...")
        for item in self.db.get_watchlist():
            ticker = item["ticker"]
            try:
                data = self.fetcher.get_stock_data(ticker, period="1y")
                if data is not None and len(data) >= 200:
                    data_p = DataProcessor.prepare_full_analysis(data, self.analyzer)
                    self.train_and_save_model(ticker, data_p)
                else:
                    logger.warning(f"⚠️  {ticker}: datos insuficientes para entrenar")
            except Exception as e:
                logger.error(f"❌ ML retrain falló para {ticker}: {e}")
        logger.info("✅ Re-entrenamiento completado")

    # ── AGENTE ────────────────────────────────────────────────────────────────

    def run_agent(self):
        logger.info("🤖 Ejecutando ciclo del Agente Proactivo...")
        try:
            summary = self.agent.run()
            if not summary:
                return

            added   = summary.get("added", [])
            removed = summary.get("removed", [])
            total   = summary.get("total_after", 0)

            if added or removed:
                msg = "🤖 *Agente Proactivo — Cambios en Watchlist*\n\n"
                if added:
                    msg += f"✅ *Agregados ({len(added)}):*\n"
                    for t in added:
                        msg += f"  • {t}: {summary['added_reasons'].get(t,'')}\n"
                if removed:
                    msg += f"\n🗑️ *Eliminados ({len(removed)}):*\n"
                    for t in removed:
                        msg += f"  • {t}: {summary['removed_reasons'].get(t,'')}\n"
                msg += f"\n📋 Total watchlist: {total}/50"

                self.notifier.send_telegram(msg, "AGENT", "watchlist_update")
                self.db.save_alert("AGENT", "watchlist_update", msg, "agent")

            self.db.sync_watchlist_from_json()
            logger.info(f"✅ Agente completado — Watchlist: {total} activos")

        except Exception as e:
            logger.error(f"❌ Error en agente proactivo: {e}")
            logger.debug(traceback.format_exc())

    # ── STATUS ────────────────────────────────────────────────────────────────

    def print_status(self):
        uptime   = datetime.now() - datetime.fromisoformat(self.stats["started_at"])
        hours, r = divmod(int(uptime.total_seconds()), 3600)
        minutes  = r // 60

        # Incluir estado del portafolio en el status
        trader_status = self.trader.get_status()
        equity_str = f"${trader_status.get('equity', 0):,.2f}" if trader_status.get("active") else "N/A"
        positions  = trader_status.get("open_positions", 0)
        trades_hoy = trader_status.get("trades_today", 0)

        logger.info(
            f"\n{'═'*50}\n"
            f"🦆 STATUS — {hours}h {minutes}m uptime\n"
            f"   Scans:      {self.stats['scans_completed']} | "
            f"Alertas: {self.stats['alerts_sent']} | "
            f"Errores: {self.stats['errors']}\n"
            f"   Modelos ML: {len(self.ml_models)}\n"
            f"   Mercado:    {'🟢 ABIERTO' if is_market_open() else '🔴 CERRADO'}\n"
            f"   💼 Paper Trading:\n"
            f"      Equity:     {equity_str}\n"
            f"      Posiciones: {positions} abiertas\n"
            f"      Trades hoy: {trades_hoy}\n"
            f"{'═'*50}"
        )

        # Reporte de performance real
        logger.info(self.perf_tracker.format_report())

    # ── SETUP Y RUN ───────────────────────────────────────────────────────────

    def setup_schedule(self):
        interval      = CONFIG["scan_interval_minutes"]
        retrain_hours = CONFIG["ml_retrain_hours"]
        agent_hours   = CONFIG["agent_interval_hours"]

        schedule.every(interval).minutes.do(self.run_scan)
        schedule.every(retrain_hours).hours.do(self.run_ml_retrain)
        schedule.every(agent_hours).hours.do(self.run_agent)
        schedule.every().day.at("09:25").do(self.run_agent)
        schedule.every(1).hours.do(self.print_status)
        schedule.every().day.at("16:05").do(self._send_daily_performance)
        schedule.every().day.at("02:00").do(
            lambda: self.db.cleanup_old_data(CONFIG["results_retention_days"])
        )

        logger.info(f"📅 Jobs programados:")
        logger.info(f"   → Scan + Trading: cada {interval} minutos")
        logger.info(f"   → ML retrain: cada {retrain_hours} horas")
        logger.info(f"   → Agente proactivo: cada {agent_hours} horas + 09:25 ET")
        logger.info(f"   → Performance report: diario a las 16:05 ET")
        logger.info(f"   → Limpieza BD: diaria a las 02:00")

    def _send_daily_performance(self):
        """Envía reporte diario de performance por Telegram al cierre."""
        try:
            metrics = self.perf_tracker.get_metrics()
            if metrics.get("status") == "insufficient_data":
                return

            msg = (
                f"📊 *PERFORMANCE DIARIA — Paper Trading*\n\n"
                f"💰 Equity: ${metrics['current_equity']:,.2f}\n"
                f"📈 Retorno total: {metrics['total_return_pct']:+.2f}%\n"
                f"📉 Drawdown actual: {metrics['current_drawdown_pct']:.2f}%\n"
                f"📐 Sharpe: {metrics['sharpe_ratio']:.3f} "
                f"(30d: {metrics['sharpe_30d']:.3f})\n"
                f"🎯 Win rate: {metrics['win_rate_pct']:.1f}%\n"
                f"💹 Profit factor: {metrics['profit_factor']:.2f}\n"
                f"📋 Trades cerrados: {metrics['total_trades']}\n"
                f"📅 Días trackeados: {metrics['days_tracked']}"
            )
            self.notifier.send_telegram(msg, "PERF", "daily_performance")
        except Exception as e:
            logger.error(f"❌ Error enviando performance: {e}")

    def run(self):
        logger.info("\n" + "═"*50)
        logger.info("🚀 PATO QUANT SCHEDULER ARRANCANDO")
        logger.info("═"*50)

        self.load_ml_models()
        self.ensure_ml_models()
        self.setup_schedule()

        logger.info("⚡ Ejecutando primer scan...")
        self.run_scan()

        logger.info("🤖 Ejecutando primer ciclo del agente...")
        self.run_agent()

        logger.info(f"⏳ Scheduler corriendo. Próximo scan en {CONFIG['scan_interval_minutes']} min.")
        logger.info("   Presiona Ctrl+C para detener.\n")

        try:
            while True:
                schedule.run_pending()
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("\n🛑 Scheduler detenido.")
            self.print_status()
        except Exception as e:
            logger.critical(f"💥 Error crítico: {e}")
            logger.critical(traceback.format_exc())
            raise


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scheduler = QuantScheduler()
    scheduler.run()
