"""
ADVANCED MACHINE LEARNING MODULE - Ensemble Predictor
Sistema avanzado que combina Random Forest + XGBoost + Logistic Regression
FIXES 2026-03-17:
  - LR dentro de VotingClassifier usa Pipeline con StandardScaler (consistencia)
  - Threshold dinámico basado en ATR del activo
  - Target sin data leakage (mantiene corrección anterior)
  - Split temporal sin shuffle (mantiene corrección anterior)
  - TimeSeriesSplit walk-forward (mantiene corrección anterior)
  - Features reducidos de 70+ a ~20 (anti-overfitting, más robusto)
  - RF con menos árboles y profundidad limitada para generalización
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, Optional
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no disponible. Usando solo RF + LR")


class AdvancedTradingMLModel:
    """
    Modelo avanzado de ML con ensemble de múltiples algoritmos.
    - Random Forest
    - XGBoost (si disponible)
    - Logistic Regression (dentro de Pipeline con StandardScaler)

    FIX principal: LR ahora usa sklearn Pipeline internamente, así que el
    VotingClassifier escala los datos de LR de forma consistente tanto en
    entrenamiento como en predicción. Ya no hay dos caminos divergentes.
    """

    def __init__(self, prediction_days: int = 5, threshold: float = 2.0):
        self.prediction_days    = prediction_days
        self.threshold          = threshold       # puede sobreescribirse dinámicamente
        self.ensemble_model     = None
        self.rf_model           = None
        self.xgb_model          = None
        self.lr_pipeline        = None            # FIX: Pipeline en lugar de modelo suelto
        self.feature_importance = None
        self.is_trained         = False
        self.training_date      = None
        self.model_metrics      = {}

    # ─────────────────────────────────────────────────────────────────────────
    # THRESHOLD DINÁMICO — FIX
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def compute_dynamic_threshold(data: pd.DataFrame,
                                   base_threshold: float = 2.0) -> float:
        """
        FIX: Threshold fijo del 2% ignora la volatilidad del activo.
        NVDA con ATR del 4% diario nunca genera señales con threshold=2%.
        BTC necesita umbrales mucho mayores.

        Calcula un threshold adaptado a la volatilidad reciente del activo:
          threshold = max(base, ATR_pct * 1.5)

        Ejemplos típicos:
          Acción estable  (ATR 1%):  threshold = max(2.0, 1.5) = 2.0%
          Acción volátil  (ATR 3%):  threshold = max(2.0, 4.5) = 4.5%
          Crypto          (ATR 6%):  threshold = max(2.0, 9.0) = 9.0%
        """
        try:
            # ATR de los últimos 14 días
            high_low    = data['High'] - data['Low']
            high_close  = (data['High'] - data['Close'].shift()).abs()
            low_close   = (data['Low']  - data['Close'].shift()).abs()
            true_range  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr         = true_range.ewm(alpha=1/14, adjust=False).mean().iloc[-1]
            atr_pct     = atr / data['Close'].iloc[-1] * 100
            dynamic     = round(max(base_threshold, atr_pct * 1.5), 2)
            return dynamic
        except Exception:
            return base_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURES AVANZADOS
    # ─────────────────────────────────────────────────────────────────────────

    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features reducidos a ~20 para evitar overfitting.
        Solo se mantienen los más robustos y con mayor poder predictivo:
          - RSI 14 (momentum principal)
          - Distancia a SMA20 y SMA50 (tendencia)
          - SMA Cross 50/200 (golden/death cross)
          - ATR 14 normalizado (volatilidad)
          - Volatilidad histórica 20d (régimen de vol)
          - RVOL 20 (volumen relativo)
          - Returns 1d, 5d, 20d (momentum multi-escala)
          - Position in 20D range (soporte/resistencia)
          - MACD Hist, ADX, StochRSI, BB_Width (ya calculados)
        """
        data = df.copy()

        # RSI 14 (el estándar — un solo periodo es suficiente)
        if 'RSI_14' not in data.columns:
            delta = data['Close'].diff()
            gain  = delta.where(delta > 0, 0).ewm(alpha=1/14, adjust=False).mean()
            loss  = (-delta.where(delta < 0, 0)).ewm(alpha=1/14, adjust=False).mean()
            rs    = gain / (loss + 1e-9)
            data['RSI_14'] = 100 - (100 / (1 + rs))

        # Distancias a SMA20 y SMA50 (las dos más predictivas)
        for period in [20, 50]:
            col = f'SMA{period}'
            if col not in data.columns:
                data[col] = data['Close'].rolling(period).mean()
            data[f'Dist_SMA{period}'] = (data['Close'] / data[col] - 1) * 100

        # Golden/Death cross
        if 'SMA50' not in data.columns:
            data['SMA50'] = data['Close'].rolling(50).mean()
        if 'SMA200' not in data.columns:
            data['SMA200'] = data['Close'].rolling(200).mean()
        data['SMA_Cross_50_200'] = (
            (data['SMA50'] > data['SMA200']).astype(int) -
            (data['SMA50'] < data['SMA200']).astype(int)
        )

        # ATR 14 normalizado (un solo periodo)
        if 'ATR_14' not in data.columns or 'ATR_14_Norm' not in data.columns:
            hl  = data['High'] - data['Low']
            hc  = (data['High'] - data['Close'].shift()).abs()
            lc  = (data['Low']  - data['Close'].shift()).abs()
            tr  = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            data['ATR_14']      = tr.ewm(alpha=1/14, adjust=False).mean()
        data['ATR_14_Norm'] = data['ATR_14'] / data['Close']

        # Volatilidad histórica 20d
        if 'Returns' not in data.columns:
            data['Returns'] = data['Close'].pct_change()
        data['HV_20'] = data['Returns'].rolling(20).std() * np.sqrt(252) * 100

        # RVOL 20 (un solo periodo)
        if 'RVOL_20' not in data.columns:
            avg = data['Volume'].rolling(20).mean()
            data['RVOL_20'] = data['Volume'] / (avg + 1e-9)

        # Retornos: 1d, 5d, 20d (corto, medio, largo)
        for period in [1, 5, 20]:
            data[f'Return_{period}D'] = data['Close'].pct_change(period) * 100

        # Posición en rango de 20 días (¿cerca de soporte o resistencia?)
        data['High_20D'] = data['High'].rolling(20).max()
        data['Low_20D']  = data['Low'].rolling(20).min()
        data['Position_in_Range_20D'] = (
            (data['Close'] - data['Low_20D']) /
            (data['High_20D'] - data['Low_20D'] + 1e-9)
        )

        return data

    # ─────────────────────────────────────────────────────────────────────────
    # PREPARAR DATOS DE ENTRENAMIENTO
    # ─────────────────────────────────────────────────────────────────────────

    def prepare_training_data(self,
                               data: pd.DataFrame,
                               dynamic_threshold: bool = True
                               ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepara features y target.

        FIX threshold dinámico: si dynamic_threshold=True, calcula el umbral
        adaptado a la volatilidad del activo antes de crear las etiquetas.
        """
        print("🔬 Creando features avanzados...")
        df = self.create_advanced_features(data)

        # ── Threshold dinámico ────────────────────────────────────────────────
        if dynamic_threshold:
            self.threshold = self.compute_dynamic_threshold(data, self.threshold)
            print(f"   Threshold dinámico: {self.threshold:.2f}% "
                  f"(ajustado por volatilidad del activo)")

        # ── Target sin data leakage ───────────────────────────────────────────
        # Future_Return[t] = (Close[t+N] - Close[t]) / Close[t]
        # Los features del día t no contienen información de días futuros.
        df['Future_Return'] = (
            df['Close'].shift(-self.prediction_days) / df['Close'] - 1
        ) * 100
        df['Target'] = (df['Future_Return'] > self.threshold).astype(int)

        # ~20 features robustos — anti-overfitting
        feature_columns = [
            'RSI_14',
            'Dist_SMA20', 'Dist_SMA50',
            'SMA_Cross_50_200',
            'ATR_14_Norm',
            'HV_20',
            'RVOL_20',
            'Return_1D', 'Return_5D', 'Return_20D',
            'Position_in_Range_20D',
            'MACD_Hist', 'ADX', 'StochRSI', 'BB_Width',
        ]

        feature_columns = [c for c in feature_columns if c in df.columns]
        print(f"✅ Features preparados: {len(feature_columns)}")

        df_clean = df[feature_columns + ['Target']].dropna()
        return df_clean[feature_columns], df_clean['Target']

    # ─────────────────────────────────────────────────────────────────────────
    # ENTRENAMIENTO
    # ─────────────────────────────────────────────────────────────────────────

    def train(self, data: pd.DataFrame,
              test_size: float = 0.2,
              optimize_hyperparameters: bool = False) -> Dict:
        print("\n" + "="*70)
        print("🚀 ENTRENANDO MODELO AVANZADO (ENSEMBLE)")
        print("="*70 + "\n")

        X, y = self.prepare_training_data(data, dynamic_threshold=True)

        if len(X) < 100:
            raise ValueError(f"Datos insuficientes. Mínimo 100, tienes {len(X)}")

        print(f"📊 Dataset: {len(X)} muestras")
        print(f"   Clase 1 (subida): {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
        print(f"   Clase 0 (no sube): {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.1f}%)")

        # ── Split temporal sin shuffle ─────────────────────────────────────────
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        print(f"   Train: {len(X_train)} | Test: {len(X_test)} (más recientes)")

        # ── Random Forest ─────────────────────────────────────────────────────
        print("\n🌲 1/3 Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=8,
            min_samples_split=20, min_samples_leaf=10,
            random_state=42, n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        rf_score = self.rf_model.score(X_test, y_test)
        print(f"   RF Accuracy: {rf_score*100:.1f}%")

        # ── XGBoost ───────────────────────────────────────────────────────────
        xgb_score = None
        if XGBOOST_AVAILABLE:
            print("\n🚀 2/3 XGBoost...")
            self.xgb_model = XGBClassifier(
                n_estimators=80, max_depth=4,
                learning_rate=0.05, subsample=0.7,
                colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42,
                use_label_encoder=False, eval_metric='logloss', n_jobs=-1
            )
            self.xgb_model.fit(X_train, y_train)
            xgb_score = self.xgb_model.score(X_test, y_test)
            print(f"   XGB Accuracy: {xgb_score*100:.1f}%")

        # ── Logistic Regression con Pipeline ─────────────────────────────────
        # FIX: LR ahora vive dentro de un Pipeline(scaler + lr).
        # Esto garantiza que el VotingClassifier escala los datos
        # internamente de forma consistente. Ya no hay dos caminos
        # divergentes entre train y predict.
        print("\n📊 3/3 Logistic Regression (Pipeline con StandardScaler)...")
        self.lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=1.0, max_iter=1000,
                                       random_state=42, n_jobs=-1))
        ])
        self.lr_pipeline.fit(X_train, y_train)
        lr_score = self.lr_pipeline.score(X_test, y_test)
        print(f"   LR Accuracy: {lr_score*100:.1f}%")

        # ── Ensemble VotingClassifier ─────────────────────────────────────────
        print("\n🎭 Creando Ensemble (Voting soft)...")
        estimators = [
            ('rf', self.rf_model),
            ('lr', self.lr_pipeline),   # FIX: Pipeline, no modelo suelto
        ]
        if XGBOOST_AVAILABLE:
            estimators.insert(1, ('xgb', self.xgb_model))

        self.ensemble_model = VotingClassifier(
            estimators=estimators, voting='soft', n_jobs=-1
        )
        self.ensemble_model.fit(X_train, y_train)
        print("   Ensemble entrenado!")

        # ── Evaluación ────────────────────────────────────────────────────────
        y_pred       = self.ensemble_model.predict(X_test)
        y_pred_proba = self.ensemble_model.predict_proba(X_test)[:, 1]

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc_roc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            auc_roc = 0.5

        # Feature importance
        rf_imp = pd.DataFrame({'feature': X.columns,
                                'importance': self.rf_model.feature_importances_})
        if XGBOOST_AVAILABLE and self.xgb_model:
            xgb_imp = pd.DataFrame({'feature': X.columns,
                                     'importance': self.xgb_model.feature_importances_})
            self.feature_importance = pd.DataFrame({
                'feature':    X.columns,
                'importance': (rf_imp['importance'].values +
                               xgb_imp['importance'].values) / 2
            }).sort_values('importance', ascending=False)
        else:
            self.feature_importance = rf_imp.sort_values('importance', ascending=False)

        # ── Walk-Forward CV ───────────────────────────────────────────────────
        print("\n🔄 Validación cruzada Walk-Forward (TimeSeriesSplit, 5 folds)...")
        tscv      = TimeSeriesSplit(n_splits=5)
        cv_scores = cross_val_score(self.ensemble_model, X_train, y_train,
                                    cv=tscv, scoring='accuracy', n_jobs=-1)
        print(f"   Walk-Forward CV: {cv_scores.mean()*100:.1f}% "
              f"(±{cv_scores.std()*100:.1f}%)")

        self.model_metrics = {
            'accuracy':   accuracy,
            'precision':  precision,
            'recall':     recall,
            'f1_score':   f1,
            'auc_roc':    auc_roc,
            'cv_mean':    cv_scores.mean(),
            'cv_std':     cv_scores.std(),
            'train_size': len(X_train),
            'test_size':  len(X_test),
            'n_features': len(X.columns),
            'features':   list(X.columns),
            'rf_accuracy': rf_score,
            'lr_accuracy': lr_score,
            'threshold':   self.threshold,
        }
        if xgb_score is not None:
            self.model_metrics['xgb_accuracy'] = xgb_score

        self.is_trained    = True
        self.training_date = datetime.now()

        print("\n" + "="*70)
        print("🏆 RESULTADOS DEL ENSEMBLE")
        print("="*70)
        print(f"{'Random Forest':<22} {rf_score*100:>6.1f}%")
        if xgb_score:
            print(f"{'XGBoost':<22} {xgb_score*100:>6.1f}%")
        print(f"{'Logistic Regression':<22} {lr_score*100:>6.1f}%")
        print(f"{'ENSEMBLE (Voting)':<22} {accuracy*100:>6.1f}%")
        print(f"Precision: {precision*100:.1f}% | Recall: {recall*100:.1f}% "
              f"| F1: {f1*100:.1f}% | AUC: {auc_roc:.3f}")
        print(f"CV Walk-Forward: {cv_scores.mean()*100:.1f}% "
              f"(±{cv_scores.std()*100:.1f}%)")
        print(f"Threshold usado: {self.threshold:.2f}%")
        print("="*70)

        print("\n🔥 Top 5 Features:")
        for _, row in self.feature_importance.head(5).iterrows():
            print(f"   {row['feature']:<32} {row['importance']*100:>5.1f}%")

        return self.model_metrics

    # ─────────────────────────────────────────────────────────────────────────
    # PREDICCIÓN
    # ─────────────────────────────────────────────────────────────────────────

    def predict(self, data: pd.DataFrame) -> Dict:
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")

        df             = self.create_advanced_features(data)
        feature_columns = self.model_metrics['features']
        X_current      = df[feature_columns].iloc[[-1]]

        proba      = self.ensemble_model.predict_proba(X_current)[0]
        pred_class = self.ensemble_model.predict(X_current)[0]
        prob_down  = proba[0]
        prob_up    = proba[1]

        # Predicciones individuales
        # FIX: lr_pipeline ya incluye el scaler — no hay que escalar manualmente
        rf_proba  = self.rf_model.predict_proba(X_current)[0][1]
        lr_proba  = self.lr_pipeline.predict_proba(X_current)[0][1]

        individual_preds = {'rf': rf_proba, 'lr': lr_proba}
        if XGBOOST_AVAILABLE and self.xgb_model:
            individual_preds['xgb'] = self.xgb_model.predict_proba(X_current)[0][1]

        confidence   = max(prob_up, prob_down)
        model_probs  = list(individual_preds.values())
        agreement    = 1 - (np.std(model_probs) / 0.5)

        if confidence > 0.75 and agreement > 0.8:  confidence_level = "MUY ALTA"
        elif confidence > 0.70:                     confidence_level = "ALTA"
        elif confidence > 0.60:                     confidence_level = "MEDIA"
        else:                                        confidence_level = "BAJA"

        if prob_up > 0.70:    recommendation = "COMPRA FUERTE"
        elif prob_up > 0.60:  recommendation = "COMPRA"
        elif prob_up < 0.30:  recommendation = "VENTA FUERTE"
        elif prob_up < 0.40:  recommendation = "VENTA"
        else:                 recommendation = "MANTENER"

        # FIX: sklearn/numpy devuelven numpy.float64/int64, no tipos nativos
        # de Python. psycopg2 no sabe adaptar numpy.float64 — al insertarlo
        # en una columna NUMERIC manda literalmente el texto "np.float64(...)"
        # y Postgres lo rechaza ("schema np does not exist"). Eso tumbaba el
        # análisis completo del ticker en scheduler.py::_analyze_ticker()
        # (save_ml_signal se llama antes de construir el resultado, y su
        # excepción se propagaba). Casteamos todo a tipos nativos acá, en el
        # origen, para que ningún consumidor de predict() pise esto de nuevo.
        return {
            'probability_up':        float(prob_up),
            'probability_down':      float(prob_down),
            'predicted_class':       int(pred_class),
            'recommendation':        recommendation,
            'confidence':            float(confidence),
            'confidence_level':      confidence_level,
            'model_agreement':       float(agreement),
            'individual_predictions': {k: float(v) for k, v in individual_preds.items()},
            'prediction_days':       self.prediction_days,
            'threshold':             float(self.threshold),
            'model_accuracy':        float(self.model_metrics['accuracy']),
            'ensemble_used':         True,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # UTILIDADES
    # ─────────────────────────────────────────────────────────────────────────

    def get_feature_importance(self) -> pd.DataFrame:
        if self.feature_importance is None:
            raise ValueError("Modelo no entrenado")
        return self.feature_importance

    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("Modelo no entrenado")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'ensemble_model':     self.ensemble_model,
                'rf_model':           self.rf_model,
                'xgb_model':          self.xgb_model,
                'lr_pipeline':        self.lr_pipeline,   # FIX: guardar pipeline
                'feature_importance': self.feature_importance,
                'model_metrics':      self.model_metrics,
                'training_date':      self.training_date,
                'prediction_days':    self.prediction_days,
                'threshold':          self.threshold,
            }, f)
        print(f"✅ Modelo ensemble guardado: {filepath}")

    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Archivo no encontrado: {filepath}")
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.ensemble_model     = d['ensemble_model']
        self.rf_model           = d['rf_model']
        self.xgb_model          = d.get('xgb_model')
        # FIX: compatibilidad hacia atrás — si el pkl viejo tiene 'lr_model'
        # en lugar de 'lr_pipeline', envolverlo en un pipeline dummy
        self.lr_pipeline        = d.get('lr_pipeline') or d.get('lr_model')
        self.feature_importance = d['feature_importance']
        self.model_metrics      = d['model_metrics']
        self.training_date      = d['training_date']
        self.prediction_days    = d['prediction_days']
        self.threshold          = d['threshold']
        self.is_trained         = True
        print(f"✅ Modelo cargado: {filepath}")
        print(f"   Entrenado: {self.training_date.strftime('%Y-%m-%d %H:%M')}")
        print(f"   Accuracy: {self.model_metrics['accuracy']*100:.1f}%")
        print(f"   Threshold: {self.threshold:.2f}%")


# ─────────────────────────────────────────────────────────────────────────────
# ALIASES DE COMPATIBILIDAD
# ─────────────────────────────────────────────────────────────────────────────

TradingMLModel = AdvancedTradingMLModel


def train_advanced_ml_model(ticker: str, data_processed: pd.DataFrame,
                             prediction_days: int = 5) -> AdvancedTradingMLModel:
    print(f"\n{'='*70}\n🤖 ENTRENANDO ENSEMBLE PARA {ticker}\n{'='*70}\n")
    model = AdvancedTradingMLModel(prediction_days=prediction_days, threshold=2.0)
    try:
        model.train(data_processed, test_size=0.2)
        return model
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def train_ml_model_for_ticker(ticker: str, data_processed: pd.DataFrame,
                               prediction_days: int = 5) -> AdvancedTradingMLModel:
    return train_advanced_ml_model(ticker, data_processed, prediction_days)


def get_ml_prediction(model: AdvancedTradingMLModel,
                       data_processed: pd.DataFrame) -> Dict:
    if model is None or not model.is_trained:
        return None
    try:
        return model.predict(data_processed)
    except Exception as e:
        print(f"❌ Error en predicción: {e}")
        return None


def format_ml_output(prediction: Dict, ticker: str) -> str:
    if prediction is None:
        return "⚠️ No hay predicción disponible"

    emoji = ("🟢" if "COMPRA" in prediction['recommendation'] else
             "🔴" if "VENTA"  in prediction['recommendation'] else "🟡")

    output = f"""
## 🤖 Predicción Machine Learning - {ticker}

### Probabilidades
- **📈 Subida en {prediction['prediction_days']} días:** {prediction['probability_up']*100:.1f}%
- **📉 Bajada en {prediction['prediction_days']} días:** {prediction['probability_down']*100:.1f}%

### Recomendación
{emoji} **{prediction['recommendation']}**

### Confianza del Modelo
- Nivel: {prediction['confidence_level']}
- Score: {prediction['confidence']*100:.1f}%
- Accuracy del modelo: {prediction['model_accuracy']*100:.1f}%
- Threshold usado: {prediction['threshold']:.2f}%
"""
    if prediction.get('individual_predictions'):
        output += "\n### Predicciones individuales\n"
        indiv = prediction['individual_predictions']
        if 'rf'  in indiv: output += f"- Random Forest: {indiv['rf']*100:.1f}%\n"
        if 'xgb' in indiv: output += f"- XGBoost: {indiv['xgb']*100:.1f}%\n"
        if 'lr'  in indiv: output += f"- Logistic Regression: {indiv['lr']*100:.1f}%\n"

        ag = prediction.get('model_agreement', 0)
        if ag > 0.85:   output += f"\n✅ Alto acuerdo entre modelos ({ag*100:.0f}%)"
        elif ag > 0.70: output += f"\nℹ️ Acuerdo moderado ({ag*100:.0f}%)"
        else:           output += f"\n⚠️ Bajo acuerdo ({ag*100:.0f}%) — precaución"

    return output
