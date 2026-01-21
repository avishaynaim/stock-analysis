# Output Format & Example

## 1. Overview

This document defines the final research output format for the stock analysis system. The output is designed for both programmatic consumption (JSON) and human readability (formatted text/HTML).

### Design Principles

1. **Complete:** All relevant analysis information included
2. **Structured:** Well-defined schema for programmatic access
3. **Explainable:** Clear reasoning for scores and recommendations
4. **Actionable:** Risk flags and confidence levels guide decision-making
5. **Reproducible:** Metadata enables exact replication

---

## 2. JSON Schema

### 2.1 Complete Ticker Analysis Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://stock-analysis/schemas/ticker-analysis.json",
  "title": "TickerAnalysis",
  "description": "Complete analysis output for a single ticker",
  "type": "object",
  "required": [
    "ticker",
    "as_of_date",
    "score",
    "score_label",
    "subscores",
    "metadata"
  ],
  "properties": {
    "ticker": {
      "type": "string",
      "description": "Stock ticker symbol",
      "pattern": "^[A-Z]{1,5}$",
      "examples": ["AAPL", "MSFT", "GOOGL"]
    },
    "company_name": {
      "type": "string",
      "description": "Full company name",
      "examples": ["Apple Inc.", "Microsoft Corporation"]
    },
    "sector": {
      "type": "string",
      "description": "GICS sector classification",
      "examples": ["Technology", "Healthcare", "Financials"]
    },
    "industry": {
      "type": "string",
      "description": "GICS industry classification",
      "examples": ["Consumer Electronics", "Software", "Banks"]
    },
    "as_of_date": {
      "type": "string",
      "format": "date-time",
      "description": "Point-in-time date for analysis (ISO 8601)"
    },
    "score": {
      "type": "number",
      "minimum": 0,
      "maximum": 10,
      "description": "Final composite score (0-10 scale)"
    },
    "score_label": {
      "type": "string",
      "enum": ["EXCEPTIONAL", "STRONG", "MODERATE", "WEAK", "POOR", "CRITICAL"],
      "description": "Human-readable score classification"
    },
    "subscores": {
      "$ref": "#/$defs/Subscores"
    },
    "probabilities": {
      "$ref": "#/$defs/Probabilities"
    },
    "risk_assessment": {
      "$ref": "#/$defs/RiskAssessment"
    },
    "price_context": {
      "$ref": "#/$defs/PriceContext"
    },
    "indicators": {
      "$ref": "#/$defs/Indicators"
    },
    "explanation": {
      "$ref": "#/$defs/Explanation"
    },
    "metadata": {
      "$ref": "#/$defs/Metadata"
    }
  },
  "$defs": {
    "Subscores": {
      "type": "object",
      "description": "Component subscores contributing to final score",
      "required": ["trend", "momentum", "volume", "relative_strength", "fundamental", "edge"],
      "properties": {
        "trend": {
          "$ref": "#/$defs/SubscoreDetail"
        },
        "momentum": {
          "$ref": "#/$defs/SubscoreDetail"
        },
        "volume": {
          "$ref": "#/$defs/SubscoreDetail"
        },
        "relative_strength": {
          "$ref": "#/$defs/SubscoreDetail"
        },
        "fundamental": {
          "$ref": "#/$defs/SubscoreDetail"
        },
        "edge": {
          "$ref": "#/$defs/SubscoreDetail"
        }
      }
    },
    "SubscoreDetail": {
      "type": "object",
      "required": ["value", "weight", "weighted_contribution"],
      "properties": {
        "value": {
          "type": "number",
          "minimum": 0,
          "maximum": 10,
          "description": "Raw subscore value (0-10)"
        },
        "weight": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Weight applied to this subscore"
        },
        "weighted_contribution": {
          "type": "number",
          "description": "Contribution to final score (value * weight)"
        },
        "components": {
          "type": "object",
          "description": "Individual indicator contributions to subscore",
          "additionalProperties": {
            "type": "number"
          }
        },
        "assessment": {
          "type": "string",
          "description": "Brief text assessment of this subscore"
        }
      }
    },
    "Probabilities": {
      "type": "object",
      "description": "Probability estimates for different horizons",
      "additionalProperties": {
        "$ref": "#/$defs/ProbabilityEstimate"
      }
    },
    "ProbabilityEstimate": {
      "type": "object",
      "required": ["horizon_days", "target_gain", "probability", "confidence"],
      "properties": {
        "horizon_days": {
          "type": "integer",
          "description": "Forward-looking period in trading days"
        },
        "target_gain": {
          "type": "number",
          "description": "Target gain threshold (e.g., 0.10 for 10%)"
        },
        "probability": {
          "type": "number",
          "minimum": 0,
          "maximum": 1,
          "description": "Estimated probability of achieving target"
        },
        "confidence": {
          "type": "string",
          "enum": ["high", "medium", "low"],
          "description": "Confidence level in estimate"
        },
        "confidence_interval": {
          "type": "object",
          "properties": {
            "lower": {"type": "number"},
            "upper": {"type": "number"},
            "level": {"type": "number", "description": "Confidence level (e.g., 0.95)"}
          }
        },
        "sample_size": {
          "type": "integer",
          "description": "Number of similar historical states used"
        },
        "estimator_breakdown": {
          "type": "object",
          "description": "Individual estimator contributions",
          "properties": {
            "empirical": {"type": "number"},
            "supervised": {"type": "number"},
            "similarity": {"type": "number"}
          }
        }
      }
    },
    "RiskAssessment": {
      "type": "object",
      "required": ["total_penalty", "risk_flags", "risk_level"],
      "properties": {
        "total_penalty": {
          "type": "number",
          "description": "Total penalty applied to score"
        },
        "risk_level": {
          "type": "string",
          "enum": ["low", "moderate", "elevated", "high", "extreme"],
          "description": "Overall risk classification"
        },
        "penalties": {
          "type": "object",
          "properties": {
            "volatility": {
              "$ref": "#/$defs/PenaltyDetail"
            },
            "drawdown": {
              "$ref": "#/$defs/PenaltyDetail"
            },
            "liquidity": {
              "$ref": "#/$defs/PenaltyDetail"
            },
            "gap_risk": {
              "$ref": "#/$defs/PenaltyDetail"
            },
            "regime": {
              "$ref": "#/$defs/PenaltyDetail"
            }
          }
        },
        "risk_flags": {
          "type": "array",
          "items": {
            "$ref": "#/$defs/RiskFlag"
          }
        }
      }
    },
    "PenaltyDetail": {
      "type": "object",
      "properties": {
        "value": {
          "type": "number",
          "description": "Penalty amount"
        },
        "triggered": {
          "type": "boolean",
          "description": "Whether penalty was triggered"
        },
        "threshold": {
          "type": "number",
          "description": "Threshold that triggered penalty"
        },
        "actual_value": {
          "type": "number",
          "description": "Actual measured value"
        }
      }
    },
    "RiskFlag": {
      "type": "object",
      "required": ["flag_id", "severity", "message"],
      "properties": {
        "flag_id": {
          "type": "string",
          "description": "Unique identifier for risk flag type"
        },
        "severity": {
          "type": "string",
          "enum": ["info", "warning", "caution", "danger"],
          "description": "Severity level of risk flag"
        },
        "category": {
          "type": "string",
          "enum": ["volatility", "liquidity", "momentum", "fundamental", "technical", "regime", "event"],
          "description": "Category of risk"
        },
        "message": {
          "type": "string",
          "description": "Human-readable risk description"
        },
        "details": {
          "type": "object",
          "description": "Additional context for the risk flag"
        }
      }
    },
    "PriceContext": {
      "type": "object",
      "description": "Current price and context information",
      "properties": {
        "current_price": {
          "type": "number"
        },
        "currency": {
          "type": "string",
          "default": "USD"
        },
        "change_1d": {
          "type": "number",
          "description": "1-day price change (%)"
        },
        "change_5d": {
          "type": "number",
          "description": "5-day price change (%)"
        },
        "change_21d": {
          "type": "number",
          "description": "21-day price change (%)"
        },
        "change_63d": {
          "type": "number",
          "description": "63-day price change (%)"
        },
        "change_252d": {
          "type": "number",
          "description": "252-day price change (%)"
        },
        "high_52w": {
          "type": "number",
          "description": "52-week high"
        },
        "low_52w": {
          "type": "number",
          "description": "52-week low"
        },
        "pct_from_52w_high": {
          "type": "number",
          "description": "Percentage below 52-week high"
        },
        "avg_volume_20d": {
          "type": "number",
          "description": "20-day average daily volume"
        },
        "market_cap": {
          "type": "number",
          "description": "Market capitalization"
        }
      }
    },
    "Indicators": {
      "type": "object",
      "description": "Detailed indicator values by timeframe and group",
      "additionalProperties": {
        "type": "object",
        "description": "Indicators for a specific timeframe",
        "additionalProperties": {
          "type": "object",
          "description": "Indicator group values"
        }
      }
    },
    "Explanation": {
      "type": "object",
      "description": "Human-readable explanations",
      "properties": {
        "summary": {
          "type": "string",
          "description": "One-paragraph analysis summary"
        },
        "bull_case": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Key bullish factors"
        },
        "bear_case": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Key bearish factors"
        },
        "key_levels": {
          "type": "object",
          "properties": {
            "support": {
              "type": "array",
              "items": {"type": "number"}
            },
            "resistance": {
              "type": "array",
              "items": {"type": "number"}
            }
          }
        },
        "catalyst_watch": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Upcoming events or catalysts"
        }
      }
    },
    "Metadata": {
      "type": "object",
      "required": ["analysis_timestamp", "model_version", "data_version"],
      "properties": {
        "analysis_timestamp": {
          "type": "string",
          "format": "date-time",
          "description": "When analysis was performed"
        },
        "model_version": {
          "type": "string",
          "description": "Version of ML models used"
        },
        "data_version": {
          "type": "string",
          "description": "Version of data snapshot"
        },
        "config_hash": {
          "type": "string",
          "description": "Hash of configuration used"
        },
        "data_range": {
          "type": "object",
          "properties": {
            "start": {"type": "string", "format": "date"},
            "end": {"type": "string", "format": "date"},
            "trading_days": {"type": "integer"}
          }
        },
        "processing_time_ms": {
          "type": "integer",
          "description": "Analysis processing time in milliseconds"
        },
        "warnings": {
          "type": "array",
          "items": {"type": "string"},
          "description": "Non-critical warnings during analysis"
        }
      }
    }
  }
}
```

### 2.2 Scan Result Schema

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://stock-analysis/schemas/scan-result.json",
  "title": "ScanResult",
  "description": "Result of universe scan operation",
  "type": "object",
  "required": ["candidates", "scan_metadata"],
  "properties": {
    "candidates": {
      "type": "array",
      "items": {
        "$ref": "ticker-analysis.json"
      },
      "description": "Ranked list of ticker analyses"
    },
    "scan_metadata": {
      "type": "object",
      "properties": {
        "universe": {"type": "string"},
        "as_of_date": {"type": "string", "format": "date-time"},
        "horizon": {"type": "string"},
        "min_score_threshold": {"type": "number"},
        "raw_universe_size": {"type": "integer"},
        "filtered_universe_size": {"type": "integer"},
        "analyzed_count": {"type": "integer"},
        "candidates_count": {"type": "integer"},
        "error_count": {"type": "integer"},
        "scan_duration_seconds": {"type": "number"},
        "filters_applied": {
          "type": "object",
          "additionalProperties": true
        }
      }
    },
    "summary_statistics": {
      "type": "object",
      "properties": {
        "avg_score": {"type": "number"},
        "median_score": {"type": "number"},
        "score_std": {"type": "number"},
        "sector_distribution": {
          "type": "object",
          "additionalProperties": {"type": "integer"}
        },
        "score_distribution": {
          "type": "object",
          "additionalProperties": {"type": "integer"}
        }
      }
    }
  }
}
```

---

## 3. Example Filled Output

### 3.1 Complete JSON Example

```json
{
  "ticker": "NVDA",
  "company_name": "NVIDIA Corporation",
  "sector": "Technology",
  "industry": "Semiconductors",
  "as_of_date": "2024-01-18T16:00:00-05:00",

  "score": 8.7,
  "score_label": "STRONG",

  "subscores": {
    "trend": {
      "value": 9.2,
      "weight": 0.18,
      "weighted_contribution": 1.656,
      "components": {
        "price_vs_sma20": 0.92,
        "price_vs_sma50": 0.88,
        "price_vs_sma200": 0.95,
        "sma20_vs_sma50": 0.85,
        "sma50_vs_sma200": 0.90,
        "adx_strength": 0.78
      },
      "assessment": "Strong uptrend across all timeframes with healthy ADX"
    },
    "momentum": {
      "value": 8.5,
      "weight": 0.18,
      "weighted_contribution": 1.530,
      "components": {
        "rsi_14": 0.72,
        "macd_histogram": 0.85,
        "roc_21": 0.88,
        "stochastic_k": 0.68,
        "williams_r": 0.70,
        "cci": 0.75
      },
      "assessment": "Positive momentum, RSI not yet overbought"
    },
    "volume": {
      "value": 8.0,
      "weight": 0.12,
      "weighted_contribution": 0.960,
      "components": {
        "volume_sma_ratio": 1.35,
        "obv_trend": 0.82,
        "accumulation_distribution": 0.78,
        "mfi": 0.65,
        "vwap_position": 0.85
      },
      "assessment": "Above-average volume confirming price action"
    },
    "relative_strength": {
      "value": 9.5,
      "weight": 0.12,
      "weighted_contribution": 1.140,
      "components": {
        "rs_vs_spy_21d": 1.85,
        "rs_vs_spy_63d": 2.12,
        "rs_vs_sector_21d": 1.45,
        "rs_percentile_252d": 0.95
      },
      "assessment": "Exceptional relative strength vs market and sector"
    },
    "fundamental": {
      "value": 7.2,
      "weight": 0.08,
      "weighted_contribution": 0.576,
      "components": {
        "pe_vs_sector": 0.55,
        "revenue_growth": 0.92,
        "earnings_surprise": 0.88,
        "margin_trend": 0.75,
        "roe": 0.82
      },
      "assessment": "Strong growth metrics, premium valuation justified by growth"
    },
    "edge": {
      "value": 8.8,
      "weight": 0.32,
      "weighted_contribution": 2.816,
      "components": {
        "probability_5d": 0.68,
        "probability_21d": 0.72,
        "probability_63d": 0.65,
        "historical_edge": 0.85
      },
      "assessment": "High probability of continued outperformance"
    }
  },

  "probabilities": {
    "5d": {
      "horizon_days": 5,
      "target_gain": 0.05,
      "probability": 0.68,
      "confidence": "high",
      "confidence_interval": {
        "lower": 0.62,
        "upper": 0.74,
        "level": 0.95
      },
      "sample_size": 1247,
      "estimator_breakdown": {
        "empirical": 0.65,
        "supervised": 0.71,
        "similarity": 0.68
      }
    },
    "21d": {
      "horizon_days": 21,
      "target_gain": 0.10,
      "probability": 0.58,
      "confidence": "high",
      "confidence_interval": {
        "lower": 0.52,
        "upper": 0.64,
        "level": 0.95
      },
      "sample_size": 892,
      "estimator_breakdown": {
        "empirical": 0.55,
        "supervised": 0.62,
        "similarity": 0.57
      }
    },
    "63d": {
      "horizon_days": 63,
      "target_gain": 0.15,
      "probability": 0.52,
      "confidence": "medium",
      "confidence_interval": {
        "lower": 0.44,
        "upper": 0.60,
        "level": 0.95
      },
      "sample_size": 534,
      "estimator_breakdown": {
        "empirical": 0.48,
        "supervised": 0.55,
        "similarity": 0.53
      }
    }
  },

  "risk_assessment": {
    "total_penalty": 0.32,
    "risk_level": "moderate",
    "penalties": {
      "volatility": {
        "value": 0.15,
        "triggered": true,
        "threshold": 0.90,
        "actual_value": 0.92
      },
      "drawdown": {
        "value": 0.0,
        "triggered": false,
        "threshold": -0.15,
        "actual_value": -0.08
      },
      "liquidity": {
        "value": 0.0,
        "triggered": false,
        "threshold": 500000,
        "actual_value": 45000000
      },
      "gap_risk": {
        "value": 0.12,
        "triggered": true,
        "threshold": 0.05,
        "actual_value": 0.062
      },
      "regime": {
        "value": 0.05,
        "triggered": true,
        "threshold": null,
        "actual_value": null
      }
    },
    "risk_flags": [
      {
        "flag_id": "HIGH_VOLATILITY",
        "severity": "warning",
        "category": "volatility",
        "message": "Volatility in 92nd percentile (annualized: 48%)",
        "details": {
          "current_volatility": 0.48,
          "percentile": 92,
          "sector_avg": 0.32
        }
      },
      {
        "flag_id": "ELEVATED_GAP_RISK",
        "severity": "caution",
        "category": "volatility",
        "message": "Average overnight gap of 6.2% over past month",
        "details": {
          "avg_gap": 0.062,
          "max_gap_30d": 0.12,
          "earnings_in_30d": true
        }
      },
      {
        "flag_id": "EARNINGS_UPCOMING",
        "severity": "info",
        "category": "event",
        "message": "Earnings announcement expected in 12 days",
        "details": {
          "expected_date": "2024-01-30",
          "days_until": 12
        }
      }
    ]
  },

  "price_context": {
    "current_price": 547.20,
    "currency": "USD",
    "change_1d": 0.028,
    "change_5d": 0.085,
    "change_21d": 0.152,
    "change_63d": 0.287,
    "change_252d": 2.35,
    "high_52w": 560.00,
    "low_52w": 138.84,
    "pct_from_52w_high": -0.023,
    "avg_volume_20d": 42500000,
    "market_cap": 1350000000000
  },

  "indicators": {
    "1d": {
      "trend": {
        "sma_20": 525.40,
        "sma_50": 498.20,
        "sma_200": 412.80,
        "ema_12": 538.50,
        "ema_26": 515.30,
        "adx": 38.5,
        "plus_di": 32.1,
        "minus_di": 12.4
      },
      "momentum": {
        "rsi_14": 68.2,
        "macd": 23.20,
        "macd_signal": 18.45,
        "macd_histogram": 4.75,
        "stochastic_k": 82.5,
        "stochastic_d": 78.2,
        "williams_r": -17.5,
        "roc_21": 15.2,
        "cci": 125.8
      },
      "volatility": {
        "atr_14": 18.50,
        "atr_percent": 0.034,
        "bollinger_upper": 575.20,
        "bollinger_lower": 495.60,
        "bollinger_width": 0.148,
        "historical_volatility_21": 0.45,
        "implied_volatility": 0.52
      },
      "volume": {
        "volume_sma_20_ratio": 1.35,
        "obv": 2850000000,
        "obv_slope": 0.82,
        "accumulation_distribution": 0.78,
        "mfi_14": 65.2,
        "vwap": 542.80
      }
    },
    "1w": {
      "trend": {
        "sma_10": 518.50,
        "sma_20": 485.20,
        "ema_12": 522.30
      },
      "momentum": {
        "rsi_14": 72.5,
        "macd": 35.80,
        "roc_4": 8.5
      }
    }
  },

  "explanation": {
    "summary": "NVIDIA shows exceptional strength with a score of 8.7/10, driven by powerful relative strength (9.5), strong trend alignment (9.2), and high probability estimates across multiple horizons. The stock is outperforming the S&P 500 by a factor of 2.1x over the past quarter and maintains healthy momentum without being severely overbought. Key risks include elevated volatility (92nd percentile) and upcoming earnings, which could introduce significant price gaps. The strong fundamentals, including 92% revenue growth momentum, support the premium valuation.",

    "bull_case": [
      "Exceptional relative strength: outperforming SPY by 2.1x over 63 days",
      "Strong trend: price above all major moving averages with rising ADX (38.5)",
      "Above-average volume confirming price action (1.35x 20-day average)",
      "High probability of 5%+ gain in next 5 days (68% confidence)",
      "Dominant market position in AI/GPU computing"
    ],

    "bear_case": [
      "Elevated volatility (48% annualized) in 92nd percentile",
      "Premium valuation: P/E above sector average",
      "Earnings event risk in 12 days could cause significant gap",
      "RSI approaching overbought territory (68.2)",
      "Large overnight gaps averaging 6.2% create execution risk"
    ],

    "key_levels": {
      "support": [525.40, 498.20, 475.00],
      "resistance": [560.00, 575.20, 600.00]
    },

    "catalyst_watch": [
      "Q4 2024 Earnings: Expected January 30, 2024",
      "CES 2024 announcements may impact sentiment",
      "AI chip demand forecasts from hyperscalers"
    ]
  },

  "metadata": {
    "analysis_timestamp": "2024-01-18T16:05:32.847Z",
    "model_version": "2.1.0",
    "data_version": "2024-01-18",
    "config_hash": "a8f3b2c1d4e5f6a7b8c9d0e1f2a3b4c5",
    "data_range": {
      "start": "2021-01-18",
      "end": "2024-01-18",
      "trading_days": 756
    },
    "processing_time_ms": 342,
    "warnings": []
  }
}
```

### 3.2 Compact Output Example

```json
{
  "ticker": "NVDA",
  "as_of_date": "2024-01-18",
  "score": 8.7,
  "score_label": "STRONG",
  "subscores": {
    "trend": 9.2,
    "momentum": 8.5,
    "volume": 8.0,
    "relative_strength": 9.5,
    "fundamental": 7.2,
    "edge": 8.8
  },
  "probability_21d": 0.58,
  "risk_level": "moderate",
  "risk_flags": ["HIGH_VOLATILITY", "ELEVATED_GAP_RISK", "EARNINGS_UPCOMING"],
  "price": 547.20,
  "change_21d": 0.152
}
```

---

## 4. Explanation Text Structure

### 4.1 Explanation Generation Framework

```python
class ExplanationGenerator:
    """
    Generate human-readable explanations from analysis results.
    """

    def __init__(self, config: ExplanationConfig):
        self.config = config
        self.templates = self._load_templates()

    def generate(self, analysis: TickerAnalysis) -> Explanation:
        """Generate complete explanation from analysis."""

        return Explanation(
            summary=self._generate_summary(analysis),
            bull_case=self._generate_bull_case(analysis),
            bear_case=self._generate_bear_case(analysis),
            key_levels=self._identify_key_levels(analysis),
            catalyst_watch=self._identify_catalysts(analysis)
        )

    def _generate_summary(self, analysis: TickerAnalysis) -> str:
        """
        Generate one-paragraph summary.

        Structure:
        1. Opening: Score and classification
        2. Key drivers: Top 2-3 subscores
        3. Probability context
        4. Risk acknowledgment
        5. Fundamental support
        """

        # Identify top drivers
        top_subscores = sorted(
            analysis.subscores.items(),
            key=lambda x: x[1].value,
            reverse=True
        )[:3]

        # Build summary
        parts = []

        # Opening
        parts.append(
            f"{analysis.ticker} shows {self._score_descriptor(analysis.score)} "
            f"strength with a score of {analysis.score:.1f}/10"
        )

        # Key drivers
        driver_text = ", ".join([
            f"{self._format_subscore_name(name)} ({detail.value:.1f})"
            for name, detail in top_subscores
        ])
        parts.append(f"driven by {driver_text}")

        # Probability
        if analysis.probabilities:
            primary_horizon = '21d'
            if primary_horizon in analysis.probabilities:
                prob = analysis.probabilities[primary_horizon]
                parts.append(
                    f"The stock has a {prob.probability:.0%} probability of "
                    f"achieving {prob.target_gain:.0%}+ gain over {prob.horizon_days} days"
                )

        # Risk acknowledgment
        if analysis.risk_assessment.risk_flags:
            risk_summary = self._summarize_risks(analysis.risk_assessment)
            parts.append(f"Key risks include {risk_summary}")

        # Combine
        return ". ".join(parts) + "."

    def _generate_bull_case(self, analysis: TickerAnalysis) -> List[str]:
        """
        Generate bullish factors.

        Categories to consider:
        - Trend strength
        - Momentum quality
        - Relative performance
        - Volume confirmation
        - Probability edge
        - Fundamental support
        """

        bull_points = []

        # Trend
        if analysis.subscores['trend'].value >= 7.0:
            trend_detail = self._describe_trend(analysis)
            bull_points.append(trend_detail)

        # Relative strength
        if analysis.subscores['relative_strength'].value >= 7.5:
            rs_detail = self._describe_relative_strength(analysis)
            bull_points.append(rs_detail)

        # Volume
        if analysis.subscores['volume'].value >= 6.5:
            vol_detail = self._describe_volume(analysis)
            bull_points.append(vol_detail)

        # Probability
        if analysis.probabilities:
            best_prob = max(
                analysis.probabilities.values(),
                key=lambda p: p.probability
            )
            if best_prob.probability >= 0.60:
                bull_points.append(
                    f"High probability of {best_prob.target_gain:.0%}+ gain "
                    f"in next {best_prob.horizon_days} days "
                    f"({best_prob.probability:.0%} confidence)"
                )

        # Momentum
        if analysis.subscores['momentum'].value >= 7.0:
            mom_detail = self._describe_momentum_bull(analysis)
            bull_points.append(mom_detail)

        return bull_points[:5]  # Limit to top 5

    def _generate_bear_case(self, analysis: TickerAnalysis) -> List[str]:
        """
        Generate bearish factors / risks.

        Categories:
        - Risk flags (converted to text)
        - Weak subscores
        - Overbought conditions
        - Valuation concerns
        - Event risks
        """

        bear_points = []

        # Convert risk flags
        for flag in analysis.risk_assessment.risk_flags:
            bear_points.append(self._risk_flag_to_text(flag))

        # Check for overbought
        if analysis.indicators:
            rsi = analysis.indicators.get('1d', {}).get('momentum', {}).get('rsi_14')
            if rsi and rsi > 70:
                bear_points.append(
                    f"RSI approaching overbought territory ({rsi:.1f})"
                )

        # Weak subscores
        for name, detail in analysis.subscores.items():
            if detail.value < 5.0:
                bear_points.append(
                    f"Weak {self._format_subscore_name(name)} "
                    f"({detail.value:.1f}/10)"
                )

        return bear_points[:5]  # Limit to top 5

    def _identify_key_levels(self, analysis: TickerAnalysis) -> Dict:
        """Identify support and resistance levels."""

        levels = {'support': [], 'resistance': []}

        if not analysis.indicators:
            return levels

        ind = analysis.indicators.get('1d', {})
        price = analysis.price_context.current_price

        # Support levels
        support_candidates = []

        # Moving averages below price
        for ma_key in ['sma_20', 'sma_50', 'sma_200']:
            ma_val = ind.get('trend', {}).get(ma_key)
            if ma_val and ma_val < price:
                support_candidates.append(ma_val)

        # Bollinger lower
        bb_lower = ind.get('volatility', {}).get('bollinger_lower')
        if bb_lower:
            support_candidates.append(bb_lower)

        # Round numbers
        round_support = self._find_round_number_below(price)
        support_candidates.append(round_support)

        levels['support'] = sorted(set(support_candidates), reverse=True)[:3]

        # Resistance levels
        resistance_candidates = []

        # 52-week high
        if analysis.price_context.high_52w:
            resistance_candidates.append(analysis.price_context.high_52w)

        # Bollinger upper
        bb_upper = ind.get('volatility', {}).get('bollinger_upper')
        if bb_upper:
            resistance_candidates.append(bb_upper)

        # Round numbers above
        round_resistance = self._find_round_number_above(price)
        resistance_candidates.append(round_resistance)

        levels['resistance'] = sorted(set(resistance_candidates))[:3]

        return levels

    # Helper methods...

    def _score_descriptor(self, score: float) -> str:
        """Get descriptive word for score level."""
        if score >= 9.0:
            return "exceptional"
        elif score >= 7.5:
            return "strong"
        elif score >= 6.0:
            return "moderate"
        elif score >= 4.0:
            return "weak"
        else:
            return "poor"

    def _format_subscore_name(self, name: str) -> str:
        """Format subscore name for display."""
        return {
            'trend': 'trend alignment',
            'momentum': 'momentum',
            'volume': 'volume profile',
            'relative_strength': 'relative strength',
            'fundamental': 'fundamentals',
            'edge': 'probability edge'
        }.get(name, name)

    def _risk_flag_to_text(self, flag: RiskFlag) -> str:
        """Convert risk flag to readable text."""
        return flag.message
```

### 4.2 Explanation Templates

```python
EXPLANATION_TEMPLATES = {
    # Summary templates by score range
    'summary': {
        'exceptional': (
            "{ticker} exhibits exceptional strength with a score of {score}/10, "
            "ranking in the top tier of analyzed stocks. "
            "{key_drivers}. "
            "{probability_statement}. "
            "{risk_caveat}"
        ),
        'strong': (
            "{ticker} shows strong characteristics with a score of {score}/10, "
            "driven by {key_drivers}. "
            "{probability_statement}. "
            "{risk_caveat}"
        ),
        'moderate': (
            "{ticker} presents a moderate opportunity with a score of {score}/10. "
            "While {positive_factors}, {negative_factors}. "
            "{probability_statement}"
        ),
        'weak': (
            "{ticker} shows weakness with a score of {score}/10. "
            "{negative_factors}. "
            "Consider waiting for improvement in {weak_areas}"
        ),
        'poor': (
            "{ticker} is currently unfavorable with a score of {score}/10. "
            "{negative_factors}. "
            "Significant improvement needed before consideration"
        )
    },

    # Subscore descriptions
    'trend': {
        'strong': "Strong uptrend: price above {ma_list} with rising ADX ({adx})",
        'weak': "Trend weakness: price below {ma_list}, ADX showing indecision ({adx})",
        'neutral': "Mixed trend signals across timeframes"
    },

    'momentum': {
        'strong': "Positive momentum with RSI at {rsi} and bullish MACD",
        'overbought': "Momentum strong but RSI overbought ({rsi}), pullback risk elevated",
        'oversold': "Oversold conditions (RSI {rsi}) may present bounce opportunity",
        'weak': "Momentum deteriorating, MACD histogram declining"
    },

    'relative_strength': {
        'exceptional': "Exceptional relative strength: outperforming {benchmark} by {factor}x over {period}",
        'strong': "Outperforming {benchmark} and sector peers",
        'weak': "Underperforming {benchmark} by {pct}% over {period}",
        'lagging': "Significantly lagging both market and sector"
    },

    'volume': {
        'confirming': "Above-average volume ({ratio}x) confirming price action",
        'diverging': "Volume declining despite price rise - potential warning",
        'accumulation': "Signs of institutional accumulation (OBV rising)",
        'distribution': "Distribution pattern detected in volume profile"
    },

    # Risk flag descriptions
    'risk_flags': {
        'HIGH_VOLATILITY': "Elevated volatility ({percentile}th percentile, {vol}% annualized)",
        'ELEVATED_GAP_RISK': "Average overnight gap of {gap}% over past month",
        'EARNINGS_UPCOMING': "Earnings announcement expected in {days} days",
        'HIGH_SHORT_INTEREST': "Short interest at {si}% of float",
        'INSIDER_SELLING': "Recent insider selling activity detected",
        'SECTOR_WEAKNESS': "Sector showing relative weakness vs market",
        'LOW_LIQUIDITY': "Below-average liquidity may impact execution",
        'EXTREME_RSI': "RSI at extreme level ({rsi}) - mean reversion risk"
    }
}
```

### 4.3 Formatted Text Output

```
═══════════════════════════════════════════════════════════════════════════════
                          NVDA - NVIDIA Corporation
                              Technology | Semiconductors
═══════════════════════════════════════════════════════════════════════════════

OVERALL SCORE: 8.7 / 10  [STRONG]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SUBSCORES                                                    Value    Weight
───────────────────────────────────────────────────────────────────────────────
  Trend              ████████████████████░░░░░░░░░░  9.2      18%     → 1.66
  Momentum           █████████████████░░░░░░░░░░░░░  8.5      18%     → 1.53
  Volume             ████████████████░░░░░░░░░░░░░░  8.0      12%     → 0.96
  Relative Strength  ███████████████████░░░░░░░░░░░  9.5      12%     → 1.14
  Fundamental        ██████████████░░░░░░░░░░░░░░░░  7.2       8%     → 0.58
  Edge               █████████████████░░░░░░░░░░░░░  8.8      32%     → 2.82
                                                    ─────────────────────────
  Subtotal                                                            8.69
  Risk Penalty                                                       -0.32
                                                    ═════════════════════════
  FINAL SCORE                                                         8.37

───────────────────────────────────────────────────────────────────────────────

PROBABILITY ESTIMATES
───────────────────────────────────────────────────────────────────────────────
  Horizon      Target    Probability    Confidence    Sample Size
  ─────────────────────────────────────────────────────────────────
   5 days      +5%          68%           HIGH           1,247
  21 days     +10%          58%           HIGH             892
  63 days     +15%          52%         MEDIUM             534

───────────────────────────────────────────────────────────────────────────────

PRICE CONTEXT
───────────────────────────────────────────────────────────────────────────────
  Current Price:     $547.20                    Market Cap:    $1.35T
  52-Week Range:     $138.84 - $560.00          Avg Volume:    42.5M

  Performance:        1D        5D       21D       63D      252D
                   +2.8%    +8.5%   +15.2%   +28.7%   +235.0%

───────────────────────────────────────────────────────────────────────────────

RISK ASSESSMENT                                          Risk Level: MODERATE
───────────────────────────────────────────────────────────────────────────────
  ⚠ WARNING   Volatility in 92nd percentile (annualized: 48%)
  ⚡ CAUTION   Average overnight gap of 6.2% over past month
  ℹ INFO      Earnings announcement expected in 12 days

  Total Risk Penalty: -0.32 points

───────────────────────────────────────────────────────────────────────────────

KEY LEVELS
───────────────────────────────────────────────────────────────────────────────
  Resistance:  $560.00 (52W High) → $575.20 (BB Upper) → $600.00 (Round)
  Support:     $525.40 (SMA20) → $498.20 (SMA50) → $475.00 (Round)

───────────────────────────────────────────────────────────────────────────────

SUMMARY
───────────────────────────────────────────────────────────────────────────────
NVIDIA shows exceptional strength with a score of 8.7/10, driven by powerful
relative strength (9.5), strong trend alignment (9.2), and high probability
estimates across multiple horizons. The stock is outperforming the S&P 500
by a factor of 2.1x over the past quarter and maintains healthy momentum
without being severely overbought. Key risks include elevated volatility
(92nd percentile) and upcoming earnings, which could introduce significant
price gaps.

BULL CASE                              │ BEAR CASE
───────────────────────────────────────┼───────────────────────────────────────
✓ Exceptional RS: 2.1x vs SPY (63d)   │ ✗ Elevated volatility (48% annual)
✓ Strong trend: above all MAs, ADX 38 │ ✗ Premium valuation vs sector
✓ Volume confirming (1.35x average)   │ ✗ Earnings risk in 12 days
✓ 68% prob of +5% in 5 days           │ ✗ RSI approaching overbought (68)
✓ Dominant AI/GPU market position     │ ✗ 6.2% avg overnight gap risk

───────────────────────────────────────────────────────────────────────────────
Analysis as of: January 18, 2024 4:00 PM ET
Model Version: 2.1.0 | Processing Time: 342ms
═══════════════════════════════════════════════════════════════════════════════
```

---

## 5. Risk Flags & Confidence Logic

### 5.1 Risk Flag Definitions

```python
@dataclass
class RiskFlagDefinition:
    """Definition of a risk flag type."""
    flag_id: str
    name: str
    category: str
    severity: str
    description: str
    detection_logic: str
    threshold: Optional[float]


RISK_FLAG_CATALOG = {
    # =========================================================================
    # VOLATILITY RISK FLAGS
    # =========================================================================

    'HIGH_VOLATILITY': RiskFlagDefinition(
        flag_id='HIGH_VOLATILITY',
        name='High Volatility',
        category='volatility',
        severity='warning',
        description='Stock volatility significantly above historical norm',
        detection_logic='Annualized volatility > 90th percentile of universe',
        threshold=0.90  # Percentile threshold
    ),

    'EXTREME_VOLATILITY': RiskFlagDefinition(
        flag_id='EXTREME_VOLATILITY',
        name='Extreme Volatility',
        category='volatility',
        severity='danger',
        description='Stock volatility at extreme levels',
        detection_logic='Annualized volatility > 98th percentile of universe',
        threshold=0.98
    ),

    'ELEVATED_GAP_RISK': RiskFlagDefinition(
        flag_id='ELEVATED_GAP_RISK',
        name='Elevated Gap Risk',
        category='volatility',
        severity='caution',
        description='High frequency of overnight price gaps',
        detection_logic='Average overnight gap > 5% over past 20 days',
        threshold=0.05
    ),

    'VIX_ELEVATED': RiskFlagDefinition(
        flag_id='VIX_ELEVATED',
        name='Elevated Market Volatility',
        category='volatility',
        severity='caution',
        description='Overall market volatility elevated',
        detection_logic='VIX > 25',
        threshold=25.0
    ),

    # =========================================================================
    # MOMENTUM RISK FLAGS
    # =========================================================================

    'RSI_OVERBOUGHT': RiskFlagDefinition(
        flag_id='RSI_OVERBOUGHT',
        name='RSI Overbought',
        category='momentum',
        severity='caution',
        description='RSI indicates overbought conditions',
        detection_logic='RSI(14) > 75',
        threshold=75.0
    ),

    'RSI_EXTREME_OVERBOUGHT': RiskFlagDefinition(
        flag_id='RSI_EXTREME_OVERBOUGHT',
        name='RSI Extremely Overbought',
        category='momentum',
        severity='warning',
        description='RSI at extreme overbought levels',
        detection_logic='RSI(14) > 85',
        threshold=85.0
    ),

    'RSI_OVERSOLD': RiskFlagDefinition(
        flag_id='RSI_OVERSOLD',
        name='RSI Oversold',
        category='momentum',
        severity='info',
        description='RSI indicates oversold conditions (potential bounce)',
        detection_logic='RSI(14) < 25',
        threshold=25.0
    ),

    'MOMENTUM_DIVERGENCE': RiskFlagDefinition(
        flag_id='MOMENTUM_DIVERGENCE',
        name='Momentum Divergence',
        category='momentum',
        severity='warning',
        description='Price making new highs but momentum indicators declining',
        detection_logic='Price at 20-day high AND RSI declining over 5 days',
        threshold=None
    ),

    # =========================================================================
    # LIQUIDITY RISK FLAGS
    # =========================================================================

    'LOW_LIQUIDITY': RiskFlagDefinition(
        flag_id='LOW_LIQUIDITY',
        name='Low Liquidity',
        category='liquidity',
        severity='caution',
        description='Below-average trading volume may impact execution',
        detection_logic='20-day average dollar volume < $1M',
        threshold=1_000_000
    ),

    'VERY_LOW_LIQUIDITY': RiskFlagDefinition(
        flag_id='VERY_LOW_LIQUIDITY',
        name='Very Low Liquidity',
        category='liquidity',
        severity='warning',
        description='Significantly below-average volume creates execution risk',
        detection_logic='20-day average dollar volume < $500K',
        threshold=500_000
    ),

    'VOLUME_DECLINING': RiskFlagDefinition(
        flag_id='VOLUME_DECLINING',
        name='Declining Volume',
        category='liquidity',
        severity='info',
        description='Volume declining during price advance (potential warning)',
        detection_logic='Price up 5%+ over 10 days AND volume down 20%+',
        threshold=None
    ),

    # =========================================================================
    # TECHNICAL RISK FLAGS
    # =========================================================================

    'BELOW_KEY_MA': RiskFlagDefinition(
        flag_id='BELOW_KEY_MA',
        name='Below Key Moving Average',
        category='technical',
        severity='caution',
        description='Price below important moving average',
        detection_logic='Price < 200-day SMA',
        threshold=None
    ),

    'DEATH_CROSS': RiskFlagDefinition(
        flag_id='DEATH_CROSS',
        name='Death Cross',
        category='technical',
        severity='warning',
        description='50-day MA crossed below 200-day MA',
        detection_logic='SMA(50) < SMA(200) AND was above within 5 days',
        threshold=None
    ),

    'EXTENDED_FROM_MA': RiskFlagDefinition(
        flag_id='EXTENDED_FROM_MA',
        name='Extended from Moving Average',
        category='technical',
        severity='caution',
        description='Price significantly extended above moving average',
        detection_logic='Price > 20% above 50-day SMA',
        threshold=0.20
    ),

    'NEAR_RESISTANCE': RiskFlagDefinition(
        flag_id='NEAR_RESISTANCE',
        name='Near Resistance',
        category='technical',
        severity='info',
        description='Price approaching significant resistance level',
        detection_logic='Price within 2% of 52-week high or identified resistance',
        threshold=0.02
    ),

    # =========================================================================
    # FUNDAMENTAL RISK FLAGS
    # =========================================================================

    'HIGH_VALUATION': RiskFlagDefinition(
        flag_id='HIGH_VALUATION',
        name='High Valuation',
        category='fundamental',
        severity='caution',
        description='Valuation metrics above sector average',
        detection_logic='P/E > 1.5x sector median',
        threshold=1.5
    ),

    'EXTREME_VALUATION': RiskFlagDefinition(
        flag_id='EXTREME_VALUATION',
        name='Extreme Valuation',
        category='fundamental',
        severity='warning',
        description='Valuation metrics significantly above norms',
        detection_logic='P/E > 2.5x sector median OR > 50',
        threshold=2.5
    ),

    'NEGATIVE_EARNINGS': RiskFlagDefinition(
        flag_id='NEGATIVE_EARNINGS',
        name='Negative Earnings',
        category='fundamental',
        severity='caution',
        description='Company currently not profitable',
        detection_logic='TTM EPS < 0',
        threshold=0
    ),

    'EARNINGS_DECLINING': RiskFlagDefinition(
        flag_id='EARNINGS_DECLINING',
        name='Declining Earnings',
        category='fundamental',
        severity='warning',
        description='Earnings trend declining',
        detection_logic='YoY EPS growth < -10%',
        threshold=-0.10
    ),

    # =========================================================================
    # EVENT RISK FLAGS
    # =========================================================================

    'EARNINGS_UPCOMING': RiskFlagDefinition(
        flag_id='EARNINGS_UPCOMING',
        name='Earnings Upcoming',
        category='event',
        severity='info',
        description='Earnings announcement approaching',
        detection_logic='Earnings expected within 14 days',
        threshold=14
    ),

    'EARNINGS_IMMINENT': RiskFlagDefinition(
        flag_id='EARNINGS_IMMINENT',
        name='Earnings Imminent',
        category='event',
        severity='caution',
        description='Earnings announcement very soon',
        detection_logic='Earnings expected within 3 days',
        threshold=3
    ),

    'EX_DIVIDEND_UPCOMING': RiskFlagDefinition(
        flag_id='EX_DIVIDEND_UPCOMING',
        name='Ex-Dividend Upcoming',
        category='event',
        severity='info',
        description='Ex-dividend date approaching',
        detection_logic='Ex-dividend within 7 days',
        threshold=7
    ),

    # =========================================================================
    # REGIME RISK FLAGS
    # =========================================================================

    'SECTOR_WEAKNESS': RiskFlagDefinition(
        flag_id='SECTOR_WEAKNESS',
        name='Sector Weakness',
        category='regime',
        severity='caution',
        description='Sector underperforming broader market',
        detection_logic='Sector RS vs SPY < 0.95 over 21 days',
        threshold=0.95
    ),

    'MARKET_DOWNTREND': RiskFlagDefinition(
        flag_id='MARKET_DOWNTREND',
        name='Market Downtrend',
        category='regime',
        severity='warning',
        description='Broader market in downtrend',
        detection_logic='SPY below 50-day AND 200-day SMA',
        threshold=None
    ),

    'HIGH_CORRELATION': RiskFlagDefinition(
        flag_id='HIGH_CORRELATION',
        name='High Market Correlation',
        category='regime',
        severity='info',
        description='Stock highly correlated with market',
        detection_logic='60-day correlation with SPY > 0.85',
        threshold=0.85
    )
}
```

### 5.2 Risk Flag Detection Logic

```python
class RiskFlagDetector:
    """
    Detect risk flags from analysis data.
    """

    def __init__(self, config: RiskConfig):
        self.config = config
        self.flag_catalog = RISK_FLAG_CATALOG

    def detect_all(
        self,
        analysis: TickerAnalysis,
        market_data: MarketData
    ) -> List[RiskFlag]:
        """Detect all applicable risk flags."""

        flags = []

        # Volatility flags
        flags.extend(self._detect_volatility_flags(analysis))

        # Momentum flags
        flags.extend(self._detect_momentum_flags(analysis))

        # Liquidity flags
        flags.extend(self._detect_liquidity_flags(analysis))

        # Technical flags
        flags.extend(self._detect_technical_flags(analysis))

        # Fundamental flags
        flags.extend(self._detect_fundamental_flags(analysis))

        # Event flags
        flags.extend(self._detect_event_flags(analysis))

        # Regime flags
        flags.extend(self._detect_regime_flags(analysis, market_data))

        # Sort by severity
        severity_order = {'danger': 0, 'warning': 1, 'caution': 2, 'info': 3}
        flags.sort(key=lambda f: severity_order.get(f.severity, 4))

        return flags

    def _detect_volatility_flags(self, analysis: TickerAnalysis) -> List[RiskFlag]:
        """Detect volatility-related flags."""

        flags = []
        ind = analysis.indicators.get('1d', {}).get('volatility', {})

        # Historical volatility
        hist_vol = ind.get('historical_volatility_21')
        vol_percentile = ind.get('volatility_percentile')

        if vol_percentile:
            if vol_percentile >= 0.98:
                flags.append(self._create_flag(
                    'EXTREME_VOLATILITY',
                    details={
                        'percentile': int(vol_percentile * 100),
                        'volatility': hist_vol
                    }
                ))
            elif vol_percentile >= 0.90:
                flags.append(self._create_flag(
                    'HIGH_VOLATILITY',
                    details={
                        'percentile': int(vol_percentile * 100),
                        'volatility': hist_vol
                    }
                ))

        # Gap risk
        avg_gap = ind.get('avg_overnight_gap')
        if avg_gap and avg_gap > 0.05:
            flags.append(self._create_flag(
                'ELEVATED_GAP_RISK',
                details={
                    'avg_gap': avg_gap,
                    'max_gap': ind.get('max_overnight_gap_20d')
                }
            ))

        return flags

    def _detect_momentum_flags(self, analysis: TickerAnalysis) -> List[RiskFlag]:
        """Detect momentum-related flags."""

        flags = []
        ind = analysis.indicators.get('1d', {}).get('momentum', {})

        rsi = ind.get('rsi_14')

        if rsi:
            if rsi > 85:
                flags.append(self._create_flag(
                    'RSI_EXTREME_OVERBOUGHT',
                    details={'rsi': rsi}
                ))
            elif rsi > 75:
                flags.append(self._create_flag(
                    'RSI_OVERBOUGHT',
                    details={'rsi': rsi}
                ))
            elif rsi < 25:
                flags.append(self._create_flag(
                    'RSI_OVERSOLD',
                    details={'rsi': rsi}
                ))

        # Momentum divergence detection
        # (Price making highs, RSI declining)
        # ... implementation

        return flags

    def _create_flag(
        self,
        flag_id: str,
        details: dict = None
    ) -> RiskFlag:
        """Create risk flag from definition."""

        definition = self.flag_catalog[flag_id]

        # Format message with details
        message = self._format_message(definition, details)

        return RiskFlag(
            flag_id=flag_id,
            severity=definition.severity,
            category=definition.category,
            message=message,
            details=details or {}
        )

    def _format_message(
        self,
        definition: RiskFlagDefinition,
        details: dict
    ) -> str:
        """Format risk flag message with details."""

        template = RISK_FLAG_TEMPLATES.get(definition.flag_id)

        if template and details:
            try:
                return template.format(**details)
            except KeyError:
                pass

        return definition.description


RISK_FLAG_TEMPLATES = {
    'HIGH_VOLATILITY': "Volatility in {percentile}th percentile (annualized: {volatility:.0%})",
    'EXTREME_VOLATILITY': "Extreme volatility in {percentile}th percentile ({volatility:.0%} annualized)",
    'ELEVATED_GAP_RISK': "Average overnight gap of {avg_gap:.1%} over past month",
    'RSI_OVERBOUGHT': "RSI overbought at {rsi:.1f}",
    'RSI_EXTREME_OVERBOUGHT': "RSI extremely overbought at {rsi:.1f}",
    'RSI_OVERSOLD': "RSI oversold at {rsi:.1f} - potential bounce candidate",
    'EARNINGS_UPCOMING': "Earnings announcement expected in {days} days",
    'EARNINGS_IMMINENT': "Earnings announcement in {days} days - high event risk",
    'LOW_LIQUIDITY': "Average daily volume ${volume:,.0f} below threshold",
    'HIGH_VALUATION': "P/E of {pe:.1f} is {ratio:.1f}x sector median",
}
```

### 5.3 Confidence Level Logic

```python
class ConfidenceCalculator:
    """
    Calculate confidence levels for probability estimates.

    Confidence is determined by:
    1. Sample size adequacy
    2. Estimator agreement
    3. Regime stability
    4. Data quality
    """

    def __init__(self, config: ConfidenceConfig):
        self.config = config

    def calculate(
        self,
        probability_estimate: float,
        estimator_outputs: Dict[str, float],
        sample_size: int,
        regime_stability: float,
        data_quality_score: float
    ) -> ConfidenceAssessment:
        """
        Calculate overall confidence in probability estimate.

        Returns:
            ConfidenceAssessment with level, interval, and breakdown
        """

        # =====================================================================
        # Factor 1: Sample Size Score (0-1)
        # =====================================================================

        sample_score = self._sample_size_score(sample_size)

        # =====================================================================
        # Factor 2: Estimator Agreement Score (0-1)
        # =====================================================================

        agreement_score = self._estimator_agreement_score(estimator_outputs)

        # =====================================================================
        # Factor 3: Regime Stability Score (0-1)
        # =====================================================================

        regime_score = regime_stability  # Already 0-1

        # =====================================================================
        # Factor 4: Data Quality Score (0-1)
        # =====================================================================

        quality_score = data_quality_score  # Already 0-1

        # =====================================================================
        # Combined Confidence Score
        # =====================================================================

        weights = self.config.confidence_weights
        combined_score = (
            sample_score * weights['sample_size'] +
            agreement_score * weights['agreement'] +
            regime_score * weights['regime'] +
            quality_score * weights['data_quality']
        )

        # =====================================================================
        # Confidence Level Classification
        # =====================================================================

        if combined_score >= 0.75:
            confidence_level = 'high'
        elif combined_score >= 0.50:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'

        # =====================================================================
        # Confidence Interval Calculation
        # =====================================================================

        interval_width = self._calculate_interval_width(
            combined_score,
            sample_size,
            probability_estimate
        )

        confidence_interval = ConfidenceInterval(
            lower=max(0, probability_estimate - interval_width / 2),
            upper=min(1, probability_estimate + interval_width / 2),
            level=0.95
        )

        return ConfidenceAssessment(
            level=confidence_level,
            score=combined_score,
            interval=confidence_interval,
            breakdown={
                'sample_size': sample_score,
                'estimator_agreement': agreement_score,
                'regime_stability': regime_score,
                'data_quality': quality_score
            }
        )

    def _sample_size_score(self, n: int) -> float:
        """
        Score based on sample size adequacy.

        Thresholds:
        - n < 30:  Low confidence (0.0 - 0.3)
        - n < 100: Medium confidence (0.3 - 0.6)
        - n < 500: Good confidence (0.6 - 0.8)
        - n >= 500: High confidence (0.8 - 1.0)
        """

        if n < 30:
            return 0.3 * (n / 30)
        elif n < 100:
            return 0.3 + 0.3 * ((n - 30) / 70)
        elif n < 500:
            return 0.6 + 0.2 * ((n - 100) / 400)
        else:
            return min(1.0, 0.8 + 0.2 * ((n - 500) / 500))

    def _estimator_agreement_score(
        self,
        outputs: Dict[str, float]
    ) -> float:
        """
        Score based on agreement between estimators.

        High agreement = high confidence
        """

        if len(outputs) < 2:
            return 0.5  # Default if only one estimator

        values = list(outputs.values())
        spread = max(values) - min(values)

        # Lower spread = higher agreement
        # Map spread 0.0-0.3 to score 1.0-0.0
        agreement = max(0, 1.0 - spread / 0.3)

        return agreement

    def _calculate_interval_width(
        self,
        confidence_score: float,
        sample_size: int,
        probability: float
    ) -> float:
        """
        Calculate confidence interval width.

        Uses Wilson score interval approximation.
        """

        from scipy import stats

        # Z-score for 95% confidence
        z = 1.96

        # Wilson score interval
        denominator = 1 + z**2 / sample_size
        center = (probability + z**2 / (2 * sample_size)) / denominator
        spread = z * np.sqrt(
            (probability * (1 - probability) + z**2 / (4 * sample_size)) / sample_size
        ) / denominator

        # Adjust by confidence score (wider interval for lower confidence)
        adjustment = 1.0 + (1.0 - confidence_score) * 0.5

        return spread * 2 * adjustment


@dataclass
class ConfidenceAssessment:
    """Complete confidence assessment."""
    level: str  # 'high', 'medium', 'low'
    score: float  # 0-1
    interval: 'ConfidenceInterval'
    breakdown: Dict[str, float]


@dataclass
class ConfidenceInterval:
    """Confidence interval for probability."""
    lower: float
    upper: float
    level: float  # e.g., 0.95 for 95% CI
```

### 5.4 Confidence Level Interpretation Guide

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONFIDENCE LEVEL INTERPRETATION                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  HIGH CONFIDENCE                                                            │
│  ────────────────                                                            │
│  Score: 0.75 - 1.00                                                         │
│                                                                             │
│  Interpretation:                                                            │
│  • Large sample size (500+ similar historical states)                       │
│  • Strong agreement across estimators (spread < 10%)                        │
│  • Stable market regime                                                     │
│  • Complete, high-quality data                                              │
│                                                                             │
│  Action: Probability estimates can be trusted for decision-making           │
│  Interval: Typically ±5-8% around point estimate                            │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────────│
│                                                                             │
│  MEDIUM CONFIDENCE                                                          │
│  ─────────────────                                                           │
│  Score: 0.50 - 0.74                                                         │
│                                                                             │
│  Interpretation:                                                            │
│  • Moderate sample size (100-500 historical states)                         │
│  • Some disagreement between estimators (spread 10-20%)                     │
│  • Some regime uncertainty                                                  │
│  • Minor data quality issues possible                                       │
│                                                                             │
│  Action: Use probability as directional guide, not precise target           │
│  Interval: Typically ±10-15% around point estimate                          │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────────│
│                                                                             │
│  LOW CONFIDENCE                                                             │
│  ──────────────                                                              │
│  Score: 0.00 - 0.49                                                         │
│                                                                             │
│  Interpretation:                                                            │
│  • Small sample size (< 100 historical states)                              │
│  • Significant estimator disagreement (spread > 20%)                        │
│  • Regime transition or instability detected                                │
│  • Data quality concerns                                                    │
│                                                                             │
│  Action: Treat probability as rough estimate only                           │
│  Interval: Typically ±15-25% around point estimate                          │
│  Recommendation: Rely more heavily on other factors (trend, momentum)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Output Format Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         OUTPUT FORMAT SUMMARY                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  JSON SCHEMA                                                                │
│  ───────────                                                                 │
│  • TickerAnalysis: Complete analysis with all components                    │
│  • ScanResult: Ranked list with scan metadata                               │
│  • Nested structures: Subscores, Probabilities, RiskAssessment              │
│  • Full type definitions with validation                                    │
│                                                                             │
│  OUTPUT FORMATS                                                             │
│  ──────────────                                                              │
│  • JSON: Full structured data for programmatic access                       │
│  • JSON Compact: Minimal fields for quick consumption                       │
│  • Text: Formatted console output with visual elements                      │
│  • HTML: Rich formatted reports (via templates)                             │
│  • CSV: Tabular export for spreadsheet analysis                             │
│                                                                             │
│  EXPLANATION STRUCTURE                                                      │
│  ─────────────────────                                                       │
│  • Summary: One-paragraph analysis overview                                 │
│  • Bull Case: 3-5 positive factors                                          │
│  • Bear Case: 3-5 risks/negative factors                                    │
│  • Key Levels: Support and resistance prices                                │
│  • Catalyst Watch: Upcoming events                                          │
│                                                                             │
│  RISK FLAGS                                                                 │
│  ──────────                                                                  │
│  Categories: Volatility, Momentum, Liquidity, Technical,                    │
│              Fundamental, Event, Regime                                     │
│  Severities: danger > warning > caution > info                              │
│  25+ defined flag types with detection logic                                │
│                                                                             │
│  CONFIDENCE LOGIC                                                           │
│  ────────────────                                                            │
│  Factors: Sample size, Estimator agreement, Regime stability, Data quality  │
│  Levels: HIGH (0.75+), MEDIUM (0.50-0.74), LOW (<0.50)                      │
│  Output: Level + Score + Confidence Interval + Factor Breakdown             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Implementation Notes

### Output Generation Code Structure

```
src/stock_analysis/output/
├── __init__.py
├── schema.py              # Pydantic models matching JSON schema
├── formatters/
│   ├── __init__.py
│   ├── json_formatter.py  # JSON output generation
│   ├── text_formatter.py  # Console text output
│   ├── html_formatter.py  # HTML report generation
│   └── csv_formatter.py   # CSV export
├── explanation/
│   ├── __init__.py
│   ├── generator.py       # ExplanationGenerator class
│   └── templates.py       # Explanation templates
├── risk/
│   ├── __init__.py
│   ├── flags.py           # Risk flag definitions
│   ├── detector.py        # RiskFlagDetector class
│   └── confidence.py      # ConfidenceCalculator class
└── serializers.py         # Output serialization utilities
```

---

*Document Version: 1.0*
*Last Updated: 2026-01-18*
