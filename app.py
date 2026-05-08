import base64
import io
import json
import pickle

import dash
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Input, Output, State, dcc, html
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # gunicorn entry point

# ── Styles ───────────────────────────────────────────────────────────────────

FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

CARD = {
    "backgroundColor": "#ffffff",
    "borderRadius": "12px",
    "padding": "28px 32px",
    "marginBottom": "16px",
    "boxShadow": "0 1px 3px rgba(0,0,0,0.07), 0 4px 16px rgba(0,0,0,0.04)",
    "border": "1px solid #f0f0f0",
}

CARD_HIDDEN = {**CARD, "display": "none"}

STEP_NUM = {
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "width": "26px",
    "height": "26px",
    "borderRadius": "50%",
    "backgroundColor": "#6366f1",
    "color": "white",
    "fontSize": "12px",
    "fontWeight": "700",
    "marginRight": "10px",
    "flexShrink": "0",
}

SECTION_TITLE = {
    "fontSize": "15px",
    "fontWeight": "600",
    "color": "#111827",
    "marginBottom": "16px",
    "display": "flex",
    "alignItems": "center",
}

BADGE_BASE = {
    "display": "inline-block",
    "padding": "3px 11px",
    "borderRadius": "20px",
    "fontWeight": "600",
    "fontSize": "12px",
    "letterSpacing": "0.02em",
}

BADGE_CLASS = {**BADGE_BASE, "backgroundColor": "#ede9fe", "color": "#6d28d9"}
BADGE_REG   = {**BADGE_BASE, "backgroundColor": "#d1fae5", "color": "#065f46"}

BTN = {
    "backgroundColor": "#6366f1",
    "color": "white",
    "border": "none",
    "padding": "10px 28px",
    "borderRadius": "8px",
    "cursor": "pointer",
    "fontWeight": "600",
    "fontSize": "14px",
    "letterSpacing": "0.01em",
    "boxShadow": "0 1px 3px rgba(99,102,241,0.4)",
    "transition": "opacity 0.15s",
}

BTN_PREDICT = {**BTN,
    "backgroundColor": "#0ea5e9",
    "boxShadow": "0 1px 3px rgba(14,165,233,0.4)",
    "marginTop": "20px",
}

INPUT_STYLE = {
    "width": "100%",
    "padding": "9px 12px",
    "border": "1px solid #e5e7eb",
    "borderRadius": "8px",
    "fontSize": "14px",
    "color": "#111827",
    "backgroundColor": "#fafafa",
    "boxSizing": "border-box",
    "outline": "none",
}

DIVIDER = {
    "borderTop": "1px solid #f3f4f6",
    "margin": "20px 0",
}

HEADER_STYLE = {
    "background": "linear-gradient(135deg, #6366f1 0%, #4f46e5 100%)",
    "padding": "36px 0 32px",
    "marginBottom": "28px",
}


def step_header(n, label):
    return html.Div([
        html.Span(str(n), style=STEP_NUM),
        html.Span(label),
    ], style=SECTION_TITLE)


# ── Layout ───────────────────────────────────────────────────────────────────

app.layout = html.Div(style={"minHeight": "100vh", "backgroundColor": "#f8f9fb", "fontFamily": FONT}, children=[
    dcc.Store(id="data-store"),
    dcc.Store(id="model-store"),
    dcc.Store(id="task-store"),
    dcc.Store(id="feature-store"),

    # Header banner
    html.Div(style=HEADER_STYLE, children=[
        html.Div(style={"maxWidth": "900px", "margin": "0 auto", "padding": "0 24px"}, children=[
            html.H1("Loan Prediction App",
                    style={"fontSize": "28px", "fontWeight": "800", "color": "white",
                           "margin": "0 0 6px 0", "letterSpacing": "-0.5px"}),
            html.P("Upload a dataset, pick a target, select features, train a model, and predict.",
                   style={"color": "rgba(255,255,255,0.75)", "margin": 0, "fontSize": "15px"}),
        ]),
    ]),

    html.Div(style={"maxWidth": "900px", "margin": "0 auto", "padding": "0 24px 48px"}, children=[

        # ── 1. Upload ──────────────────────────────────────────────────────
        html.Div(style=CARD, children=[
            step_header(1, "Upload Dataset"),
            dcc.Upload(
                id="upload",
                children=html.Div([
                    html.Div("📂", style={"fontSize": "28px", "marginBottom": "8px"}),
                    html.Div([
                        html.Span("Drag & drop a CSV file, or "),
                        html.Span("browse", style={"color": "#6366f1", "fontWeight": "600", "cursor": "pointer"}),
                    ], style={"fontSize": "14px", "color": "#6b7280"}),
                    html.Div(".csv files only", style={"fontSize": "12px", "color": "#9ca3af", "marginTop": "4px"}),
                ], style={"textAlign": "center"}),
                style={
                    "border": "2px dashed #e0e0f0",
                    "borderRadius": "10px",
                    "padding": "36px 24px",
                    "backgroundColor": "#fafaff",
                    "cursor": "pointer",
                    "transition": "border-color 0.2s",
                },
                accept=".csv",
            ),
            html.Div(id="upload-status", style={"marginTop": "12px", "fontSize": "13px"}),
        ]),

        # ── 2. Target ──────────────────────────────────────────────────────
        html.Div(id="section-target", style=CARD_HIDDEN, children=[
            step_header(2, "Select Target Column"),
            html.P("Choose the column you want to predict. The task type (classification or regression) will be detected automatically.",
                   style={"fontSize": "13px", "color": "#6b7280", "marginTop": "-10px", "marginBottom": "14px"}),
            dcc.Dropdown(id="target-dropdown", placeholder="Choose target column…", clearable=False,
                         style={"fontSize": "14px"}),
            html.Div(id="target-stats", style={"marginTop": "20px"}),
        ]),

        # ── 3. Features ────────────────────────────────────────────────────
        html.Div(id="section-features", style=CARD_HIDDEN, children=[
            step_header(3, "Select Features"),
            html.P("Choose the input columns the model will learn from. ID-like columns are excluded automatically.",
                   style={"fontSize": "13px", "color": "#6b7280", "marginTop": "-10px", "marginBottom": "14px"}),
            dcc.Dropdown(id="feature-dropdown", placeholder="Choose one or more feature columns…",
                         multi=True, style={"fontSize": "14px"}),
            html.Div(id="correlation-chart", style={"marginTop": "20px"}),
        ]),

        # ── 4. Train ───────────────────────────────────────────────────────
        html.Div(id="section-train", style=CARD_HIDDEN, children=[
            step_header(4, "Train Model"),
            html.P("Trains a scikit-learn Pipeline (imputation → scaling → model) on an 80/20 split.",
                   style={"fontSize": "13px", "color": "#6b7280", "marginTop": "-10px", "marginBottom": "16px"}),
            html.Button("Train Model", id="train-btn", n_clicks=0, style=BTN),
            html.Div(id="train-results", style={"marginTop": "20px"}),
        ]),

        # ── 5. Predict ─────────────────────────────────────────────────────
        html.Div(id="section-predict", style=CARD_HIDDEN, children=[
            step_header(5, "Make a Prediction"),
            html.P("Fill in the feature values below and click Predict.",
                   style={"fontSize": "13px", "color": "#6b7280", "marginTop": "-10px", "marginBottom": "16px"}),
            html.Div(id="predict-inputs"),
            html.Button("Predict", id="predict-btn", n_clicks=0, style=BTN_PREDICT),
            html.Div(id="predict-output", style={"marginTop": "20px"}),
        ]),
    ]),
])

# ── Helpers ───────────────────────────────────────────────────────────────────

def detect_task(series: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(series) or series.nunique() <= 10:
        return "classification"
    return "regression"


def df_to_store(df: pd.DataFrame) -> str:
    return df.to_json(date_format="iso", orient="split")


def store_to_df(data: str) -> pd.DataFrame:
    return pd.read_json(io.StringIO(data), orient="split")


def encode_model(pipeline) -> str:
    return base64.b64encode(pickle.dumps(pipeline)).decode()


def decode_model(encoded: str):
    return pickle.loads(base64.b64decode(encoded))


# ── Callback 1: Upload CSV ────────────────────────────────────────────────────

@app.callback(
    Output("data-store", "data"),
    Output("upload-status", "children"),
    Output("section-target", "style"),
    Output("target-dropdown", "options"),
    Input("upload", "contents"),
    State("upload", "filename"),
    prevent_initial_call=True,
)
def handle_upload(contents, filename):
    if contents is None:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    try:
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        if df.empty or len(df.columns) < 2:
            raise ValueError("File has no usable data.")
    except Exception as e:
        err = html.Span(f"Error reading file: {e}",
                        style={"color": "#dc2626", "fontWeight": "500"})
        return dash.no_update, err, dash.no_update, dash.no_update

    options = [{"label": c, "value": c} for c in df.columns]
    status = html.Div([
        html.Span("✓ ", style={"color": "#10b981", "fontWeight": "700"}),
        html.Span(f"{filename} — {df.shape[0]:,} rows × {df.shape[1]} columns",
                  style={"color": "#374151", "fontWeight": "500"}),
    ])
    section_style = {**CARD, "display": "block"}

    return df_to_store(df), status, section_style, options


# ── Callback 2: Target selected → stats ───────────────────────────────────────

@app.callback(
    Output("target-stats", "children"),
    Output("task-store", "data"),
    Output("section-features", "style"),
    Output("feature-dropdown", "options"),
    Output("feature-dropdown", "value"),
    Output("section-train", "style", allow_duplicate=True),
    Output("section-predict", "style", allow_duplicate=True),
    Output("model-store", "data", allow_duplicate=True),
    Input("target-dropdown", "value"),
    State("data-store", "data"),
    prevent_initial_call=True,
)
def handle_target(target, data):
    if not target or not data:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    df = store_to_df(data)
    task = detect_task(df[target])

    if task == "classification":
        counts = df[target].value_counts().reset_index()
        counts.columns = [target, "count"]
        fig = px.bar(counts, x=target, y="count",
                     title="Class Distribution",
                     color=target,
                     color_discrete_sequence=["#6366f1", "#a5b4fc", "#c7d2fe"])
        fig.update_layout(showlegend=False, margin=dict(t=40, b=20),
                          plot_bgcolor="white", paper_bgcolor="white",
                          font_family=FONT)
        fig.update_traces(marker_line_width=0)
        badge = html.Span("Classification", style=BADGE_CLASS)
    else:
        mean_val = df[target].mean()
        fig = px.histogram(df, x=target, title=f"{target} Distribution  (mean = {mean_val:.2f})",
                           color_discrete_sequence=["#6366f1"])
        fig.update_layout(margin=dict(t=40, b=20),
                          plot_bgcolor="white", paper_bgcolor="white",
                          font_family=FONT)
        fig.update_traces(marker_line_width=0)
        badge = html.Span("Regression", style=BADGE_REG)

    stats = html.Div([
        html.Div([
            html.Span("Detected task type:", style={"fontSize": "13px", "color": "#6b7280"}),
            html.Span(" "),
            badge,
        ], style={"marginBottom": "16px"}),
        dcc.Graph(figure=fig, config={"displayModeBar": False},
                  style={"borderRadius": "8px", "overflow": "hidden"}),
    ])

    # Exclude target and high-cardinality columns (unique values > 50% of rows)
    # — catches ID-like columns such as Loan_ID that would explode one-hot encoding
    def is_usable(col):
        if col == target:
            return False
        if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() / len(df) > 0.5:
            return False
        return True

    feat_options = [{"label": c, "value": c} for c in df.columns if is_usable(c)]
    section_style = {**CARD, "display": "block"}
    hidden = {**CARD, "display": "none"}

    return stats, task, section_style, feat_options, None, hidden, hidden, None


# ── Callback 3: Features selected → correlation chart ─────────────────────────

@app.callback(
    Output("correlation-chart", "children"),
    Output("feature-store", "data"),
    Output("section-train", "style"),
    Output("section-predict", "style", allow_duplicate=True),
    Output("model-store", "data", allow_duplicate=True),
    Input("feature-dropdown", "value"),
    State("data-store", "data"),
    State("target-dropdown", "value"),
    prevent_initial_call=True,
)
def handle_features(features, data, target):
    if not features or not data or not target:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    df = store_to_df(data)
    encoded = pd.get_dummies(df[features + [target]])
    target_cols = [c for c in encoded.columns if c == target or c.startswith(target + "_")]

    if target_cols:
        target_col = target_cols[0]
        corr = encoded.drop(columns=target_cols).corrwith(encoded[target_col]).abs().sort_values()
        corr_df = corr.reset_index()
        corr_df.columns = ["Feature", "Correlation"]
        fig = px.bar(corr_df, x="Correlation", y="Feature", orientation="h",
                     title="Feature Correlation with Target",
                     color="Correlation",
                     color_continuous_scale=["#e0e7ff", "#6366f1"])
        fig.update_layout(margin=dict(t=40, b=20), coloraxis_showscale=False,
                          plot_bgcolor="white", paper_bgcolor="white",
                          font_family=FONT)
        fig.update_traces(marker_line_width=0)
        chart = dcc.Graph(figure=fig, config={"displayModeBar": False})
    else:
        chart = html.P("Could not compute correlation for the selected target.",
                       style={"color": "#6b7280"})

    return chart, json.dumps(features), {**CARD, "display": "block"}, {**CARD, "display": "none"}, None


# ── Callback 4: Train ─────────────────────────────────────────────────────────

@app.callback(
    Output("train-results", "children"),
    Output("model-store", "data"),
    Output("section-predict", "style"),
    Output("predict-inputs", "children"),
    Input("train-btn", "n_clicks"),
    State("data-store", "data"),
    State("target-dropdown", "value"),
    State("feature-store", "data"),
    State("task-store", "data"),
    prevent_initial_call=True,
)
def handle_train(n_clicks, data, target, features_json, task):
    if not n_clicks or not data or not target or not features_json:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update

    features = json.loads(features_json)
    df = store_to_df(data)

    X = df[features]
    y = df[target]

    if task == "classification":
        y = y.astype(str)

    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(df[c])]

    transformers = []
    if num_cols:
        transformers.append(("num", Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("scale", StandardScaler()),
        ]), num_cols))
    if cat_cols:
        transformers.append(("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols))

    preprocessor = ColumnTransformer(transformers, remainder="drop")
    model = LogisticRegression(max_iter=5000) if task == "classification" else LinearRegression()
    pipeline = Pipeline([("pre", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    if task == "classification":
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        metrics = html.Div([
            _success_banner("Model trained successfully — Logistic Regression"),
            html.Div([
                _metric_box("Accuracy", f"{acc:.4f}", "#6366f1"),
                _metric_box("F1 Score", f"{f1:.4f}", "#0ea5e9"),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ])
    else:
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics = html.Div([
            _success_banner("Model trained successfully — Linear Regression"),
            html.Div([
                _metric_box("R²", f"{r2:.4f}", "#6366f1"),
                _metric_box("RMSE", f"{rmse:.4f}", "#0ea5e9"),
            ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}),
        ])

    predict_inputs = _build_predict_inputs(df, features)

    return metrics, encode_model(pipeline), {**CARD, "display": "block"}, predict_inputs


def _success_banner(text):
    return html.Div([
        html.Span("✓  ", style={"fontWeight": "800"}),
        html.Span(text),
    ], style={
        "backgroundColor": "#f0fdf4",
        "border": "1px solid #bbf7d0",
        "borderRadius": "8px",
        "padding": "10px 16px",
        "color": "#166534",
        "fontWeight": "600",
        "fontSize": "13px",
        "marginBottom": "16px",
    })


def _metric_box(label, value, accent="#6366f1"):
    return html.Div([
        html.Div(label, style={"fontSize": "11px", "fontWeight": "600", "letterSpacing": "0.06em",
                               "textTransform": "uppercase", "color": "#9ca3af", "marginBottom": "6px"}),
        html.Div(value, style={"fontSize": "28px", "fontWeight": "800", "color": accent,
                               "letterSpacing": "-0.5px"}),
    ], style={
        "backgroundColor": "#fafafa",
        "border": "1px solid #e5e7eb",
        "borderRadius": "10px",
        "padding": "16px 24px",
        "minWidth": "130px",
    })


def _build_predict_inputs(df, features):
    # Two-column grid layout
    inputs = []
    for col in features:
        label = html.Label(col, style={
            "fontSize": "12px", "fontWeight": "600", "letterSpacing": "0.04em",
            "textTransform": "uppercase", "color": "#6b7280",
            "marginBottom": "6px", "display": "block",
        })
        if not pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = sorted(df[col].dropna().unique().tolist())
            control = dcc.Dropdown(
                id={"type": "pred-input", "col": col},
                options=[{"label": v, "value": v} for v in unique_vals],
                placeholder=f"Select…",
                style={"fontSize": "14px"},
            )
        else:
            control = dcc.Input(
                id={"type": "pred-input", "col": col},
                type="number",
                placeholder=f"e.g. {df[col].median():.0f}",
                style=INPUT_STYLE,
            )
        inputs.append(html.Div([label, control], style={
            "marginBottom": "16px",
            "flex": "1 1 calc(50% - 8px)",
            "minWidth": "200px",
        }))
    return html.Div(inputs, style={"display": "flex", "flexWrap": "wrap", "gap": "0 16px"})


# ── Callback 5: Predict ───────────────────────────────────────────────────────

@app.callback(
    Output("predict-output", "children"),
    Input("predict-btn", "n_clicks"),
    State({"type": "pred-input", "col": dash.ALL}, "value"),
    State({"type": "pred-input", "col": dash.ALL}, "id"),
    State("model-store", "data"),
    State("task-store", "data"),
    State("feature-store", "data"),
    prevent_initial_call=True,
)
def handle_predict(n_clicks, values, ids, model_encoded, task, features_json):
    if not n_clicks or not model_encoded:
        return dash.no_update

    features = json.loads(features_json)
    col_map = {id_obj["col"]: val for id_obj, val in zip(ids, values)}

    if any(col_map.get(f) is None for f in features):
        return html.P("Please fill in all fields before predicting.",
                      style={"color": "#b45309", "fontWeight": "500"})

    row = {f: [col_map[f]] for f in features}
    X_new = pd.DataFrame(row)

    pipeline = decode_model(model_encoded)
    prediction = pipeline.predict(X_new)[0]

    if task == "classification":
        result_label = "Predicted Class"
        result_value = str(prediction)
        accent = "#6366f1"
        confidence_block = html.Div()
        if hasattr(pipeline.named_steps["model"], "predict_proba"):
            proba = pipeline.predict_proba(X_new)[0]
            classes = pipeline.named_steps["model"].classes_
            sorted_pairs = sorted(zip(classes, proba), key=lambda x: -x[1])
            bars = []
            for cls, p in sorted_pairs:
                bars.append(html.Div([
                    html.Div([
                        html.Span(str(cls), style={"fontSize": "13px", "fontWeight": "600",
                                                   "color": "#374151", "minWidth": "40px"}),
                        html.Span(f"{p:.1%}", style={"fontSize": "13px", "color": "#6b7280",
                                                     "marginLeft": "auto"}),
                    ], style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px"}),
                    html.Div(html.Div(style={
                        "height": "6px", "borderRadius": "3px",
                        "backgroundColor": "#6366f1", "width": f"{p*100:.1f}%",
                    }), style={"backgroundColor": "#e0e7ff", "borderRadius": "3px", "height": "6px"}),
                ], style={"marginBottom": "10px"}))
            confidence_block = html.Div([
                html.Div("Class Probabilities", style={"fontSize": "12px", "fontWeight": "600",
                         "color": "#9ca3af", "letterSpacing": "0.05em",
                         "textTransform": "uppercase", "marginBottom": "10px"}),
                *bars,
            ], style={"marginTop": "16px"})
    else:
        result_label = "Predicted Value"
        result_value = f"{prediction:.4f}"
        accent = "#0ea5e9"
        confidence_block = html.Div()

    return html.Div([
        html.Div([
            html.Div(result_label, style={"fontSize": "11px", "fontWeight": "600",
                     "letterSpacing": "0.06em", "textTransform": "uppercase",
                     "color": "#9ca3af", "marginBottom": "6px"}),
            html.Div(result_value, style={"fontSize": "32px", "fontWeight": "800",
                     "color": accent, "letterSpacing": "-0.5px"}),
        ], style={
            "backgroundColor": "#fafafa",
            "border": f"1px solid {accent}30",
            "borderLeft": f"4px solid {accent}",
            "borderRadius": "10px",
            "padding": "18px 24px",
        }),
        confidence_block,
    ])


if __name__ == "__main__":
    app.run(debug=True)
