import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
import shap

# Load data and model
df = pd.read_csv("./data/framingham.csv")
model = joblib.load("model.pkl")

event_col = "CVD"
time_col = "TIMECVD"

features = [
    "SEX", "AGE", "SYSBP", "DIABP", "BPMEDS",
    "CURSMOKE", "CIGPDAY", "EDUC", "TOTCHOL",
    "BMI", "GLUCOSE", "DIABETES", "HEARTRTE"
]

data = df[features + [event_col, time_col]].dropna()
data = data[data[time_col] > 0]

X = data[features]
y = Surv.from_dataframe(event_col, time_col, data)

risk_scores = model.predict(X)
c_index = concordance_index_censored(
    data[event_col].astype(bool),
    data[time_col],
    risk_scores
)[0]

event_rate = data[event_col].mean()

median_followup_years = (
    data[data[event_col] == 1][time_col].median()
    / 365.25
)
n_patients = len(data)


# App setup
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server


def metric_card(title, value):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="text-muted"),
            html.H3(value)
        ]),
        className="shadow-sm"
    )


# Layout
app.layout = dbc.Container([

    html.H1("Framingham Survival Analysis Dashboard", className="mt-4"),

    dbc.Row([
        dbc.Col(metric_card("Patients", f"{n_patients:,}"), md=3),
        dbc.Col(metric_card("Event Rate", f"{event_rate:.1%}"), md=3),
        dbc.Col(metric_card("Median Follow-up", f"{median_followup_years:.1f} years"), md=3),
        dbc.Col(metric_card("C-index", f"{c_index:.3f}"), md=3),
    ], className="mb-4"),

    dbc.Row([

        dbc.Col([
            html.H4("Patient Risk Predictor"),
            dbc.Label("Sex"),
            dcc.Dropdown(
                id="sex",
                options=[
                    {"label": "Female", "value": 1},
                    {"label": "Male", "value": 2},
                ],
                value=1
            ),

            dbc.Label("Age"),
            dbc.Input(id="age", type="number", value=50),

            dbc.Label("Heart Rate"),
            dbc.Input(id="heartrte", type="number", value=75),

            dbc.Label("Systolic BP"),
            dbc.Input(id="sysbp", type="number", value=130),

            dbc.Label("Diastolic BP"),
            dbc.Input(id="diabp", type="number", value=80),

            dbc.Label("Total Cholesterol"),
            dbc.Input(id="totchol", type="number", value=220),

            dbc.Label("BMI"),
            dbc.Input(id="bmi", type="number", value=26),

            dbc.Label("Glucose"),
            dbc.Input(id="glucose", type="number", value=85),

            dbc.Label("Diabetes"),
            dcc.Dropdown(
                id="diabetes",
                options=[
                    {"label": "No", "value": 0},
                    {"label": "Yes", "value": 1},
                ],
                value=0
            ),

            dbc.Label("On BP Medication"),
            dcc.Dropdown(
                id="bpmeds",
                options=[
                    {"label": "No", "value": 0},
                    {"label": "Yes", "value": 1},
                ],
                value=0
            ),

            dbc.Label("Current Smoker"),
            dcc.Dropdown(
                id="cursmoke",
                options=[
                    {"label": "No", "value": 0},
                    {"label": "Yes", "value": 1},
                ],
                value=0
            ),

            dbc.Label("Cigarettes Per Day"),
            dbc.Input(id="cigpday", type="number", value=0),

            dbc.Label("Education"),
            dcc.Dropdown(
                id="educ",
                options=[
                    {"label": "Less than Secondary School", "value": 1},
                    {"label": "Secondary School", "value": 2},
                    {"label": "Undergraduate", "value": 3},
                    {"label": "Graduate", "value": 4},
                ],
                value=0
            ),

            dbc.Button("Predict Risk", id="predict-btn", color="primary", className="mt-3")
        ], md=4),

        dbc.Col([
            html.H4("Predicted Survival Curve"),
            dcc.Graph(id="survival-curve"),
            html.Div(id="prediction-output", className="mt-4")
        ], md=8)

    ]),

    html.Hr(),

    dbc.Row([
        dbc.Col([
            html.H4("Observed Time-to-Event Distribution"),
            dcc.Graph(id="time-distribution")
        ], md=6),

        dbc.Col([
            html.H4("Risk Score Distribution"),
            dcc.Graph(id="risk-distribution")
        ], md=6)
    ])

], fluid=True)


# Static plots

@app.callback(
    Output("time-distribution", "figure"),
    Input("predict-btn", "n_clicks")
)
def update_time_distribution(_):
    observed_times = (
        data[data[event_col] == 1][time_col]
        / 365.25
    )

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=observed_times,
        nbinsx=30
    ))

    fig.update_layout(
        xaxis_title="Observed Time to CVD Event (Years)",
        yaxis_title="Count",
        template="plotly_white"
    )
    return fig


@app.callback(
    Output("risk-distribution", "figure"),
    Input("predict-btn", "n_clicks")
)
def update_risk_distribution(_):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=risk_scores,
        nbinsx=30
    ))
    fig.update_layout(
        xaxis_title="Predicted Risk Score",
        yaxis_title="Count",
        template="plotly_white"
    )
    return fig


# Prediction callback

@app.callback(
    Output("prediction-output", "children"),
    Output("survival-curve", "figure"),
    Input("predict-btn", "n_clicks"),
    State("sex", "value"),
    State("age", "value"),
    State("sysbp", "value"),
    State("diabp", "value"),
    State("bpmeds", "value"),
    State("cursmoke", "value"),
    State("cigpday", "value"),
    State("educ", "value"),
    State("totchol", "value"),
    State("bmi", "value"),
    State("glucose", "value"),
    State("diabetes", "value"),
    State("heartrte", "value")
)
def predict_survival(
    n_clicks, sex, age, sysbp, diabp, bpmeds,
    cursmoke, cigpday, educ, totchol, bmi,
    glucose, diabetes, heartrte
):

    patient = pd.DataFrame([{
        "SEX": sex,
        "AGE": age,
        "SYSBP": sysbp,
        "DIABP": diabp,
        "BPMEDS": bpmeds,
        "CURSMOKE": cursmoke,
        "CIGPDAY": cigpday,
        "EDUC": educ,
        "TOTCHOL": totchol,
        "BMI": bmi,
        "GLUCOSE": glucose,
        "DIABETES": diabetes,
        "HEARTRTE": heartrte
    }])

    surv_fn = model.predict_survival_function(patient)[0]

    times_years = surv_fn.x / 365.25
    survival_probs = surv_fn.y
    risk_probs = 1 - survival_probs

    risk_5yr = 1 - surv_fn(365.25 * 5)
    risk_10yr = 1 - surv_fn(365.25 * 10)
    risk_20yr = 1 - surv_fn(365.25 * 20)

    prediction_text = dbc.Card(
        dbc.CardBody([
            html.H5("Predicted CVD Risk"),
            html.P(f"5-year risk: {risk_5yr:.1%}"),
            html.P(f"10-year risk: {risk_10yr:.1%}"),
            html.P(f"20-year risk: {risk_20yr:.1%}")
        ]),
        className="shadow-sm"
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=times_years,
        y=survival_probs,
        mode="lines",
        name="Survival Probability"
    ))

    fig.update_layout(
        xaxis_title="Years",
        yaxis_title="Probability of Remaining CVD-Free",
        template="plotly_white",
        yaxis=dict(range=[0, 1])
    )

    return prediction_text, fig


# Run app

if __name__ == "__main__":
    app.run(debug=True)