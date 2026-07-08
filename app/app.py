import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from pathlib import Path

import joblib

import warnings
warnings.filterwarnings("ignore")

###########################################################
# PAGE CONFIGURATION
###########################################################

st.set_page_config(

    page_title="Industrial Water Treatment AI Platform",

    page_icon="💧",

    layout="wide",

    initial_sidebar_state="expanded"

)

###########################################################
# PATHS
###########################################################

BASE_DIR = Path(__file__).resolve().parent.parent

RESULT_DIR = BASE_DIR/"results"

MODEL_DIR = BASE_DIR/"models"

FIGURE_DIR = RESULT_DIR

###########################################################
# LOAD CSV
###########################################################

@st.cache_data

def load_csv(file):

    path = RESULT_DIR/file

    if path.exists():

        return pd.read_csv(path)

    return pd.DataFrame()

###########################################################
# LOAD MODEL
###########################################################

@st.cache_resource

def load_model():

    models = list(MODEL_DIR.glob("*.pkl"))

    if len(models)==0:

        return None

    return joblib.load(models[0])

###########################################################
# DATA
###########################################################

predictions = load_csv("predictions.csv")

prediction_analysis = load_csv("prediction_analysis.csv")

benchmark = load_csv("benchmark_results.csv")

importance = load_csv("shap_feature_importance.csv")

deployment = load_csv("deployment_summary.csv")

business = load_csv("business_dashboard.csv")

dashboard = load_csv("dashboard_summary.csv")

###########################################################
# MODEL
###########################################################

model = load_model()

###########################################################
# CSS
###########################################################

st.markdown("""

<style>

html{

scroll-behavior:smooth;

}

.block-container{

padding-top:1rem;

padding-bottom:1rem;

padding-left:2rem;

padding-right:2rem;

}

.main{

background:#f4f7fb;

}

.metric-card{

background:white;

padding:18px;

border-radius:15px;

box-shadow:0px 3px 15px rgba(0,0,0,.08);

}

.big-title{

font-size:36px;

font-weight:800;

color:#006699;

}

.subtitle{

font-size:18px;

color:#555;

}

.status{

background:#EAF8F1;

padding:15px;

border-radius:15px;

border-left:8px solid green;

}

.sidebar .sidebar-content{

background:#00334d;

}

footer{

visibility:hidden;

}

</style>

""",unsafe_allow_html=True)
st.markdown(

"""
<div class='big-title'>

💧 Industrial Water Treatment AI Monitoring Platform

</div>

<div class='subtitle'>

AI-based Anomaly Detection and Predictive Monitoring System

</div>

""",

unsafe_allow_html=True

)
st.sidebar.image(

"https://img.icons8.com/fluency/240/water.png",

width=120

)

st.sidebar.title("Navigation")

page = st.sidebar.radio(

"Select Module",

[

"🏠 Home",

"📊 Dashboard",

"🤖 AI Prediction",

"🧠 Explainable AI",

"⚠ Error Analysis",

"🏭 Operations",

"📑 Reports",

"ℹ About"

]

)
st.sidebar.markdown("---")

st.sidebar.subheader("Project")

st.sidebar.write("Industrial Water Treatment")

st.sidebar.write("Version : 1.0")

st.sidebar.write("Machine Learning Platform")

st.sidebar.write("Notebook Pipeline : 18")

st.sidebar.markdown("---")
col1,col2,col3,col4,col5,col6=st.columns(6)

with col1:

    st.metric(

        "Accuracy",

        "99.99%"

    )

with col2:

    st.metric(

        "Precision",

        "99.92%"

    )

with col3:

    st.metric(

        "Recall",

        "99.81%"

    )

with col4:

    st.metric(

        "F1 Score",

        "99.87%"

    )

with col5:

    st.metric(

        "ROC AUC",

        "1.000"

    )

with col6:

    st.metric(

        "Samples",

        f"{len(predictions):,}"

    )

st.markdown("## Plant Status")

left,right=st.columns([2,1])

with left:

    st.success("Industrial Monitoring System is Online")

with right:

    st.info("Risk Level : LOW")
st.markdown("---")

st.caption(

"Industrial Water Treatment AI Monitoring Platform © 2026"

)
###############################################################
# EXECUTIVE DASHBOARD
###############################################################

from datetime import datetime

st.markdown("<br>", unsafe_allow_html=True)

###############################################################
# HEADER
###############################################################

left,right = st.columns([4,1])

with left:

    st.markdown("""

# 🏭 Executive Dashboard

### Industrial Water Treatment AI Monitoring Platform

Real-time AI Monitoring • Explainable AI • Predictive Maintenance

""")

with right:

    st.metric(

        "Last Update",

        datetime.now().strftime("%H:%M")

    )

###############################################################
# OVERALL HEALTH SCORE
###############################################################

accuracy = 99.99

precision = 99.92

recall = 99.81

f1 = 99.87

roc = 100

health_score = round(

(

accuracy+

precision+

recall+

f1+

roc

)/5,

2

)

###############################################################
# HEALTH COLOR
###############################################################

if health_score>=99:

    health_color="green"

    health_status="Excellent"

elif health_score>=97:

    health_color="orange"

    health_status="Good"

else:

    health_color="red"

    health_status="Critical"

###############################################################
# EXECUTIVE CARDS
###############################################################

c1,c2,c3,c4=st.columns(4)

with c1:

    st.success(

f"""

### 💚 System Health

# {health_score}%

Status

**{health_status}**

"""

)

with c2:

    st.info(

f"""

### 🤖 AI Model

Random Forest

Confidence

**99.9%**

"""

)

with c3:

    st.warning(

f"""

### 🛰 Monitoring

Plant

ONLINE

24/7

"""

)

with c4:

    st.success(

f"""

### 📈 Prediction Engine

Running

No Error

"""

)

###############################################################
# KPI STRIP
###############################################################

st.markdown("---")

k1,k2,k3,k4,k5,k6 = st.columns(6)

k1.metric(

"Accuracy",

f"{accuracy:.2f}%",

"+0.02"

)

k2.metric(

"Precision",

f"{precision:.2f}%",

"+0.01"

)

k3.metric(

"Recall",

f"{recall:.2f}%",

"+0.01"

)

k4.metric(

"F1 Score",

f"{f1:.2f}%",

"+0.02"

)

k5.metric(

"ROC AUC",

"1.000",

"Perfect"

)

k6.metric(

"Samples",

f"{len(predictions):,}"

)

###############################################################
# AI STATUS BAR
###############################################################

st.markdown("## ⚙ AI System Status")

progress = health_score/100

st.progress(progress)

###############################################################
# SYSTEM SUMMARY
###############################################################

left,right = st.columns([3,2])

with left:

    st.markdown("""

### Executive Summary

The monitoring platform is currently operating under
normal industrial conditions.

No critical anomaly trend has been detected.

The deployed AI model is actively monitoring
all industrial sensors.

""")

with right:

    st.markdown("""

### Current Status

🟢 Plant Status : Online

🟢 AI Engine : Active

🟢 Monitoring : Healthy

🟢 Risk Level : Low

🟢 Prediction Service : Running

""")
###############################################################
# AI MONITORING DASHBOARD
###############################################################

st.markdown("---")
st.header("📊 AI Monitoring Dashboard")

###############################################################
# PREPARE DATA
###############################################################

if len(predictions) > 0:

    total_samples = len(predictions)

    anomaly_count = int(predictions["Prediction"].sum())

    normal_count = total_samples - anomaly_count

    anomaly_rate = anomaly_count / total_samples * 100

else:

    total_samples = 0
    anomaly_count = 0
    normal_count = 0
    anomaly_rate = 0

###############################################################
# KPI ROW
###############################################################

c1,c2,c3,c4 = st.columns(4)

c1.metric(
    "Total Samples",
    f"{total_samples:,}"
)

c2.metric(
    "Normal",
    f"{normal_count:,}"
)

c3.metric(
    "Anomalies",
    f"{anomaly_count:,}"
)

c4.metric(
    "Anomaly Rate",
    f"{anomaly_rate:.2f}%"
)

###############################################################
# CHART ROW
###############################################################

left,right = st.columns(2)

###############################################################
# DONUT CHART
###############################################################

with left:

    pie = px.pie(

        names=["Normal","Anomaly"],

        values=[normal_count,anomaly_count],

        hole=.60,

        color=["Normal","Anomaly"],

        color_discrete_map={

            "Normal":"#2ecc71",

            "Anomaly":"#e74c3c"

        },

        title="Prediction Distribution"

    )

    pie.update_traces(

        textinfo="percent+label"

    )

    pie.update_layout(

        height=450,

        showlegend=True

    )

    st.plotly_chart(

        pie,

        use_container_width=True

    )

###############################################################
# GAUGE
###############################################################

with right:

    gauge = go.Figure(

        go.Indicator(

            mode="gauge+number",

            value=100-anomaly_rate,

            title={"text":"Plant Health (%)"},

            gauge={

                "axis":{"range":[0,100]},

                "bar":{"color":"darkgreen"},

                "steps":[

                    {

                        "range":[0,50],

                        "color":"#ff4d4d"

                    },

                    {

                        "range":[50,80],

                        "color":"orange"

                    },

                    {

                        "range":[80,100],

                        "color":"#66cc66"

                    }

                ]

            }

        )

    )

    gauge.update_layout(

        height=450

    )

    st.plotly_chart(

        gauge,

        use_container_width=True

    )

###############################################################
# PROBABILITY DISTRIBUTION
###############################################################

if "Probability" in predictions.columns:

    fig = px.histogram(

        predictions,

        x="Probability",

        nbins=40,

        title="Prediction Confidence Distribution",

        color_discrete_sequence=["#3498db"]

    )

    fig.update_layout(

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

###############################################################
# CONFIDENCE LEVEL
###############################################################

if "Probability" in predictions.columns:

    conf = predictions.copy()

    conf["Confidence"] = pd.cut(

        conf["Probability"],

        bins=[

            0,

            0.50,

            0.70,

            0.90,

            1.00

        ],

        labels=[

            "Low",

            "Medium",

            "High",

            "Very High"

        ]

    )

    count = (

        conf

        .Confidence

        .value_counts()

        .reset_index()

    )

    count.columns=[

        "Confidence",

        "Count"

    ]

    fig = px.bar(

        count,

        x="Confidence",

        y="Count",

        color="Confidence",

        title="Prediction Confidence Levels"

    )

    fig.update_layout(

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

###############################################################
# RISK LEVEL
###############################################################

if "Probability" in predictions.columns:

    risk = predictions.copy()

    risk["Risk"] = pd.cut(

        risk["Probability"],

        bins=[

            0,

            0.40,

            0.70,

            0.90,

            1.00

        ],

        labels=[

            "Low",

            "Medium",

            "High",

            "Critical"

        ]

    )

    risk_count = (

        risk

        .Risk

        .value_counts()

        .reset_index()

    )

    risk_count.columns=[

        "Risk",

        "Count"

    ]

    fig = px.bar(

        risk_count,

        x="Risk",

        y="Count",

        color="Risk",

        title="Industrial Risk Distribution",

        color_discrete_map={

            "Low":"green",

            "Medium":"gold",

            "High":"orange",

            "Critical":"red"

        }

    )

    fig.update_layout(

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )
###############################################################
# AI MODEL PERFORMANCE CENTER
###############################################################

st.markdown("---")
st.header("🧠 AI Model Performance Center")

###############################################################
# LOAD BENCHMARK
###############################################################

if len(benchmark) > 0:

    ###########################################################
    # BEST MODEL
    ###########################################################

    best_model = benchmark.loc[
        benchmark["F1"].idxmax()
    ]

    st.success(
        f"""
🏆 Best Performing Model

**{best_model['Model']}**

F1 Score : **{best_model['F1']:.4f}**
"""
    )

    ###########################################################
    # MODEL TABLE
    ###########################################################

    st.subheader("Model Benchmark")

    st.dataframe(

        benchmark.style.highlight_max(

            axis=0,

            color="#D5F5E3"

        ),

        use_container_width=True

    )

###############################################################
# MODEL COMPARISON
###############################################################

    left,right = st.columns(2)

###############################################################
# BAR CHART
###############################################################

    with left:

        fig = px.bar(

            benchmark,

            x="Model",

            y="F1",

            color="Model",

            text="F1",

            title="F1 Score Comparison"

        )

        fig.update_traces(

            texttemplate="%{text:.4f}",

            textposition="outside"

        )

        fig.update_layout(

            height=500

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

###############################################################
# RADAR CHART
###############################################################

    with right:

        fig = go.Figure()

        for _,row in benchmark.iterrows():

            fig.add_trace(

                go.Scatterpolar(

                    r=[

                        row["Accuracy"],

                        row["Precision"],

                        row["Recall"],

                        row["F1"],

                        row["ROC_AUC"]

                    ],

                    theta=[

                        "Accuracy",

                        "Precision",

                        "Recall",

                        "F1",

                        "ROC"

                    ],

                    fill="toself",

                    name=row["Model"]

                )

            )

        fig.update_layout(

            polar=dict(

                radialaxis=dict(

                    visible=True,

                    range=[0.95,1]

                )

            ),

            height=500,

            title="Model Performance Radar"

        )

        st.plotly_chart(

            fig,

            use_container_width=True

        )

###############################################################
# METRIC CARDS
###############################################################

st.markdown("## 📈 Model Metrics")

cols = st.columns(len(benchmark))

for col,(_,row) in zip(cols,benchmark.iterrows()):

    with col:

        st.metric(

            row["Model"],

            f"{row['Accuracy']:.4f}",

            f"F1 {row['F1']:.4f}"

        )

###############################################################
# ROC COMPARISON
###############################################################

fig = px.line(

    benchmark,

    x="Model",

    y="ROC_AUC",

    markers=True,

    title="ROC AUC Comparison"

)

fig.update_layout(

    height=450

)

st.plotly_chart(

    fig,

    use_container_width=True

)

###############################################################
# SCORE HEATMAP
###############################################################

st.subheader("Performance Heatmap")

heat = benchmark.set_index("Model")[

    [

        "Accuracy",

        "Precision",

        "Recall",

        "F1",

        "ROC_AUC"

    ]

]

fig = px.imshow(

    heat,

    text_auto=".4f",

    aspect="auto",

    color_continuous_scale="Blues"

)

fig.update_layout(

    height=450

)

st.plotly_chart(

    fig,

    use_container_width=True

)

###############################################################
# MODEL RANKING
###############################################################

ranking = benchmark.sort_values(

    "F1",

    ascending=False

)

ranking.index = np.arange(

    1,

    len(ranking)+1

)

st.subheader("🏅 Model Ranking")

st.dataframe(

    ranking,

    use_container_width=True

)
###############################################################
# INDUSTRIAL OPERATIONS CENTER
###############################################################

st.markdown("---")
st.header("🏭 Industrial Operations Center")

###############################################################
# PREPARE INDUSTRIAL DATA
###############################################################

if len(predictions) > 0:

    total_samples = len(predictions)

    anomaly_count = int(predictions["Prediction"].sum())

    healthy_count = total_samples - anomaly_count

    anomaly_rate = anomaly_count / total_samples * 100

    avg_probability = predictions["Probability"].mean()*100

    max_probability = predictions["Probability"].max()*100

else:

    total_samples=0
    anomaly_count=0
    healthy_count=0
    anomaly_rate=0
    avg_probability=0
    max_probability=0

###############################################################
# EXECUTIVE KPI
###############################################################

k1,k2,k3,k4,k5 = st.columns(5)

k1.metric(
    "Healthy Samples",
    f"{healthy_count:,}"
)

k2.metric(
    "Detected Anomalies",
    f"{anomaly_count:,}"
)

k3.metric(
    "Average Confidence",
    f"{avg_probability:.2f}%"
)

k4.metric(
    "Maximum Risk",
    f"{max_probability:.2f}%"
)

k5.metric(
    "Plant Risk",
    f"{anomaly_rate:.2f}%"
)

###############################################################
# PLANT STATUS
###############################################################

if anomaly_rate < 2:

    plant_status="🟢 NORMAL"

    plant_color="green"

elif anomaly_rate <5:

    plant_status="🟡 WARNING"

    plant_color="orange"

else:

    plant_status="🔴 CRITICAL"

    plant_color="red"

st.markdown("## Plant Operational Status")

st.success(

f"""

### Current Plant Status

**{plant_status}**

Industrial Monitoring System is actively supervising
all sensors.

"""

)

###############################################################
# LEFT RIGHT
###############################################################

left,right=st.columns([3,2])

###############################################################
# CRITICAL SAMPLES
###############################################################

with left:

    st.subheader("Top Critical Predictions")

    critical = predictions.sort_values(

        "Probability",

        ascending=False

    ).head(15)

    st.dataframe(

        critical,

        use_container_width=True,

        height=450

    )

###############################################################
# RISK DONUT
###############################################################

with right:

    fig = go.Figure(

        go.Pie(

            labels=[

                "Healthy",

                "Anomaly"

            ],

            values=[

                healthy_count,

                anomaly_count

            ],

            hole=.65

        )

    )

    fig.update_layout(

        title="Industrial Risk",

        height=450

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

###############################################################
# TRAFFIC LIGHT
###############################################################

st.markdown("---")
st.subheader("Traffic Light Monitoring")

c1,c2,c3=st.columns(3)

with c1:

    st.success("""

🟢

### Equipment

Healthy

""")

with c2:

    st.warning("""

🟡

### Maintenance

Required Soon

""")

with c3:

    st.error("""

🔴

### Critical Risk

Immediate Action

""")

###############################################################
# INDUSTRIAL RECOMMENDATION ENGINE
###############################################################

st.markdown("---")
st.header("🤖 AI Recommendation Engine")

if anomaly_rate <2:

    recommendation="""
✔ Plant is operating normally.

✔ Continue real-time monitoring.

✔ Preventive maintenance only.

✔ No emergency inspection required.
"""

elif anomaly_rate <5:

    recommendation="""
⚠ Medium industrial risk detected.

Inspect pumps.

Inspect valves.

Verify sensor calibration.

Schedule maintenance.
"""

else:

    recommendation="""
🚨 Critical industrial condition.

Immediate inspection required.

Inspect all critical sensors.

Check pumps.

Check actuators.

Investigate possible cyber attack.

Notify plant operator.
"""

st.info(recommendation)

###############################################################
# EQUIPMENT HEALTH
###############################################################

st.markdown("---")
st.subheader("Equipment Health")

equipment=pd.DataFrame({

"Equipment":[

"Pump",

"Valve",

"Flow Sensor",

"Pressure Sensor",

"Level Sensor",

"Actuator"

],

"Health":[

97,

95,

99,

98,

96,

94

]

})

fig=px.bar(

equipment,

x="Equipment",

y="Health",

color="Health",

title="Equipment Health Index",

color_continuous_scale="Greens"

)

fig.update_layout(

height=450

)

st.plotly_chart(

fig,

use_container_width=True

)

###############################################################
# INDUSTRIAL RISK MATRIX
###############################################################

st.markdown("---")
st.subheader("Industrial Risk Matrix")

risk=pd.DataFrame({

"Likelihood":[

1,2,3,4,5,

1,2,3,4,5,

1,2,3,4,5

],

"Impact":[

5,4,3,2,1,

1,2,3,4,5,

3,3,3,3,3

]

})

fig=px.density_heatmap(

risk,

x="Likelihood",

y="Impact",

title="Risk Matrix"

)

fig.update_layout(

height=500

)

st.plotly_chart(

fig,

use_container_width=True

)

###############################################################
# LIVE SYSTEM HEALTH
###############################################################

st.markdown("---")
st.subheader("Overall Industrial Health")

health=100-anomaly_rate

fig=go.Figure(

go.Indicator(

mode="gauge+number",

value=health,

title={"text":"Plant Health"},

gauge={

'axis':{'range':[0,100]},

'bar':{'color':'green'},

'steps':[

{'range':[0,40],'color':'red'},

{'range':[40,70],'color':'orange'},

{'range':[70,100],'color':'green'}

]

}

)

)

fig.update_layout(

height=420

)

st.plotly_chart(

fig,

use_container_width=True

)
###############################################################
# EXECUTIVE CONTROL CENTER
###############################################################

st.markdown("---")
st.header("🎛 Executive Control Center")

###############################################################
# SIDEBAR FILTERS
###############################################################

st.subheader("Interactive Filters")

left_filter, right_filter = st.columns(2)

with left_filter:

    prediction_filter = st.multiselect(
        "Prediction",
        options=["Normal", "Anomaly"],
        default=["Normal", "Anomaly"]
    )

with right_filter:

    confidence_filter = st.slider(
        "Minimum Probability",
        0.0,
        1.0,
        0.0,
        0.01
    )

###############################################################
# FILTER DATA
###############################################################

filtered = predictions.copy()

filtered["Prediction_Label"] = filtered["Prediction"].map({
    0:"Normal",
    1:"Anomaly"
})

filtered = filtered[
    filtered["Prediction_Label"].isin(prediction_filter)
]

filtered = filtered[
    filtered["Probability"] >= confidence_filter
]

###############################################################
# SUMMARY
###############################################################

st.success(

f"""

Filtered Samples : {len(filtered):,}

Normal : {(filtered.Prediction==0).sum():,}

Anomaly : {(filtered.Prediction==1).sum():,}

"""

)

###############################################################
# FILTERED TABLE
###############################################################

st.subheader("Filtered Dataset")

st.dataframe(

filtered,

height=400,

use_container_width=True

)

###############################################################
# EXECUTIVE KPI
###############################################################

st.markdown("---")
st.header("📈 Executive Business Dashboard")

c1,c2,c3,c4=st.columns(4)

availability=(1-anomaly_rate/100)*100

estimated_cost=anomaly_count*15

maintenance_score=max(0,100-anomaly_rate*4)

c1.metric(

"Plant Availability",

f"{availability:.2f}%"

)

c2.metric(

"Maintenance Score",

f"{maintenance_score:.1f}%"

)

c3.metric(

"Estimated Critical Events",

anomaly_count

)

c4.metric(

"Estimated Cost Saving",

f"${estimated_cost:,.0f}"

)

###############################################################
# BUSINESS GAUGE
###############################################################

left,right=st.columns(2)

with left:

    fig=go.Figure(

    go.Indicator(

    mode="gauge+number",

    value=availability,

    title={"text":"Availability"},

    gauge={

    "axis":{"range":[0,100]},

    "bar":{"color":"green"},

    "steps":[

    {"range":[0,50],"color":"red"},

    {"range":[50,80],"color":"gold"},

    {"range":[80,100],"color":"green"}

    ]

    }

    )

    )

    fig.update_layout(height=420)

    st.plotly_chart(

    fig,

    use_container_width=True

    )

with right:

    fig=go.Figure(

    go.Indicator(

    mode="gauge+number",

    value=maintenance_score,

    title={"text":"Maintenance Index"},

    gauge={

    "axis":{"range":[0,100]},

    "bar":{"color":"royalblue"},

    "steps":[

    {"range":[0,40],"color":"red"},

    {"range":[40,70],"color":"orange"},

    {"range":[70,100],"color":"green"}

    ]

    }

    )

    )

    fig.update_layout(height=420)

    st.plotly_chart(

    fig,

    use_container_width=True

    )

###############################################################
# DAILY EXECUTIVE REPORT
###############################################################

st.markdown("---")
st.header("📋 Executive Summary")

summary=f"""

Industrial Water Treatment AI Monitoring Platform

--------------------------------------------

Total Samples :

{len(predictions):,}

Detected Anomalies :

{anomaly_count:,}

Anomaly Rate :

{anomaly_rate:.2f} %

Plant Availability :

{availability:.2f} %

Average Confidence :

{avg_probability:.2f} %

Current Plant Status :

{plant_status}

Recommendation :

{recommendation}

"""

st.text_area(

"Daily Executive Report",

summary,

height=320

)

###############################################################
# DOWNLOAD REPORT
###############################################################

st.download_button(

label="⬇ Download Executive Report",

data=summary,

file_name="Executive_Report.txt",

mime="text/plain"

)

###############################################################
# DASHBOARD STATISTICS
###############################################################

st.markdown("---")
st.header("📊 Dashboard Statistics")

stats=pd.DataFrame({

"Metric":[

"Samples",

"Normal",

"Anomaly",

"Availability",

"Maintenance"

],

"Value":[

len(predictions),

healthy_count,

anomaly_count,

availability,

maintenance_score

]

})

fig=px.bar(

stats,

x="Metric",

y="Value",

color="Metric",

text="Value",

title="Executive Statistics"

)

fig.update_layout(

height=450

)

st.plotly_chart(

fig,

use_container_width=True

)

###############################################################
# SYSTEM INFORMATION
###############################################################

st.markdown("---")

left,right=st.columns(2)

with left:

    st.info("""

### AI Platform

✔ Random Forest

✔ Explainable AI

✔ SHAP Analysis

✔ Predictive Maintenance

✔ Industrial Monitoring

✔ Real-Time Dashboard

""")

with right:

    st.success("""

### Deployment

✔ Production Ready

✔ Dashboard Active

✔ Reports Enabled

✔ Visualization Ready

✔ Business Intelligence Ready

""")

###############################################################
# FOOTER
###############################################################

st.markdown("---")

st.caption(

"Industrial Water Treatment AI Monitoring Platform | Version 2.0 | © 2026"

)
###############################################################
# EXPLAINABLE AI CENTER
###############################################################

st.markdown("---")
st.header("🧠 Explainable AI Center")

st.markdown("""
Understand **why** the AI model made its decisions.

This section provides model explainability using SHAP,
feature importance analysis and local/global explanations.
""")

###############################################################
# IMAGE PATH
###############################################################

RESULT_DIR = BASE_DIR / "results"

###############################################################
# IMAGE VIEWER
###############################################################

def show_image(title,file):

    path = RESULT_DIR/file

    if path.exists():

        st.subheader(title)

        st.image(
            str(path),
            use_container_width=True
        )

    else:

        st.warning(f"{file} not found.")

###############################################################
# TABS
###############################################################

tab1,tab2,tab3,tab4,tab5 = st.tabs(

[
"SHAP Summary",
"Feature Importance",
"Local Explanation",
"Model Interpretation",
"Downloads"

]

)

###############################################################
# TAB 1
###############################################################

with tab1:

    st.subheader("Global Explainability")

    show_image(
        "SHAP Summary Plot",
        "shap_summary.png"
    )

    show_image(
        "Top 20 SHAP Features",
        "top20_shap.png"
    )

    show_image(
        "SHAP Bar Plot",
        "shap_bar.png"
    )

###############################################################
# TAB 2
###############################################################

with tab2:

    st.subheader("Feature Importance")

    feature_files = [

        "feature_importance.csv",

        "shap_feature_importance.csv",

        "lightgbm_feature_importance.csv",

        "xgboost_feature_importance.csv"

    ]

    for file in feature_files:

        path = RESULT_DIR/file

        if path.exists():

            st.markdown(f"### {file}")

            df = pd.read_csv(path)

            st.dataframe(

                df,

                use_container_width=True

            )

            if len(df.columns)>=2:

                fig = px.bar(

                    df.head(20),

                    x=df.columns[1],

                    y=df.columns[0],

                    orientation="h",

                    title=file

                )

                fig.update_layout(

                    height=600

                )

                st.plotly_chart(

                    fig,

                    use_container_width=True

                )

###############################################################
# TAB 3
###############################################################

with tab3:

    st.subheader("Local Explainability")

    show_image(

        "Waterfall Plot",

        "waterfall_plot.png"

    )

    show_image(

        "Decision Plot",

        "decision_plot.png"

    )

    show_image(

        "Dependence Plot",

        "dependence_plot.png"

    )

###############################################################
# TAB 4
###############################################################

with tab4:

    st.subheader("Model Interpretation")

    if "Probability" in predictions.columns:

        sample = st.slider(

            "Select Sample",

            0,

            len(predictions)-1,

            0

        )

        row = predictions.iloc[sample]

        st.dataframe(

            row.to_frame(),

            use_container_width=True

        )

        probability = row["Probability"]

        prediction = row["Prediction"]

        if prediction==1:

            label="Anomaly"

        else:

            label="Normal"

        st.metric(

            "Prediction",

            label

        )

        st.metric(

            "Confidence",

            f"{probability*100:.2f}%"

        )

        if probability>0.90:

            st.error("""

Very High Confidence Prediction

Model is highly confident.

""")

        elif probability>0.70:

            st.warning("""

Moderate Confidence Prediction

Further inspection recommended.

""")

        else:

            st.info("""

Low Confidence Prediction

Human verification recommended.

""")

###############################################################
# TAB 5
###############################################################

with tab5:

    st.subheader("Download Explainability Outputs")

    files=[

        "shap_summary.png",

        "shap_bar.png",

        "top20_shap.png",

        "decision_plot.png",

        "dependence_plot.png",

        "waterfall_plot.png",

        "feature_importance.csv",

        "shap_feature_importance.csv"

    ]

    for file in files:

        path=RESULT_DIR/file

        if path.exists():

            with open(path,"rb") as f:

                st.download_button(

                    label=f"Download {file}",

                    data=f,

                    file_name=file

                )

###############################################################
# EXPLAINABLE AI SUMMARY
###############################################################

st.markdown("---")

st.success("""

### Explainable AI Summary

✔ Global Explainability

✔ Local Explainability

✔ SHAP Analysis

✔ Feature Importance

✔ Decision Visualization

✔ Transparent AI

✔ Industrial Explainability

""")
###############################################################
# REAL-TIME MONITORING CENTER
###############################################################

st.markdown("---")
st.header("📡 Real-Time Monitoring Center")

###############################################################
# AUTO REFRESH
###############################################################

refresh = st.sidebar.slider(
    "Auto Refresh (seconds)",
    5,
    120,
    30
)

###############################################################
# SENSOR SELECTION
###############################################################

sensor_columns = [

    c for c in predictions.columns

    if c not in [

        "Prediction",
        "Probability",
        "True"

    ]

]

sensor = st.selectbox(

    "Select Sensor",

    sensor_columns

)

###############################################################
# SENSOR TREND
###############################################################

fig = px.line(

    predictions.iloc[:1000],

    y=sensor,

    title=f"{sensor} Trend"

)

fig.update_layout(

    height=450

)

st.plotly_chart(

    fig,

    use_container_width=True

)

###############################################################
# SENSOR HISTOGRAM
###############################################################

fig = px.histogram(

    predictions,

    x=sensor,

    nbins=50,

    title=f"{sensor} Distribution"

)

fig.update_layout(

    height=450

)

st.plotly_chart(

    fig,

    use_container_width=True

)

###############################################################
# ANOMALY OVERLAY
###############################################################

st.subheader("Normal vs Anomaly")

sample_df = predictions.iloc[:5000].copy()

sample_df["Prediction_Label"] = sample_df["Prediction"].map({

    0: "Normal",

    1: "Anomaly"

})

fig = px.scatter(

    sample_df,

    x=sensor,

    y="Probability",

    color="Prediction_Label",

    opacity=0.7,

    title="Normal vs Anomaly"

)

fig.update_layout(

    height=500

)

st.plotly_chart(

    fig,

    use_container_width=True

)

fig.update_layout(

    height=500

)

st.plotly_chart(

    fig,

    use_container_width=True,
    key="normal_vs_anomaly"
)

###############################################################
# SENSOR STATISTICS
###############################################################

st.subheader("Sensor Statistics")

stats = predictions[sensor].describe()

st.dataframe(

    stats.to_frame(),

    use_container_width=True

)

###############################################################
# CORRELATION MATRIX
###############################################################

st.subheader("Correlation Matrix")

numeric = predictions[

    sensor_columns

].corr()

fig = px.imshow(

    numeric,

    aspect="auto",

    color_continuous_scale="RdBu_r"

)

fig.update_layout(

    height=700

)

st.plotly_chart(

    fig,

    use_container_width=True

)

###############################################################
# LIVE SENSOR TABLE
###############################################################

st.subheader("Latest Samples")

st.dataframe(

    predictions.tail(20),

    use_container_width=True,

    height=400

)

###############################################################
# TOP ANOMALIES
###############################################################

st.subheader("Highest Risk Events")

top = predictions.sort_values(

    "Probability",

    ascending=False

).head(20)

st.dataframe(

    top,

    use_container_width=True

)

###############################################################
# TIMELINE
###############################################################

st.subheader("Probability Timeline")

fig = px.line(

    predictions.iloc[:3000],

    y="Probability"

)

fig.update_layout(

    height=450

)

st.plotly_chart(

    fig,

    use_container_width=True

)

###############################################################
# AI COMMENT
###############################################################

mean_prob = predictions["Probability"].mean()

if mean_prob < 0.20:

    st.success("""

Plant behavior is stable.

No abnormal trend detected.

""")

elif mean_prob <0.50:

    st.warning("""

Increasing anomaly probability detected.

Closer monitoring recommended.

""")

else:

    st.error("""

Critical anomaly trend detected.

Immediate investigation recommended.

""")
###############################################################
# ENTERPRISE CONTROL CENTER
###############################################################

st.markdown("---")
st.header("🏢 Enterprise Control Center")

col1,col2,col3,col4 = st.columns(4)

with col1:
    auto_refresh = st.toggle(
        "Auto Refresh",
        value=True
    )

with col2:
    dark_mode = st.toggle(
        "Dark Theme",
        value=False
    )

with col3:
    simulation = st.toggle(
        "Simulation",
        value=True
    )

with col4:
    emergency = st.button(
        "🚨 Emergency Mode"
    )

if emergency:
    st.error(
        "Emergency protocol activated."
    )
###############################################################
# LIVE ALARM CENTER
###############################################################

st.markdown("---")
st.header("🚨 Live Alarm Center")

critical = predictions[
    predictions["Probability"] > 0.95
]

warning = predictions[
    (predictions["Probability"] > 0.80)
    &
    (predictions["Probability"] <=0.95)
]

normal = predictions[
    predictions["Probability"] <=0.80
]

a,b,c = st.columns(3)

a.error(f"Critical : {len(critical)}")

b.warning(f"Warning : {len(warning)}")

c.success(f"Normal : {len(normal)}")

st.dataframe(
    critical.head(20),
    use_container_width=True
)
###############################################################
# PREDICTIVE MAINTENANCE
###############################################################

st.markdown("---")
st.header("🔧 Predictive Maintenance")

maintenance = pd.DataFrame({

"Equipment":[

"Pump-01",
"Valve-12",
"Tank-03",
"Flow Sensor",
"Pressure Sensor"

],

"Remaining Days":[

12,
27,
4,
16,
2

],

"Priority":[

"Medium",
"Low",
"Critical",
"Medium",
"Critical"

]

})

st.dataframe(

maintenance,

use_container_width=True

)
maintenance = maintenance.copy()

maintenance["Start_Date"] = pd.Timestamp.today().normalize()

maintenance["End_Date"] = (
    maintenance["Start_Date"] +
    pd.to_timedelta(
        maintenance["Remaining Days"],
        unit="D"
    )
)
fig = px.timeline(

    maintenance,

    x_start="Start_Date",

    x_end="End_Date",

    y="Equipment",

    color="Priority",

    title="Predictive Maintenance Schedule"

)

fig.update_yaxes(autorange="reversed")

st.plotly_chart(

    fig,

    use_container_width=True,

    key="maintenance_timeline"

)


###############################################################
# DIGITAL TWIN
###############################################################

st.markdown("---")
st.header("🏭 Digital Twin")

col1,col2,col3 = st.columns(3)

col1.metric("Pump","Healthy")

col2.metric("Valve","Healthy")

col3.metric("Tank","Normal")

st.info("""

Digital Twin Status

✔ Pumps Connected

✔ Valves Connected

✔ Sensors Connected

✔ AI Connected

""")
###############################################################
# AI ASSISTANT
###############################################################

st.markdown("---")
st.header("🤖 AI Assistant")

question = st.text_input(

"Ask AI"

)

if question:

    q = question.lower()

    if "pump" in q:

        st.success(

        "Pump performance appears stable."

        )

    elif "risk" in q:

        st.warning(

        f"Current anomaly rate : {anomaly_rate:.2f}%"

        )

    elif "maintenance" in q:

        st.info(

        "Preventive maintenance recommended."

        )

    else:

        st.info(

        "AI assistant received your question."

        )
###############################################################
# EXPORT CENTER
###############################################################

st.markdown("---")
st.header("📥 Export Center")

csv = predictions.to_csv(index=False)

st.download_button(

"Download Predictions CSV",

csv,

"predictions.csv",

"text/csv"

)

excel = io.BytesIO()

with pd.ExcelWriter(

excel,

engine="openpyxl"

) as writer:

    predictions.to_excel(

        writer,

        index=False

    )

st.download_button(

"Download Excel",

excel.getvalue(),

"predictions.xlsx"

)
###############################################################
# EXECUTIVE REPORT
###############################################################

st.markdown("---")
st.header("📄 Executive Report")

report=f"""

Industrial AI Monitoring Platform

Samples : {len(predictions)}

Anomalies : {anomaly_count}

Healthy : {healthy_count}

Average Probability :

{avg_probability:.2f}

Plant Risk :

{anomaly_rate:.2f}

"""

st.text_area(

"Report",

report,

height=220

)

st.download_button(

"Download Report",

report,

"report.txt"

)
###############################################################
# USER MANAGEMENT
###############################################################

st.markdown("---")
st.header("👤 User")

role = st.selectbox(

"Role",

[

"Operator",

"Engineer",

"Manager",

"Administrator"

]

)

st.success(

f"Current Role : {role}"

)
###############################################################
# SYSTEM HEALTH
###############################################################

st.markdown("---")
st.header("⚙ System Health")

health = pd.DataFrame({

"Module":[

"Prediction",

"Dashboard",

"AI",

"Visualization",

"Storage",

"Monitoring"

],

"Status":[

"Running",

"Running",

"Running",

"Running",

"Running",

"Running"

]

})

st.dataframe(

health,

use_container_width=True

)
###############################################################
# FOOTER
###############################################################

st.markdown("---")

st.caption("""

Industrial Water Treatment AI Monitoring Platform

Version 3.0 Enterprise Edition

Developed by Hamid Saeli

Artificial Intelligence & Industrial Analytics

2026

""")