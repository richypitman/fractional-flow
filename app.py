import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from fractional_flow import (
    relPermModel,
    fractionalFlowCurve,
    fractionalFlowGradient,
    shockFront,
)

import plotly.express as px
import plotly.graph_objects as go


def plot_rel_perm(sw_arr: list, krw_arr: list, krow_arr: list) -> plt.Figure:
    fig = go.Figure()
    fig.add_trace(go.Line(x=sw_arr, y=krw_arr, line=dict(color="blue")))
    fig.add_trace(go.Line(x=sw_arr, y=krow_arr, line=dict(color="green")))

    fig.update_layout(
        height=350,
        width=350,
        xaxis=dict(title_text="Sw", range=[0, 1]),
        yaxis=dict(title_text="krw, krow", range=[0, 1]),
        margin=dict(t=0, r=0, b=0, l=0),
        showlegend=False,
        template="simple_white",
    )

    return fig


def plot_fw(sw_arr: list, fw_arr: list) -> plt.Figure:
    fig = go.Figure()
    fig.add_trace(go.Line(x=sw_arr, y=fw_arr, line=dict(color="blue")))

    fig.update_layout(
        height=350,
        width=350,
        xaxis=dict(title_text="Sw", range=[0, 1]),
        yaxis=dict(title_text="fw", range=[0, 1]),
        margin=dict(t=0, r=0, b=0, l=0),
        template="simple_white",
    )

    return fig


def plot_dfw_dsw(sw_arr: list, dfw_dsw_arr: list) -> plt.Figure:
    fig = go.Figure()
    fig.add_trace(go.Line(x=sw_arr, y=dfw_dsw_arr, line=dict(color="blue")))

    fig.update_layout(
        height=350,
        width=350,
        xaxis=dict(title_text="Sw", range=[0, 1]),
        yaxis=dict(title_text="dfw/dSw", range=[0, 1.1 * max(dfw_dsw_arr)]),
        margin=dict(t=0, r=0, b=0, l=0),
        template="simple_white",
    )

    return fig


def plot_shock(x_arr: list, sw_shock_arr: list) -> plt.Figure:
    fig = go.Figure()
    fig.add_trace(go.Line(x=x_arr, y=sw_shock_arr, line=dict(color="blue")))

    fig.update_layout(
        height=350,
        width=350,
        xaxis=dict(title_text="x", range=[0, 1]),
        yaxis=dict(title_text="Sw", range=[0, 1]),
        margin=dict(t=0, r=0, b=0, l=0),
        template="simple_white",
    )

    return fig


# App input
swi = st.sidebar.number_input(
    label="Initial water saturation, Swi",
    min_value=0.01,
    max_value=1.0,
    value=0.15,
    step=0.01,
)
sorw = st.sidebar.number_input(
    label="Residual oil saturation, Sorw",
    min_value=0.01,
    max_value=1.0 - swi,
    value=0.15,
    step=0.01,
)
krwe = st.sidebar.number_input(
    label="Endpoint water relative permeability, krw(Sorw)",
    min_value=0.05,
    max_value=1.0,
    value=0.5,
    step=0.01,
)
kroe = st.sidebar.number_input(
    label="Endpoint oil relative permeability, krow(Swi)",
    min_value=0.05,
    max_value=1.0,
    value=1.0,
    step=0.01,
)
nw = st.sidebar.number_input(
    label="Corey parameter water, nw",
    min_value=0.05,
    max_value=5.0,
    value=2.0,
    step=0.05,
)
no = st.sidebar.number_input(
    label="Corey parameter oil, no", min_value=0.05, max_value=5.0, value=2.0, step=0.05
)
muo = st.sidebar.number_input(
    label="Water viscosity, muw", min_value=0.05, max_value=5.0, value=1.0, step=0.05
)
muw = st.sidebar.number_input(
    label="Oil viscosity, muo", min_value=0.05, max_value=1000.0, value=1.0, step=0.05
)
num_points = st.sidebar.number_input(
    label="Number of points", min_value=10, max_value=1000, value=100, step=1
)

# Create curves
kr = relPermModel(swi, 1 - sorw, krwe, kroe, nw, no, num_points)
kr.create_relperm_model()
fw = fractionalFlowCurve(kr.sw_arr, kr.krw_arr, kr.kro_arr, muw, muo)
fw.create_fw()
fw.perform_welge_construction()
dfw_dsw = fractionalFlowGradient(fw.fw_arr, kr.sw_arr)
dfw_dsw.create_dfw_dsw()
shock_front = shockFront(kr.sw_arr, dfw_dsw.dfw_dsw_arr, fw.swi, fw.swbt, fw.swf)
shock_front.create_shock_front()

# App display
st.title("Buckley-Leverett solution")
# st.write(fw.swbt)

left_column, right_column = st.columns([1, 1])
with left_column:
    # st.write(shock_front.x_arr)
    st.write(plot_rel_perm(kr.sw_arr, kr.krw_arr, kr.kro_arr))
    st.write(plot_dfw_dsw(kr.sw_arr, dfw_dsw.dfw_dsw_arr))
with right_column:
    # st.write(shock_front.sw_shock_arr)
    st.write(plot_fw(kr.sw_arr, fw.fw_arr))
    st.write(plot_shock(shock_front.x_arr, shock_front.sw_shock_arr))
