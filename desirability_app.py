#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from matplotlib import cm
from matplotlib.colors import Normalize
from statsmodels.tools.tools import add_constant
import matplotlib as mpl
from matplotlib import font_manager as fm
from pathlib import Path

# CARLITO FONT
font_dir = Path(__file__).parent / "assets" / "fonts"
if font_dir.exists():
    for fp in font_dir.glob("*.ttf"):
        fm.fontManager.addfont(str(fp))
    mpl.rcParams["font.family"] = ["Carlito", "DejaVu Sans"]
else:
    mpl.rcParams["font.family"] = ["DejaVu Sans"]

# PAGE TITLE
st.set_page_config(page_title="Au@SiO₂ nanocarriers synthesis optimization", layout="centered")
st.title("🎯 Au@SiO₂ nanocarriers synthesis optimization")

# DATA LOADING
df = pd.read_excel("BBD_plan_Au@SiO2.xlsx")
grid = pd.read_csv("grid.csv")
desirability = np.load("desirability.npy")
model_zeta = joblib.load("model_zeta.pkl")
model_shell = joblib.load("model_shell.pkl")
scaler = joblib.load("scaler.pkl")

predictors = ['Temperature (°C)', 'TEOS (µL)', 'Time (h)']

# SLIDERS
st.sidebar.header("🔧 SELECT SYNTHESIS PARAMETERS")
temp_val = st.sidebar.slider("Temperature (°C)", float(df[predictors[0]].min()), float(df[predictors[0]].max()), float(df[predictors[0]].mean()), step = 0.1, format="%.1f")
teos_val = st.sidebar.slider("TEOS (µL)", float(df[predictors[1]].min()), float(df[predictors[1]].max()), float(df[predictors[1]].mean()), step = 0.1, format="%.1f")
time_val = st.sidebar.slider("Time (h)", float(df[predictors[2]].min()), float(df[predictors[2]].max()), float(df[predictors[2]].mean()), step = 0.1, format="%.1f")

# PREDICTION
def prepare_point(temp_val, teos_val, time_val):
    point = pd.DataFrame({
        'Temperature (°C)': [temp_val],
        'TEOS (µL)': [teos_val],
        'Time (h)': [time_val]
    })
    expected_cols = scaler.feature_names_in_
    point_scaled = pd.DataFrame(scaler.transform(point[expected_cols]), columns=expected_cols)

    for p in predictors:
        point_scaled[f'{p}^2'] = point_scaled[p]**2

    for term in model_zeta.model.exog_names:
        if ':' in term:
            a, b = term.split(':')
            point_scaled[term] = point_scaled[a] * point_scaled[b]

    point_const = add_constant(point_scaled, has_constant='add')
    return point_const

pt = prepare_point(temp_val, teos_val, time_val)
z_pred = model_zeta.predict(pt[model_zeta.model.exog_names])
s_pred = model_shell.predict(pt[model_shell.model.exog_names])

# DESIRABILITY FUNCTION
def d_min(x, low_good, high_bad):
    return np.where(x <= low_good, 1.0,
                    np.where(x >= high_bad, 0.0,
                             (high_bad - x) / (high_bad - low_good)))

d_z = d_min(z_pred, -40, -30)
d_s = d_min(s_pred, 25, 45)
des_score = (d_z**0.3) * (d_s**0.7)

# RESULTS
st.markdown(f"""
### ⌨️ RESULTS
- **Predicted zeta potential:** {z_pred[0]:.1f} mV
- **Predicted shell thickness:** {s_pred[0]:.1f} nm
- **Global desirability score:** {des_score[0]:.1f}
""")

# === WYKRES 3D ===
st.markdown("### 📊 3D VISUALIZATION")
fig = plt.figure(figsize=(6, 5))
ax = fig.add_subplot(111, projection='3d')
norm = Normalize(vmin=desirability.min(), vmax=desirability.max())
sm = cm.ScalarMappable(cmap='viridis', norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.4, aspect=30, location="right", pad=0.1)
font_lbl = fm.FontProperties(family="Carlito", weight="bold", size=14)
cbar.set_label('Desirability', fontsize=12, fontproperties=font_lbl)
font_lbl_small = fm.FontProperties(family="Carlito", size=10)
for label in cbar.ax.get_yticklabels():
    label.set_fontproperties(font_lbl_small)
ax.scatter(grid['Time (h)'], grid['TEOS (µL)'], grid['Temperature (°C)'],
           c=desirability, cmap='viridis', s=2, alpha=0.1, zorder=1)
ax.scatter(time_val, teos_val, temp_val, color='red', s=200, marker='x', linewidth=3, zorder=20)
ax.set_xlabel('Time (h)', fontsize=14, labelpad=0, fontproperties=font_lbl) #new
ax.set_ylabel('TEOS (µL)', fontsize=14, labelpad=0, fontproperties=font_lbl) #new
ax.set_zlabel('Temperature (°C)', fontsize=14, labelpad=0, fontproperties=font_lbl) #new
ax.tick_params(axis='both', which='major', labelsize=10, width=0.1, pad=-1)
ax.xaxis.line.set_linewidth(0.5)
ax.yaxis.line.set_linewidth(0.5)
ax.zaxis.line.set_linewidth(0.5)
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0)) 
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)


# In[ ]:




