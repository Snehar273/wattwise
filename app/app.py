import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import sqlite3
import os
from prophet import Prophet
from datetime import datetime

st.set_page_config(
    page_title="WattWise — Smart Electricity Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], * {
    font-family: 'Outfit', sans-serif !important;
}

.stApp { background: #f0f4f8 !important; }
.stApp > header { background: transparent !important; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1400px !important; }

/* ── Hide keyboard_double_arrow_right sidebar toggle ── */
[data-testid="collapsedControl"]              { display: none !important; }
[data-testid="stSidebarCollapseButton"]       { display: none !important; }
button[data-testid="stSidebarCollapseButton"] { display: none !important; }
header[data-testid="stHeader"] button         { display: none !important; }
.st-emotion-cache-1cypcdb                     { display: none !important; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 2px solid #e2e8f0 !important;
    min-width: 280px !important;
    display: block !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1rem 1.2rem !important; }
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] * { font-family: 'Outfit', sans-serif !important; color: #334155 !important; }
[data-testid="stSidebar"] strong, [data-testid="stSidebar"] b { color: #0f172a !important; font-weight: 700 !important; }
[data-testid="stSidebar"] label { font-size: 13px !important; font-weight: 600 !important; color: #475569 !important; }
[data-testid="stSidebar"] input {
    background: #f8fafc !important; border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important; color: #0f172a !important;
    font-size: 14px !important; font-family: 'Outfit', sans-serif !important;
}
[data-testid="stSidebar"] input:focus {
    border-color: #f97316 !important;
    box-shadow: 0 0 0 3px rgba(249,115,22,0.1) !important;
}
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #f8fafc !important; border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important; font-size: 14px !important; color: #0f172a !important;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 55%, #0c4a6e 100%);
    border-radius: 20px; padding: 2.2rem 2.8rem;
    margin-bottom: 1.6rem; position: relative; overflow: hidden;
}
.hero::before {
    content: ''; position: absolute;
    width: 260px; height: 260px; top: -70px; right: -50px;
    background: radial-gradient(circle, rgba(251,191,36,0.18) 0%, transparent 65%);
    border-radius: 50%;
}
.hero-pill {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(251,191,36,0.15); border: 1.5px solid rgba(251,191,36,0.35);
    color: #fbbf24; font-size: 11px; font-weight: 700; padding: 5px 14px;
    border-radius: 100px; letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 0.8rem;
}
.hero-title { font-size: 2.4rem; font-weight: 800; color: #ffffff; margin: 0 0 0.4rem; letter-spacing: -0.5px; line-height: 1.1; }
.hero-title .acc { color: #fbbf24; }
.hero-sub { font-size: 1rem; color: #94a3b8; font-weight: 400; }

/* ── Metric cards ── */
.cards-row { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 1.6rem; }
.card {
    background: #ffffff; border-radius: 16px; padding: 1.5rem 1.6rem;
    border: 2px solid #e2e8f0; position: relative; overflow: hidden;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05); transition: transform 0.18s, box-shadow 0.18s;
    height: 130px; box-sizing: border-box;
}
.card:hover { transform: translateY(-3px); box-shadow: 0 10px 28px rgba(0,0,0,0.09); }
.card-top { height: 5px; position: absolute; top: 0; left: 0; right: 0; border-radius: 16px 16px 0 0; }
.card-icon { position: absolute; top: 1.2rem; right: 1.3rem; width: 40px; height: 40px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 18px; }
.card-label { font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px; color: #94a3b8; margin-bottom: 6px; padding-top: 12px; }
.card-value { font-size: 2rem; font-weight: 800; line-height: 1; color: #0f172a; margin-bottom: 5px; }
.card-sub { font-size: 12px; color: #94a3b8; }
.card.orange .card-top  { background: linear-gradient(90deg,#f97316,#fb923c); }
.card.orange .card-icon { background: #fff7ed; color: #f97316; }
.card.blue   .card-top  { background: linear-gradient(90deg,#3b82f6,#60a5fa); }
.card.blue   .card-icon { background: #eff6ff; color: #3b82f6; }
.card.teal   .card-top  { background: linear-gradient(90deg,#14b8a6,#2dd4bf); }
.card.teal   .card-icon { background: #f0fdfa; color: #14b8a6; }
.card.violet .card-top  { background: linear-gradient(90deg,#8b5cf6,#a78bfa); }
.card.violet .card-icon { background: #f5f3ff; color: #8b5cf6; }

/* ── Section heading ── */
.sh {
    font-size: 1.05rem; font-weight: 700; color: #0f172a;
    margin: 1.4rem 0 0.8rem; display: flex; align-items: center; gap: 8px;
}
.sh::after { content: ''; flex: 1; height: 2px; background: linear-gradient(90deg,#e2e8f0,transparent); border-radius: 2px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #ffffff !important; border-radius: 14px !important; padding: 5px !important;
    gap: 4px !important; border: 2px solid #e2e8f0 !important; box-shadow: 0 1px 6px rgba(0,0,0,0.05) !important;
}
.stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: 10px !important; color: #64748b !important; font-weight: 600 !important; font-size: 14px !important; padding: 9px 22px !important; border: none !important; }
.stTabs [aria-selected="true"] { background: #0f172a !important; color: #fbbf24 !important; }

/* ── Buttons (default orange) ── */
.stButton > button {
    background: linear-gradient(135deg,#f97316,#ea580c) !important;
    color: #ffffff !important; font-weight: 700 !important; font-size: 14px !important;
    border: none !important; border-radius: 10px !important; padding: 0.5rem 1.3rem !important;
    transition: opacity 0.18s, transform 0.15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* ── ICON BUTTONS for Save (green) and Delete (red) in data table ── */
.save-icon-btn > div > button {
    background: #f0fdf4 !important;
    color: #16a34a !important;
    border: 2px solid #bbf7d0 !important;
    border-radius: 10px !important;
    padding: 0.45rem 0.7rem !important;
    font-size: 18px !important;
    font-weight: 400 !important;
    line-height: 1 !important;
    min-width: 42px !important;
    width: 42px !important;
    transition: background 0.15s, border-color 0.15s, transform 0.12s !important;
}
.save-icon-btn > div > button:hover {
    background: #dcfce7 !important;
    border-color: #86efac !important;
    transform: scale(1.08) !important;
    opacity: 1 !important;
}

.del-icon-btn > div > button {
    background: #fef2f2 !important;
    color: #dc2626 !important;
    border: 2px solid #fecaca !important;
    border-radius: 10px !important;
    padding: 0.45rem 0.7rem !important;
    font-size: 18px !important;
    font-weight: 400 !important;
    line-height: 1 !important;
    min-width: 42px !important;
    width: 42px !important;
    transition: background 0.15s, border-color 0.15s, transform 0.12s !important;
}
.del-icon-btn > div > button:hover {
    background: #fee2e2 !important;
    border-color: #fca5a5 !important;
    transform: scale(1.08) !important;
    opacity: 1 !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="thumb"] { background: #f97316 !important; border: 3px solid #ffffff !important; box-shadow: 0 2px 8px rgba(249,115,22,0.35) !important; }
.stSlider [data-baseweb="track-fill"] { background: #f97316 !important; }

/* ── Forecast table ── */
.fc-wrap { background:#ffffff; border:2px solid #e2e8f0; border-radius:16px; overflow:hidden; margin-bottom:1rem; }
.fc-table { width:100%; border-collapse:collapse; }
.fc-table th { background:#f8fafc; color:#64748b; font-weight:700; font-size:11px; text-transform:uppercase; letter-spacing:1px; padding:14px 18px; border-bottom:2px solid #e2e8f0; text-align:left; }
.fc-table td { padding:14px 18px; border-bottom:1px solid #f1f5f9; font-size:14px; color:#334155; }
.fc-table tr:last-child td { border-bottom:none; }
.fc-table tr:hover td { background:#f8fafc; }
.fc-month { font-weight:700; color:#0f172a; font-size:15px; }
.fc-kwh   { color:#f97316; font-weight:700; font-size:15px; }
.fc-bill b { color:#14b8a6; font-weight:700; }
.bup { background:#fee2e2; color:#ef4444; padding:3px 9px; border-radius:100px; font-size:11px; font-weight:700; margin-left:6px; }
.bdn { background:#dcfce7; color:#16a34a; padding:3px 9px; border-radius:100px; font-size:11px; font-weight:700; margin-left:6px; }

/* ── Tip cards ── */
.tip { background:#fffbeb; border:2px solid #fde68a; border-left:5px solid #f59e0b; border-radius:12px; padding:1rem 1.4rem; margin-top:1.2rem; }
.tip.warn { background:#fff1f2; border-color:#fecdd3; border-left-color:#f43f5e; }
.tip-t { font-weight:700; font-size:14px; color:#b45309; margin-bottom:4px; }
.tip.warn .tip-t { color:#be123c; }
.tip-b { font-size:13px; color:#78716c; line-height:1.65; }

/* ── How-to cards ── */
.hw-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:16px; margin-bottom:1rem; }
.hw-card { background:#ffffff; border:2px solid #e2e8f0; border-radius:16px; padding:1.6rem; box-shadow:0 1px 4px rgba(0,0,0,0.04); display:flex; flex-direction:column; min-height:200px; }
.hw-num  { font-size:2.4rem; font-weight:800; color:#f97316; opacity:0.22; line-height:1; margin-bottom:12px; }
.hw-t    { font-weight:700; font-size:15px; color:#0f172a; margin-bottom:8px; }
.hw-b    { font-size:13px; color:#64748b; line-height:1.65; flex:1; }

/* ── Sample table ── */
.sm-wrap { background:#ffffff; border:2px solid #e2e8f0; border-radius:16px; overflow:hidden; }
.sm-table { width:100%; border-collapse:collapse; }
.sm-table th { background:#f8fafc; color:#64748b; font-weight:700; font-size:11px; text-transform:uppercase; letter-spacing:1px; padding:12px 18px; border-bottom:2px solid #e2e8f0; text-align:left; }
.sm-table td { padding:11px 18px; border-bottom:1px solid #f1f5f9; font-size:13px; color:#334155; }
.sm-table tr:last-child td { border-bottom:none; }
.sm-table b { color:#f97316; }

/* ── Stat chips ── */
.chip { display:inline-block; background:#f0f9ff; border:2px solid #bae6fd; border-radius:12px; padding:12px 20px; text-align:center; margin-right:10px; margin-bottom:10px; }
.chip-v { font-size:1.5rem; font-weight:800; color:#0369a1; line-height:1; }
.chip-l { font-size:11px; color:#7dd3fc; font-weight:500; margin-top:3px; }

/* ── Sidebar logo ── */
.logo-area { text-align:center; padding:0.5rem 0 1rem; border-bottom:2px solid #e2e8f0; margin-bottom:1.2rem; }
.logo-title { font-size:1.45rem; font-weight:800; color:#0f172a !important; margin-top:6px; letter-spacing:-0.3px; }
.logo-title .acc { color:#f97316 !important; }
.logo-sub { font-size:11px; color:#94a3b8 !important; letter-spacing:0.4px; margin-top:2px; }
</style>
""", unsafe_allow_html=True)

# ── DB ──────────────────────────────────────────────────────────
DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'user_data.db')

def init_db():
    c = sqlite3.connect(DB_PATH)
    c.cursor().execute('''CREATE TABLE IF NOT EXISTS usage_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        month TEXT NOT NULL UNIQUE,
        kwh REAL NOT NULL,
        added_on TEXT DEFAULT CURRENT_TIMESTAMP)''')
    c.commit(); c.close()

def save_entry(month, kwh):
    c = sqlite3.connect(DB_PATH)
    c.cursor().execute("INSERT OR REPLACE INTO usage_entries (month,kwh) VALUES(?,?)", (month,kwh))
    c.commit(); c.close()

def update_entry(eid, kwh):
    c = sqlite3.connect(DB_PATH)
    c.cursor().execute("UPDATE usage_entries SET kwh=? WHERE id=?", (kwh,eid))
    c.commit(); c.close()

def delete_entry(eid):
    c = sqlite3.connect(DB_PATH)
    c.cursor().execute("DELETE FROM usage_entries WHERE id=?", (eid,))
    c.commit(); c.close()

def load_entries():
    c = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM usage_entries ORDER BY month ASC", c)
    c.close()
    return df

init_db()

# ── Model ───────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'prophet_model.pkl')

@st.cache_resource
def load_model():
    with open(MODEL_PATH, 'rb') as f:
        return pickle.load(f)

model = load_model()

# ── Plot base ───────────────────────────────────────────────────
def base_layout(h=320, legend=True, ly=-0.22):
    d = dict(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Outfit', color='#94a3b8', size=12),
        xaxis=dict(gridcolor='#f1f5f9', linecolor='#e2e8f0',
                   tickcolor='#e2e8f0', tickfont=dict(color='#94a3b8', family='Outfit')),
        yaxis=dict(gridcolor='#f1f5f9', linecolor='#e2e8f0',
                   tickcolor='#e2e8f0', tickfont=dict(color='#94a3b8', family='Outfit')),
        margin=dict(t=20, b=20, l=10, r=10),
        hovermode='x unified',
        hoverlabel=dict(bgcolor='#0f172a', bordercolor='#1e3a5f',
                        font=dict(color='#ffffff', size=13, family='Outfit')),
        height=h, showlegend=legend,
    )
    if legend:
        d['legend'] = dict(orientation='h', y=ly,
                           font=dict(color='#64748b', size=12, family='Outfit'),
                           bgcolor='rgba(0,0,0,0)')
    return d

# ══ SIDEBAR ══════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="logo-area">
      <svg width="52" height="52" viewBox="0 0 52 52" fill="none">
        <rect width="52" height="52" rx="15" fill="#0f172a"/>
        <polygon points="31,8 18,28 27,28 22,44 35,24 26,24" fill="#fbbf24"/>
        <circle cx="39" cy="14" r="4" fill="#fbbf24" opacity="0.25"/>
        <circle cx="13" cy="39" r="3" fill="#38bdf8" opacity="0.25"/>
      </svg>
      <div class="logo-title">Watt<span class="acc">Wise</span></div>
      <div class="logo-sub">AI-Powered Energy Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Add Monthly Entry**")

    MONTH_NAMES = ["January","February","March","April","May","June",
                   "July","August","September","October","November","December"]
    YEARS = [str(y) for y in range(2020, 2027)]

    mc1, mc2 = st.columns(2)
    with mc1:
        sel_m = st.selectbox("Month", MONTH_NAMES, index=0, label_visibility="visible")
    with mc2:
        sel_y = st.selectbox("Year", YEARS, index=YEARS.index("2024"), label_visibility="visible")

    kwh_str  = st.text_input("Usage (kWh)", value="", placeholder="e.g.  420")
    rate_val = st.number_input("Rate (₹ / kWh)", min_value=1.0, max_value=20.0,
                                value=6.0, step=0.5, format="%.1f")

    if st.button("💾  Save Entry", use_container_width=True):
        try:
            kwh_val = float(kwh_str)
            if kwh_val <= 0:
                st.error("kWh must be greater than 0")
            else:
                month_str = f"{sel_y}-{MONTH_NAMES.index(sel_m)+1:02d}-01"
                save_entry(month_str, kwh_val)
                st.success(f"✅ Saved {kwh_val} kWh for {sel_m} {sel_y}")
                st.rerun()
        except ValueError:
            st.error("Please enter a valid number for kWh")

    st.divider()
    months_ahead = st.slider("Months to predict", 1, 12, 3)
    st.caption("Add 3+ months of data for personalised predictions.")

# ── Data ────────────────────────────────────────────────────────
user_df = load_entries()
if not user_df.empty:
    user_df['month'] = pd.to_datetime(user_df['month'])
    user_df = user_df.sort_values('month').reset_index(drop=True)

R        = rate_val
avg_kwh  = round(user_df['kwh'].mean(), 1) if not user_df.empty else 0
max_kwh  = round(user_df['kwh'].max(),  1) if not user_df.empty else 0
avg_bill = int(round(avg_kwh * R, 0))
n_months = len(user_df)

# ══ HERO ══════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="hero-pill">AI Energy Intelligence</div>
  <div class="hero-title">Watt<span class="acc">Wise</span></div>
  <div class="hero-sub">Predict your electricity bill before it arrives — powered by machine learning</div>
</div>
""", unsafe_allow_html=True)

# ══ METRIC CARDS ══════════════════════════════════════════════════
st.markdown(f"""
<div class="cards-row">
  <div class="card orange">
    <div class="card-top"></div>
    <div class="card-icon">&#9889;</div>
    <div class="card-label">Avg Monthly Usage</div>
    <div class="card-value">{avg_kwh}</div>
    <div class="card-sub">kWh per month</div>
  </div>
  <div class="card blue">
    <div class="card-top"></div>
    <div class="card-icon">&#128200;</div>
    <div class="card-label">Peak Usage</div>
    <div class="card-value">{max_kwh}</div>
    <div class="card-sub">kWh highest month</div>
  </div>
  <div class="card teal">
    <div class="card-top"></div>
    <div class="card-icon">&#128176;</div>
    <div class="card-label">Avg Monthly Bill</div>
    <div class="card-value">&#8377;{avg_bill}</div>
    <div class="card-sub">at &#8377;{R}/kWh</div>
  </div>
  <div class="card violet">
    <div class="card-top"></div>
    <div class="card-icon">&#128197;</div>
    <div class="card-label">Months Tracked</div>
    <div class="card-value">{n_months}</div>
    <div class="card-sub">data points recorded</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ══ TABS ══════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard", "Predictions", "My Data", "How It Works"
])

# ══ TAB 1 ════════════════════════════════════════════════════════
with tab1:
    if user_df.empty:
        st.markdown("""
        <div style="text-align:center;padding:5rem 1rem;">
          <div style="font-size:4rem;margin-bottom:1rem;">&#9889;</div>
          <div style="font-size:1.3rem;font-weight:700;color:#64748b;">No data yet</div>
          <div style="font-size:14px;color:#94a3b8;margin-top:6px;">
            Add your monthly electricity usage from the sidebar to get started.</div>
        </div>""", unsafe_allow_html=True)
    else:
        c1, c2 = st.columns([2, 1])
        with c1:
            st.markdown('<div class="sh">Usage Trend</div>', unsafe_allow_html=True)
            f1 = go.Figure()
            f1.add_trace(go.Scatter(
                x=user_df['month'], y=user_df['kwh'],
                fill='tozeroy', fillcolor='rgba(249,115,22,0.07)',
                line=dict(color='#f97316', width=3), mode='lines+markers',
                marker=dict(size=8, color='#f97316', line=dict(width=2.5, color='#ffffff')),
                name='Usage (kWh)',
                hovertemplate='<b>%{x|%B %Y}</b><br>%{y:.1f} kWh<extra></extra>'
            ))
            f1.add_trace(go.Scatter(
                x=user_df['month'], y=[user_df['kwh'].mean()] * len(user_df),
                line=dict(color='#3b82f6', width=1.8, dash='dot'),
                mode='lines', name='Average', hoverinfo='skip'
            ))
            f1.update_layout(**base_layout(300, legend=True, ly=-0.22))
            f1.update_layout(yaxis_title='kWh')
            st.plotly_chart(f1, use_container_width=True)

        with c2:
            st.markdown('<div class="sh">Usage Split</div>', unsafe_allow_html=True)
            f2 = go.Figure(go.Pie(
                labels=['AC/HVAC','Laundry','Kitchen','Other'],
                values=[35, 20, 15, 30], hole=0.64,
                marker=dict(colors=['#f97316','#3b82f6','#14b8a6','#8b5cf6'],
                            line=dict(color='#ffffff', width=3)),
                textinfo='percent', textfont=dict(size=12, family='Outfit')
            ))
            f2.add_annotation(text=f"<b>{int(avg_kwh)}</b><br>kWh avg",
                              x=0.5, y=0.5, showarrow=False,
                              font=dict(size=15, color='#0f172a', family='Outfit'))
            f2.update_layout(
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Outfit', color='#94a3b8', size=12),
                margin=dict(t=20, b=20, l=10, r=10), height=300, showlegend=True,
                legend=dict(orientation='v', x=1.0, y=0.5,
                            font=dict(color='#64748b', size=11, family='Outfit'),
                            bgcolor='rgba(0,0,0,0)')
            )
            st.plotly_chart(f2, use_container_width=True)

        st.markdown('<div class="sh">Month-by-Month Breakdown</div>', unsafe_allow_html=True)
        avg  = user_df['kwh'].mean()
        cols = ['#f43f5e' if k > avg else '#f97316' for k in user_df['kwh']]
        f3   = go.Figure(go.Bar(
            x=user_df['month'].dt.strftime('%b %Y'), y=user_df['kwh'],
            marker_color=cols, marker_line_color='rgba(0,0,0,0)',
            text=user_df['kwh'].round(0).astype(int), textposition='outside',
            textfont=dict(color='#64748b', size=12, family='Outfit'),
            hovertemplate='<b>%{x}</b><br>%{y:.1f} kWh<extra></extra>'
        ))
        f3.add_hline(y=avg, line_dash='dot', line_color='#3b82f6',
                     annotation_text=f'Avg {avg:.0f} kWh',
                     annotation_font=dict(color='#3b82f6', size=11, family='Outfit'))
        f3.update_layout(**base_layout(290, legend=False))
        f3.update_layout(yaxis_title='kWh')
        st.plotly_chart(f3, use_container_width=True)

# ══ TAB 2 — Predictions ════════════════════════════════════════
with tab2:
    if user_df.empty:
        st.info("Add your usage data from the sidebar first!")
    elif len(user_df) < 3:
        st.info("Add at least 3 months of your own data for personalised predictions.")
    else:
        pdf = user_df[['month','kwh']].rename(columns={'month':'ds','kwh':'y'})
        um  = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
            interval_width=0.80
        )
        if len(user_df) >= 13:
            um.add_seasonality(name='yearly', period=365.25, fourier_order=3)
        um.fit(pdf)

        future    = um.make_future_dataframe(periods=months_ahead, freq='MS')
        forecast  = um.predict(future)
        future_fc = forecast.tail(months_ahead).copy()

        hist_max = float(user_df['kwh'].max())
        hist_min = float(user_df['kwh'].min())
        future_fc['yhat']       = future_fc['yhat'].clip(lower=hist_min*0.6, upper=hist_max*1.4)
        future_fc['yhat_lower'] = future_fc['yhat_lower'].clip(lower=hist_min*0.6, upper=hist_max*1.4)
        future_fc['yhat_upper'] = future_fc['yhat_upper'].clip(lower=hist_min*0.6, upper=hist_max*1.4)

        st.success(f"Model personalised on your {len(user_df)} months of data!")
        st.markdown('<div class="sh">Bill Forecast</div>', unsafe_allow_html=True)

        prev = float(user_df['kwh'].iloc[-1])
        rows = ""
        for _, row in future_fc.iterrows():
            kp   = round(row['yhat'], 1)
            bmin = int(round(row['yhat_lower'] * R, 0))
            bmid = int(round(row['yhat'] * R, 0))
            bmax = int(round(row['yhat_upper'] * R, 0))
            diff  = kp - prev
            badge = (f'<span class="bup">up +{abs(diff):.0f}</span>'      if diff > 2
                     else f'<span class="bdn">down -{abs(diff):.0f}</span>' if diff < -2
                     else '<span style="background:#f1f5f9;color:#64748b;padding:3px 9px;border-radius:100px;font-size:11px;font-weight:700;margin-left:6px;">stable</span>')
            rows += f"""<tr>
              <td class="fc-month">{row['ds'].strftime('%B %Y')}</td>
              <td class="fc-kwh">{kp} kWh {badge}</td>
              <td class="fc-bill">&#8377;{bmin} – <b>&#8377;{bmid}</b> – &#8377;{bmax}</td>
            </tr>"""
            prev = kp

        st.markdown(f"""
        <div class="fc-wrap"><table class="fc-table">
          <thead><tr>
            <th>Month</th><th>Predicted Usage</th>
            <th>Bill Range (Min – Expected – Max)</th>
          </tr></thead>
          <tbody>{rows}</tbody>
        </table></div>""", unsafe_allow_html=True)

        st.markdown('<div class="sh">Forecast Chart</div>', unsafe_allow_html=True)
        f4 = go.Figure()
        f4.add_trace(go.Scatter(
            x=user_df['month'], y=user_df['kwh'], mode='lines+markers', name='Actual',
            line=dict(color='#f97316', width=3),
            marker=dict(size=7, color='#f97316', line=dict(width=2.5, color='#ffffff')),
            hovertemplate='<b>%{x|%B %Y}</b><br>Actual: %{y:.1f} kWh<extra></extra>'
        ))
        f4.add_trace(go.Scatter(
            x=pd.concat([future_fc['ds'], future_fc['ds'][::-1]]),
            y=pd.concat([future_fc['yhat_upper'], future_fc['yhat_lower'][::-1]]),
            fill='toself', fillcolor='rgba(59,130,246,0.07)',
            line=dict(color='rgba(0,0,0,0)'), name='Confidence Band', hoverinfo='skip'
        ))
        f4.add_trace(go.Scatter(
            x=future_fc['ds'], y=future_fc['yhat'].round(1),
            mode='lines+markers', name='Predicted',
            line=dict(color='#3b82f6', width=3, dash='dash'),
            marker=dict(size=8, color='#3b82f6', line=dict(width=2.5, color='#ffffff')),
            hovertemplate='<b>%{x|%B %Y}</b><br>Predicted: %{y:.1f} kWh<extra></extra>'
        ))
        f4.update_layout(**base_layout(360, legend=True, ly=-0.18))
        f4.update_layout(yaxis_title='kWh')
        st.plotly_chart(f4, use_container_width=True)

        ap = float(future_fc['yhat'].mean())
        if ap > float(user_df['kwh'].mean()) * 1.05:
            dp = round((ap / float(user_df['kwh'].mean()) - 1) * 100, 1)
            st.markdown(f"""<div class="tip warn">
              <div class="tip-t">Usage is Rising</div>
              <div class="tip-b">Predicted usage is <b>{dp}% above</b> your average.
              Try: reduce AC by 2 degrees, switch to LED lighting, unplug idle devices.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="tip">
              <div class="tip-t">Usage Looks Stable</div>
              <div class="tip-b">Your predicted usage is stable or falling.
              Tip: Unplugging standby devices saves an extra 5-10% per month!</div>
            </div>""", unsafe_allow_html=True)

# ══ TAB 3 — My Data ══════════════════════════════════════════════
with tab3:
    fresh = load_entries()
    if fresh.empty:
        st.info("No entries yet. Add data from the sidebar to get started!")
    else:
        fresh['month'] = pd.to_datetime(fresh['month'])
        fresh = fresh.sort_values('month').reset_index(drop=True)

        st.markdown('<div class="sh">All Entries</div>', unsafe_allow_html=True)

        # Column headers
        h0,h1,h2,h3,h4,h5 = st.columns([2.2, 1.6, 1.6, 1.4, 0.55, 0.55])
        for col, lbl in zip([h0,h1,h2,h3,h4,h5],
                            ["Month","Usage (kWh)","Est. Bill","New kWh","Save","Delete"]):
            col.markdown(
                f"<div style='font-size:11px;font-weight:700;text-transform:uppercase;"
                f"letter-spacing:1px;color:#64748b;padding-bottom:8px;"
                f"border-bottom:2px solid #e2e8f0;'>{lbl}</div>",
                unsafe_allow_html=True)

        for _, row in fresh.iterrows():
            rid   = int(row['id'])
            month = row['month'].strftime('%B %Y')
            kwh   = float(row['kwh'])
            bill  = int(round(kwh * R, 0))

            c0,c1,c2,c3,c4,c5 = st.columns([2.2, 1.6, 1.6, 1.4, 0.55, 0.55])

            with c0:
                st.markdown(f"<div style='padding:10px 0 6px;font-weight:700;color:#0f172a;font-size:14px'>{month}</div>", unsafe_allow_html=True)
            with c1:
                st.markdown(f"<div style='padding:10px 0 6px;color:#f97316;font-weight:700;font-size:14px'>{kwh} kWh</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div style='padding:10px 0 6px;color:#14b8a6;font-weight:600;font-size:14px'>&#8377;{bill}</div>", unsafe_allow_html=True)
            with c3:
                new_v = st.number_input("", min_value=1.0, max_value=5000.0,
                                         value=kwh, step=10.0,
                                         key=f"nv_{rid}", label_visibility="collapsed")
            with c4:
                # Green checkmark icon button
                st.markdown('<div class="save-icon-btn">', unsafe_allow_html=True)
                if st.button("✔", key=f"sv_{rid}", help=f"Save changes for {month}"):
                    update_entry(rid, new_v)
                    st.toast(f"Updated {month}!")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            with c5:
                # Red trash icon button
                st.markdown('<div class="del-icon-btn">', unsafe_allow_html=True)
                if st.button("✕", key=f"dl_{rid}", help=f"Delete {month}"):
                    delete_entry(rid)
                    st.toast(f"Deleted {month}")
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<hr style='margin:0;border:none;border-top:1px solid #f1f5f9'>",
                        unsafe_allow_html=True)

        st.markdown("---")
        dexp = fresh.copy()
        dexp['Month']       = dexp['month'].dt.strftime('%B %Y')
        dexp['Usage (kWh)'] = dexp['kwh']
        dexp['Est. Bill']   = (dexp['kwh'] * R).round(0).astype(int)
        csv = dexp[['Month','Usage (kWh)','Est. Bill']].to_csv(index=False).encode()
        st.download_button("⬇️  Download My Data as CSV", data=csv,
                           file_name="wattwise_data.csv", mime="text/csv")

# ══ TAB 4 ════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sh">How to Use WattWise</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="hw-grid">
      <div class="hw-card">
        <div class="hw-num">01</div>
        <div class="hw-t">Add Your Data</div>
        <div class="hw-b">Use the sidebar to enter your monthly kWh reading.
        You will find this number on your electricity bill each month.</div>
      </div>
      <div class="hw-card">
        <div class="hw-num">02</div>
        <div class="hw-t">Set Your Rate</div>
        <div class="hw-b">Enter your electricity rate (Rs/kWh).
        The default is Rs 6 — check your actual bill for the exact rate in your area.</div>
      </div>
      <div class="hw-card">
        <div class="hw-num">03</div>
        <div class="hw-t">Get Predictions</div>
        <div class="hw-b">Go to the Predictions tab, pick how many months ahead,
        and see your AI-powered bill forecast instantly with confidence ranges.</div>
      </div>
      <div class="hw-card">
        <div class="hw-num">04</div>
        <div class="hw-t">Track Monthly</div>
        <div class="hw-b">Keep adding data every month. More data means smarter
        predictions that adapt to your personal usage patterns over time.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sh">Sample Data to Test Right Now</div>', unsafe_allow_html=True)
    st.markdown("""
    <p style="font-size:14px;color:#64748b;margin-bottom:10px;">
    Enter these one by one from the sidebar to see the full app in action:</p>
    <div class="sm-wrap"><table class="sm-table">
      <thead><tr><th>Month</th><th>Year</th><th>Usage (kWh)</th><th>Reason</th></tr></thead>
      <tbody>
        <tr><td>January</td><td>2024</td><td><b>420</b></td><td>Winter — heater usage high</td></tr>
        <tr><td>February</td><td>2024</td><td><b>390</b></td><td>Shorter month, slightly lower</td></tr>
        <tr><td>March</td><td>2024</td><td><b>310</b></td><td>Spring — mild weather</td></tr>
        <tr><td>April</td><td>2024</td><td><b>280</b></td><td>Best weather — lowest usage</td></tr>
        <tr><td>May</td><td>2024</td><td><b>340</b></td><td>Summer begins — AC kicks in</td></tr>
        <tr><td>June</td><td>2024</td><td><b>510</b></td><td>Peak summer — heavy AC use</td></tr>
        <tr><td>July</td><td>2024</td><td><b>530</b></td><td>Hottest month — maximum usage</td></tr>
        <tr><td>August</td><td>2024</td><td><b>490</b></td><td>Still hot but cooling slightly</td></tr>
        <tr><td>September</td><td>2024</td><td><b>380</b></td><td>Monsoon — AC reduces</td></tr>
        <tr><td>October</td><td>2024</td><td><b>290</b></td><td>Festival season — moderate</td></tr>
        <tr><td>November</td><td>2024</td><td><b>320</b></td><td>Cooler nights return</td></tr>
        <tr><td>December</td><td>2024</td><td><b>400</b></td><td>Winter — heating picks up</td></tr>
      </tbody>
    </table></div>""", unsafe_allow_html=True)

    st.markdown('<div class="sh">About the Model</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        st.markdown("""<div class="hw-card" style="min-height:auto">
          <div class="hw-t">What model powers WattWise?</div>
          <div class="hw-b" style="margin-bottom:12px">
          <b>Facebook Prophet</b> — a production-grade time series model built by Meta,
          used by thousands of companies worldwide for accurate forecasting at scale.
          </div>
          <div class="hw-t">Why Prophet?</div>
          <div class="hw-b">It automatically detects yearly seasonal patterns like summer
          spikes and winter heating, and handles missing months gracefully.</div>
        </div>""", unsafe_allow_html=True)
    with cb:
        st.markdown("""<div class="hw-card" style="min-height:auto">
          <div class="hw-t">Model Accuracy — Tested on Kaggle UCI Dataset</div>
          <div class="hw-b" style="margin-bottom:14px">
          Trained on 4 years of real household power data from 2006 to 2010 with 2 million+ readings.
          </div>
          <div class="chip"><div class="chip-v">10.48%</div><div class="chip-l">MAPE Error</div></div>
          <div class="chip"><div class="chip-v">71.77</div><div class="chip-l">MAE (kWh)</div></div>
          <div class="chip"><div class="chip-v">91.29</div><div class="chip-l">RMSE (kWh)</div></div>
        </div>""", unsafe_allow_html=True)

# ── Footer ───────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:2.5rem 0 0.5rem;color:#cbd5e1;font-size:13px;
border-top:2px solid #e2e8f0;margin-top:1.5rem;">
  WattWise &nbsp;&#183;&nbsp; Python &nbsp;&#183;&nbsp; Prophet &nbsp;&#183;&nbsp; Streamlit &nbsp;&#183;&nbsp; SQLite
</div>""", unsafe_allow_html=True)