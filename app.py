import streamlit as st
import akshare as ak
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import time
import os

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(layout="wide", page_title="A股 RRG 极速仓储版 v19")
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .stDeployButton, [data-testid="stToolbar"], footer {display:none;}
    [data-testid="stSidebar"] { min-width: 380px; }
    h1 { color: #00CC96; text-shadow: 2px 2px 4px #000000; }
    div.stButton > button { width: 100%; border-radius: 5px; }
    /* 核心修复：强行隐藏会导致动画崩溃的 Streamlit 原生全屏按钮 */
    button[title="View fullscreen"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 A股 RRG 极速仓储系统 (v19.0)")

# ==========================================
# 2. 本地仓库配置
# ==========================================
DATA_DIR = "rrg_data_warehouse"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

SECTOR_CONFIG = {
    "煤炭": {"etf": "sh515220", "keyword": "煤炭"},
    "钢铁": {"etf": "sh515210", "keyword": "钢铁"},
    "有色": {"etf": "sh512400", "keyword": "有色"},
    "石油": {"etf": "sh561360", "keyword": "石油"},
    "电力": {"etf": "sh561560", "keyword": "电力"},
    "化工": {"etf": "sh516020", "keyword": "化学"},
    "银行": {"etf": "sh512800", "keyword": "银行"},
    "证券": {"etf": "sh512880", "keyword": "证券"},
    "保险": {"etf": "sh515050", "keyword": "保险"},
    "房地产": {"etf": "sh512200", "keyword": "房地产"},
    "半导体": {"etf": "sh512480", "keyword": "半导体"},
    "芯片": {"etf": "sz159995", "keyword": "半导体"},
    "光伏": {"etf": "sh515790", "keyword": "光伏"},
    "新能车": {"etf": "sh515030", "keyword": "汽车整车"},
    "电池": {"etf": "sz159755", "keyword": "电池"},
    "白酒": {"etf": "sh512690", "keyword": "酿酒"},
    "医药": {"etf": "sh512010", "keyword": "医药"},
    "家电": {"etf": "sh561120", "keyword": "家电"},
    "游戏": {"etf": "sh516770", "keyword": "游戏"},
    "养殖": {"etf": "sz159865", "keyword": "农牧"},
    "通信": {"etf": "sh515880", "keyword": "通信"},
    "计算机": {"etf": "sz159998", "keyword": "计算机"},
}

# ==========================================
# 3. 辅助函数
# ==========================================
@st.cache_data(ttl=3600)
def get_real_board_code(keyword):
    try:
        df = ak.stock_board_industry_name_em()
        target = df[df['板块名称'] == keyword]
        if target.empty:
            target = df[df['板块名称'].str.contains(keyword)]
        if not target.empty:
            return target.iloc[0]['板块名称'], target.iloc[0]['板块代码']
        return None, None
    except: return None, None

@st.cache_data(ttl=3600)
def get_constituents_safe(board_name, limit):
    try:
        df = ak.stock_board_industry_cons_em(symbol=board_name)
        for col in ['总市值', '总市值(元)', '流通市值']:
            if col in df.columns:
                df = df.sort_values(by=col, ascending=False)
                break
        df = df.head(limit)
        res = {}
        for _, row in df.iterrows():
            code = str(row['代码'])
            fc = f"sh{code}" if code.startswith(('6','9','5')) else f"sz{code}"
            res[fc] = row['名称']
        return res
    except: return {}

# ==========================================
# 4. 侧边栏 (完整升级版)
# ==========================================
with st.sidebar:
    st.header("1️⃣ 视角选择")
    level = st.radio("模式", ["Level 1: 全行业 ETF 轮动", "Level 2: 板块内抓龙头"])
    
    current_pool = {}
    
    # ----------------------------------------
    # 🌟 新增：三大经典基准内置下拉菜单
    # ----------------------------------------
    BENCHMARK_DICT = {
        "沪深300 (大盘/机构视角)": "sh510300",
        "红利ETF (避险/熊市视角)": "sh510880",
        "中证2000 (微盘/游资视角)": "sh563300",
        "自定义 (输入美股或其它代码)": "custom"
    }
    bench_choice = st.selectbox("🎯 选择参照系基准 (坐标系中心)", list(BENCHMARK_DICT.keys()))
    
    if bench_choice == "自定义 (输入美股或其它代码)":
        benchmark_code = st.text_input("请输入基准代码 (如 spy, sh510300)", "spy").strip()
    else:
        benchmark_code = BENCHMARK_DICT[bench_choice]
        
    st.caption(f"当前生效基准: {benchmark_code}")
    # ----------------------------------------
    
    force_update = st.button("🔄 强制更新今日数据 (慢)", help="如果发现数据不是最新的，点此按钮强制重新下载")
    
    if "Level 1" in level:
        current_pool = {v['etf']: k for k, v in SECTOR_CONFIG.items()}
        if benchmark_code in current_pool: del current_pool[benchmark_code]
    else:
        sector_key = st.selectbox("选择行业", list(SECTOR_CONFIG.keys()))
        cfg = SECTOR_CONFIG[sector_key]
        real_name, real_code = get_real_board_code(cfg['keyword'])
        
        if real_name:
            # 默认：个股深挖时，基准自动切换为对应的行业ETF
            benchmark_code = cfg['etf'] 
            st.caption(f"板块: {real_name} | 自动切换为板块基准: {sector_key}ETF ({benchmark_code})")
            
            top_n = st.slider("龙头数", 5, 50, 20)
            with st.spinner("获取名单..."):
                current_pool = get_constituents_safe(real_name, top_n)
        else:
            st.error("板块匹配失败")
            
        extra = st.text_input("➕ 搅局者 (代码,名称)", "")
        if extra:
            p = extra.split(',')
            current_pool[p[0].strip()] = p[1].strip() if len(p)>1 else p[0].strip()

    # === 👇这里是上次被误删的参数设置部分，现在补回来了👇 ===
    st.divider()
    st.header("2️⃣ 参数")
    col1, col2 = st.columns(2)
    with col1:
        period = st.radio("周期", ["日线", "周线"], index=0)
    with col2:
        window = st.number_input("RS窗口", 5, 60, 14)
    
    period_code = 'W-FRI' if "周" in period else 'D'
    tail_len = st.slider("拖尾", 1, 20, 8)# ==========================================
# 5. 智能仓储引擎 (Local First)
# ==========================================
def fetch_net(code, start_date):
    """联网下载"""
    try:
        if any(x in code for x in ['51','15','56']): df = ak.fund_etf_hist_sina(symbol=code)
        else: df = ak.stock_zh_index_daily(symbol=code)
        if not df.empty:
            df['date'] = pd.to_datetime(df['date'])
            return df.set_index('date')[['close']]
    except: pass
    try:
        clean = code.replace("sh","").replace("sz","")
        df = ak.stock_zh_a_hist(symbol=clean, start_date=start_date.replace("-", ""), adjust="qfq")
        if not df.empty:
            df['日期'] = pd.to_datetime(df['日期'])
            return df.set_index('日期')[['收盘']].rename(columns={'收盘':'close'})
    except: pass
    return None

def get_data_smart(code, start_date, force_refresh=False):
    """核心逻辑：本地优先 -> 否则联网"""
    file_path = os.path.join(DATA_DIR, f"{code}.csv")
    today_str = datetime.date.today().strftime('%Y-%m-%d')
    
    # 1. 检查本地是否有文件，且是今天更新的
    if os.path.exists(file_path) and not force_refresh:
        # 获取文件修改时间
        mtime = datetime.date.fromtimestamp(os.path.getmtime(file_path))
        if mtime == datetime.date.today():
            # 是今天的数据，直接读！
            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                return df['close']
            except: pass # 文件坏了，往下走
            
    # 2. 如果强制刷新，或者本地没有，或者文件过期 -> 联网下载
    df_new = fetch_net(code, start_date)
    if df_new is not None and not df_new.empty:
        try:
            df_new.to_csv(file_path) # 存档
            return df_new['close']
        except: pass
        return df_new['close']
        
    # 3. 实在连不上网，哪怕是旧文件也拿出来顶替
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            return df['close']
        except: pass
        
    return None

@st.cache_data(ttl=3600)
def load_data_v19(pool, bench, start, _force):
    data = {}
    fails = []
    status = st.empty()
    bar = st.progress(0)
    
    # 基准
    status.text(f"读取基准: {bench}...")
    b_s = get_data_smart(bench, start, _force)
    if b_s is None: return None, ["基准"]
    
    data['__BENCH__'] = b_s
    full_idx = b_s.index
    
    # 标的
    total = len(pool)
    for i, (k, v) in enumerate(pool.items()):
        status.text(f"读取数据 ({i+1}/{total}): {v}...")
        bar.progress((i+1)/total)
        
        s = get_data_smart(k, start, _force)
        if s is not None:
            s = s.reindex(full_idx).ffill()
            data[v] = s
        else:
            fails.append(v)
            
    status.empty(); bar.empty()
    return pd.DataFrame(data), fails

# ==========================================
# 6. 计算逻辑
# ==========================================
def calculate_rrg(df, period, window, tail):
    if period == 'D': df_res = df
    else: df_res = df.resample(period).last()
    
    df_res = df_res.dropna(how='all')
    if len(df_res) < window + 5: return pd.DataFrame(), [], "数据长度不足"

    bench = df_res['__BENCH__']
    worm_data = []
    dates = df_res.index[window+10:]
    if len(dates) > 52: dates = dates[-52:]
    str_dates = [d.strftime('%Y-%m-%d') for d in dates]
    
    for col in df_res.columns:
        if col == '__BENCH__': continue
        series = df_res[col]
        if series.notna().sum() < window + 5: continue
        
        rs = series / bench
        ratio = 100 + ((rs - rs.rolling(window).mean()) / rs.rolling(window).std())
        mom = 100 + ((ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std())
        
        # 平滑
        ratio = ratio.rolling(3).mean()
        mom = mom.rolling(3).mean()
        
        temp = pd.DataFrame({'R': ratio, 'M': mom, 'P': series}, index=df_res.index)
        
        for d_str in str_dates:
            try:
                cur = pd.to_datetime(d_str)
                hist = temp.loc[:cur].tail(tail + 1)
                if len(hist) > 0 and not np.isnan(hist.iloc[-1]['R']):
                    worm_data.append({
                        'Frame': d_str,
                        'Name': col,
                        'X': hist['R'].tolist(),
                        'Y': hist['M'].tolist(),
                        'P': hist['P'].iloc[-1]
                    })
            except: pass
            
    return pd.DataFrame(worm_data), str_dates, "OK"

# ==========================================
# 7. 主程序
# ==========================================
if st.button("🚀 开始分析", type="primary"):
    start_date = "2021-01-01"
    
    # 只有点击了侧边栏的强制更新，_force 才会为 True
    raw_df, fails = load_data_v19(current_pool, benchmark_code, start_date, force_update)
    
    if fails: st.toast(f"缺失: {len(fails)}", icon="⚠️")
    
    if raw_df is None:
        st.error("❌ 基准数据获取失败")
    else:
        worms, dates, msg = calculate_rrg(raw_df, period_code, window, tail_len)
        
        if worms.empty:
            st.error(f"❌ 错误: {msg}")
        else:
            # === Plotly ===
            fig = go.Figure()
            def add_q(x0, x1, y0, y1, c):
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=y0, y1=y1, fillcolor=c, opacity=0.08, line_width=0, layer="below")
            add_q(90,100,100,110,"cyan"); add_q(100,110,100,110,"green")
            add_q(90,100,90,100,"red"); add_q(100,110,90,100,"yellow")
            fig.add_hline(y=100, line_color="#444"); fig.add_vline(x=100, line_color="#444")
            
            fig.add_annotation(x=105,y=105,text="领先",showarrow=False,font=dict(color="green",size=16))
            fig.add_annotation(x=95,y=105,text="改善",showarrow=False,font=dict(color="cyan",size=16))
            fig.add_annotation(x=95,y=95,text="落后",showarrow=False,font=dict(color="red",size=16))
            fig.add_annotation(x=105,y=95,text="转弱",showarrow=False,font=dict(color="yellow",size=16))
            fig.add_annotation(text="zf制作", xref="paper", yref="paper", x=0.99, y=0.99, showarrow=False, font=dict(size=30, color="rgba(255,255,255,0.1)"), align="right")

            last_d = dates[-1]
            init = worms[worms['Frame'] == last_d]
            
            for name in worms['Name'].unique():
                row = init[init['Name'] == name]
                x, y = (row.iloc[0]['X'], row.iloc[0]['Y']) if not row.empty else ([],[])
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=name, marker=dict(size=[4]*(len(x)-1)+[14], line=dict(width=1,color='white'))))
            
            frames = []
            for d in dates:
                fd = []
                frm = worms[worms['Frame'] == d]
                for name in worms['Name'].unique():
                    r = frm[frm['Name'] == name]
                    fd.append(go.Scatter(x=r.iloc[0]['X'], y=r.iloc[0]['Y']) if not r.empty else go.Scatter(x=[],y=[]))
                frames.append(go.Frame(data=fd, name=d))
            fig.frames = frames
            
# === 优化的图表布局与工具栏 ===
            fig.update_layout(
                title=f"RRG 轮动图 ({last_d})", 
                template="plotly_dark", 
                height=850, # 调高高度，无需全屏也能看清
                margin=dict(t=100), # 给顶部留出空间，防止按钮挡住标题
                xaxis=dict(range=[94,106], title="RS-Ratio (趋势)"), 
                yaxis=dict(range=[94,106], title="RS-Mom (动能)"),
                
                # 图例放在底部
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.15, 
                    xanchor="center",
                    x=0.5
                ),
                
                # 将播放按钮移至【左上角】安全区，彻底避开下方滑动条
                updatemenus=[dict(
                    type="buttons", direction="left",
                    buttons=[
                        dict(label="▶️ 播放", method="animate", args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True)]),
                        dict(label="⏸️ 暂停", method="animate", args=[[None], dict(mode="immediate")])
                    ],
                    pad={"r": 10, "t": 10}, 
                    showactive=True, 
                    x=0.0, xanchor="left", y=1.15, yanchor="bottom" # 移至图表左上方
                )],
                sliders=[dict(steps=[dict(method='animate', args=[[d], dict(mode='immediate')], label=d) for d in dates])]
            )
            
            # 精简自带工具栏，防止误触
            st.plotly_chart(
                fig, 
                use_container_width=True,
                config={
                    'displaylogo': False,
                    'scrollZoom': True,
                    'modeBarButtonsToRemove': ['autoScale2d', 'hoverCompareCartesian', 'hoverClosestCartesian', 'toggleSpikelines']
                }
            )
            
            # === 表格 (使用原生组件，永不报错) ===
            st.subheader(f"📊 详细数据 ({last_d})")
            final = init[['Name', 'P', 'X', 'Y']].copy()
            final['X'] = final['X'].apply(lambda x: x[-1] if x else 0)
            final['Y'] = final['Y'].apply(lambda y: y[-1] if y else 0)
            final.columns = ['名称', '最新价', '趋势(Ratio)', '动能(Mom)']
            
            st.dataframe(
                final.set_index('名称'),
                column_config={
                    "趋势(Ratio)": st.column_config.ProgressColumn("趋势强度", min_value=90, max_value=110, format="%.2f"),
                    "动能(Mom)": st.column_config.NumberColumn(format="%.2f"),
                    "最新价": st.column_config.NumberColumn(format="%.2f"),
                },
                use_container_width=True
            )