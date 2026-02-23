import streamlit as st
import akshare as ak
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(layout="wide", page_title="全球 RRG (V23.1 完美修复版)")
st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .stDeployButton, [data-testid="stToolbar"], footer {display:none;}
    [data-testid="stSidebar"] { min-width: 380px; }
    h1 { color: #00CC96; text-shadow: 2px 2px 4px #000000; }
    div.stButton > button { width: 100%; border-radius: 5px; }
    button[title="View fullscreen"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心数据库与弹药库
# ==========================================
DATA_DIR = "rrg_data_warehouse"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

A_SECTOR_CONFIG = {
    "煤炭": {"etf": "sh515220", "keyword": "煤炭"}, "钢铁": {"etf": "sh515210", "keyword": "钢铁"},
    "有色": {"etf": "sh512400", "keyword": "有色"}, "石油": {"etf": "sh561360", "keyword": "石油"},
    "电力": {"etf": "sh561560", "keyword": "电力"}, "化工": {"etf": "sh516020", "keyword": "化学"},
    "银行": {"etf": "sh512800", "keyword": "银行"}, "证券": {"etf": "sh512880", "keyword": "证券"},
    "保险": {"etf": "sh515050", "keyword": "保险"}, "房地产": {"etf": "sh512200", "keyword": "房地产"},
    "半导体": {"etf": "sh512480", "keyword": "半导体"}, "芯片": {"etf": "sz159995", "keyword": "半导体"},
    "光伏": {"etf": "sh515790", "keyword": "光伏"}, "新能车": {"etf": "sh515030", "keyword": "汽车整车"},
    "电池": {"etf": "sz159755", "keyword": "电池"}, "白酒": {"etf": "sh512690", "keyword": "酿酒"},
    "医药": {"etf": "sh512010", "keyword": "医药"}, "家电": {"etf": "sh561120", "keyword": "家电"},
    "游戏": {"etf": "sh516770", "keyword": "游戏"}, "养殖": {"etf": "sz159865", "keyword": "农牧"},
    "通信": {"etf": "sh515880", "keyword": "通信"}, "计算机": {"etf": "sz159998", "keyword": "计算机"}
}

# 11大标准GICS行业 + 专属基建
US_SECTOR_CONFIG = {
    "科技 (XLK)": {"etf": "XLK", "stocks": {"AAPL":"苹果", "MSFT":"微软", "NVDA":"英伟达", "AVGO":"博通", "ADBE":"Adobe", "CRM":"赛富时", "AMD":"AMD", "ACN":"埃森哲", "CSCO":"思科", "INTC":"英特尔"}},
    "医疗健康 (XLV)": {"etf": "XLV", "stocks": {"LLY":"礼来", "UNH":"联合健康", "JNJ":"强生", "MRK":"默沙东", "ABBV":"艾伯维", "TMO":"赛默飞", "ABT":"雅培", "PFE":"辉瑞", "DHR":"丹纳赫", "AMGN":"安进"}},
    "金融 (XLF)": {"etf": "XLF", "stocks": {"BRK-B":"伯克希尔", "JPM":"摩根大通", "V":"Visa", "MA":"万事达", "BAC":"美国银行", "WFC":"富国银行", "MS":"摩根士丹利", "GS":"高盛", "C":"花旗", "BLK":"贝莱德"}},
    "非必选消费 (XLY)": {"etf": "XLY", "stocks": {"AMZN":"亚马逊", "TSLA":"特斯拉", "HD":"家得宝", "MCD":"麦当劳", "NKE":"耐克", "SBUX":"星巴克", "LOW":"劳氏", "BKNG":"Booking", "TJX":"TJX公司", "TGT":"塔吉特"}},
    "工业 (XLI)": {"etf": "XLI", "stocks": {"GE":"通用电气", "CAT":"卡特彼勒", "RTX":"雷神", "BA":"波音", "UNP":"联合太平洋", "HON":"霍尼韦尔", "UPS":"UPS", "LMT":"洛马", "DE":"约翰迪尔", "MMM":"3M"}},
    "日常消费 (XLP)": {"etf": "XLP", "stocks": {"PG":"宝洁", "COST":"好市多", "WMT":"沃尔玛", "PEP":"百事", "KO":"可口可乐", "PM":"菲利普莫里斯", "MO":"奥驰亚", "EL":"雅诗兰黛", "CL":"高露洁", "KMB":"金佰利"}},
    "能源 (XLE)": {"etf": "XLE", "stocks": {"XOM":"埃克森美孚", "CVX":"雪佛龙", "COP":"康菲", "SLB":"斯伦贝谢", "EOG":"EOG能源", "MPC":"马拉松原油", "PXD":"先锋自然", "VLO":"瓦莱罗能源", "PSX":"菲利普斯66", "OXY":"西方石油"}},
    "公用事业 (XLU)": {"etf": "XLU", "stocks": {"NEE":"新纪元能源", "SO":"南方公司", "DUK":"杜克能源", "SRE":"桑普拉能源", "AEP":"美国电力", "D":"道明尼资源", "PCG":"太平洋燃气", "EXC":"艾斯能", "XEL":"新世纪能源", "ED":"联合爱迪生"}},
    "房地产 (XLRE)": {"etf": "XLRE", "stocks": {"PLD":"普洛斯", "AMT":"美国电塔", "EQIX":"易昆尼克斯", "CCI":"冠城国际", "PSA":"大众仓储", "O":"RealtyIncome", "SPG":"西蒙地产", "WELL":"Welltower", "DLR":"数字房地产", "CSGP":"CoStar"}},
    "材料 (XLB)": {"etf": "XLB", "stocks": {"LIN":"林德", "SHW":"宣伟", "FCX":"自由港", "ECL":"艺康", "APD":"空气化工", "NEM":"纽蒙特", "DOW":"陶氏", "NUE":"纽柯", "CTVA":"科迪华", "VMC":"火神材料"}},
    "通信服务 (XLC)": {"etf": "XLC", "stocks": {"GOOGL":"谷歌A", "META":"Meta", "NFLX":"奈飞", "TMUS":"T-Mobile", "DIS":"迪士尼", "VZ":"威瑞森", "CMCSA":"康卡斯特", "T":"AT&T", "CHTR":"特许通讯", "EA":"EA游戏"}},
    "本土基建异动 (AIRR)": {"etf": "AIRR", "stocks": {"STRL":"Sterling", "MTZ":"MasTec", "EME":"EMCOR", "FIX":"Comfort", "PRIM":"Primoris", "DY":"Dycom", "PWR":"Quanta", "URI":"UnitedRent", "PAVE":"PAVE基建", "XHB":"建筑商"}}
}

# ==========================================
# 3. 数据层与侧边栏
# ==========================================
@st.cache_data(ttl=3600)
def get_real_board_code(keyword):
    try:
        df = ak.stock_board_industry_name_em()
        target = df[df['板块名称'] == keyword]
        if target.empty: target = df[df['板块名称'].str.contains(keyword)]
        if not target.empty: return target.iloc[0]['板块名称'], target.iloc[0]['板块代码']
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

def fetch_net_us(code, period_code):
    try:
        if period_code in ['1h', '60m']: start = (datetime.datetime.now() - datetime.timedelta(days=700)).strftime("%Y-%m-%d")
        elif period_code in ['15m']: start = (datetime.datetime.now() - datetime.timedelta(days=50)).strftime("%Y-%m-%d")
        else: start = "2023-01-01"
        df = yf.Ticker(code).history(interval=period_code, start=start)
        if df.empty: return None
        df = df.reset_index()
        time_col = df.columns[0]
        df['date'] = df[time_col].dt.tz_localize(None)
        return df.set_index('date')[['Close']].rename(columns={'Close':'close'})
    except: return None

def fetch_net_a(code, start_date):
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

def get_data_smart(code, start_date, force_refresh, is_us, p_code):
    file_path = os.path.join(DATA_DIR, f"{code}_{p_code}.csv" if is_us else f"{code}.csv")
    if os.path.exists(file_path) and not force_refresh:
        if datetime.date.fromtimestamp(os.path.getmtime(file_path)) == datetime.date.today():
            try: return pd.read_csv(file_path, index_col=0, parse_dates=True)['close']
            except: pass
    df_new = fetch_net_us(code, p_code) if is_us else fetch_net_a(code, start_date)
    if df_new is not None and not df_new.empty:
        try: df_new.to_csv(file_path)
        except: pass
        return df_new['close']
    if os.path.exists(file_path):
        try: return pd.read_csv(file_path, index_col=0, parse_dates=True)['close']
        except: pass
    return None

@st.cache_data(ttl=3600)
def load_data_v23(pool, bench, start, _force, is_us, p_code):
    data, fails = {}, []
    status = st.empty()
    bar = st.progress(0)
    
    status.text(f"读取基准: {bench}...")
    b_s = get_data_smart(bench, start, _force, is_us, p_code)
    if b_s is None: return None, ["基准"]
    data['__BENCH__'] = b_s
    full_idx = b_s.index
    
    total = len(pool)
    for i, (k, v) in enumerate(pool.items()):
        status.text(f"读取数据 ({i+1}/{total}): {v}...")
        bar.progress((i+1)/total)
        s = get_data_smart(k, start, _force, is_us, p_code)
        if s is not None:
            s = s[~s.index.duplicated(keep='last')]
            data[v] = s.reindex(full_idx).ffill()
        else: fails.append(v)
            
    status.empty(); bar.empty()
    return pd.DataFrame(data), fails

with st.sidebar:
    st.title("🚀 全球 RRG (V23.1 完美修复版)")
    market = st.selectbox("🌍 市场环境", ["🇨🇳 A股 (动态抓取)", "🇺🇸 美股 (高频接入)"], index=0)
    is_us = "美股" in market
    
    st.divider()
    st.header("1️⃣ 视角选择")
    level = st.radio("模式", ["Level 1: 全行业 ETF 轮动", "Level 2: 板块内抓龙头"])
    
    if is_us:
        BENCHMARK_DICT = {"标普500 (SPY)": "SPY", "纳斯达克 (QQQ)": "QQQ", "罗素2000 (IWM)": "IWM"}
    else:
        BENCHMARK_DICT = {"沪深300 (机构)": "sh510300", "红利ETF (避险)": "sh510880", "中证2000 (游资)": "sh563300"}
        
    bench_choice = st.selectbox("🎯 参照系基准 (中心)", list(BENCHMARK_DICT.keys()) + ["自定义输入"])
    if bench_choice == "自定义输入": benchmark_code = st.text_input("代码", "SPY" if is_us else "sh510300").strip().upper()
    else: benchmark_code = BENCHMARK_DICT[bench_choice]
    st.caption(f"当前基准: {benchmark_code}")
    
    force_update = st.button("🔄 强制更新今日数据")
    current_pool = {}
    
    if is_us:
        if "Level 1" in level:
            current_pool = {v['etf']: k for k, v in US_SECTOR_CONFIG.items()}
            if benchmark_code in current_pool: del current_pool[benchmark_code]
        else:
            sector_key = st.selectbox("选择美股板块", list(US_SECTOR_CONFIG.keys()))
            cfg = US_SECTOR_CONFIG[sector_key]
            benchmark_code = cfg['etf'] 
            st.caption(f"板块基准自动切换为: {benchmark_code}")
            current_pool = cfg['stocks']
    else:
        # 🚨 就是这里！修复了之前忘改名字的 Bug，把 SECTOR_CONFIG 改成了 A_SECTOR_CONFIG
        if "Level 1" in level:
            current_pool = {v['etf']: k for k, v in A_SECTOR_CONFIG.items()}
            if benchmark_code in current_pool: del current_pool[benchmark_code]
        else:
            sector_key = st.selectbox("选择行业", list(A_SECTOR_CONFIG.keys()))
            cfg = A_SECTOR_CONFIG[sector_key]
            real_name, real_code = get_real_board_code(cfg['keyword'])
            if real_name:
                benchmark_code = cfg['etf'] 
                st.caption(f"板块: {real_name} | 基准: {benchmark_code}")
                top_n = st.slider("龙头数", 5, 50, 20)
                with st.spinner("获取名单..."): current_pool = get_constituents_safe(real_name, top_n)
            else: st.error("板块匹配失败")
            
    extra = st.text_input("➕ 搅局者 (代码,名称)", "")
    if extra:
        p = extra.split(',')
        current_pool[p[0].strip().upper() if is_us else p[0].strip()] = p[1].strip() if len(p)>1 else p[0].strip()

    st.divider()
    st.header("2️⃣ 参数 (引擎与尾气)")
    
    if is_us:
        period_name = st.radio("时间周期", ["日线 (1d)", "周线 (1wk)", "1小时 (1h)", "15分钟 (15m)"], index=0)
        period_code = period_name.split('(')[1].replace(')', '')
    else:
        period = st.radio("时间周期", ["日线", "周线"], index=0)
        period_code = 'W-FRI' if "周" in period else 'D'
        
    col1, col2 = st.columns(2)
    with col1: window = st.number_input("RS窗口 (计算引擎)", 5, 60, 14, help="影响数据计算周期")
    with col2: tail_len = st.number_input("拖尾 (视觉轨迹)", 1, 30, 8, help="影响画出多长的尾巴")

# ==========================================
# 4. 计算逻辑 (🌟 V23 核心：预平滑 Z-Score 模型)
# ==========================================
def calculate_rrg(df, period, window, tail):
    if period in ['D', '1d', '1h', '15m']: df_res = df
    else: df_res = df.resample('W-FRI').last()
    
    df_res = df_res.dropna(how='all')
    if len(df_res) < window + 5: return pd.DataFrame(), [], "数据长度不足"

    bench = df_res['__BENCH__']
    worm_data = []
    dates = df_res.index[window+10:]
    if len(dates) > 52: dates = dates[-52:]
    
    time_format = '%Y-%m-%d %H:%M' if period in ['1h', '15m'] else '%Y-%m-%d'
    str_dates = [d.strftime(time_format) for d in dates]
    
    for col in df_res.columns:
        if col == '__BENCH__': continue
        series = df_res[col]
        if series.notna().sum() < window + 5: continue
        
        # 1. 相对强弱计算
        rs = series / bench
        
        # 2. 预平滑
        rs_smooth = rs.ewm(span=5, adjust=False).mean()
        
        # 3. 真实的 Ratio
        rs_mean = rs_smooth.rolling(window).mean()
        rs_std = rs_smooth.rolling(window).std()
        ratio = 100 + ((rs_smooth - rs_mean) / rs_std) * 1.5
        
        # 4. 真实的 Momentum
        mom = 100 + ((ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std()) * 1.5
        
        # 5. 画图渲染层平滑
        ratio = ratio.ewm(span=3, adjust=False).mean()
        mom = mom.ewm(span=3, adjust=False).mean()
        
        temp = pd.DataFrame({'R': ratio, 'M': mom, 'P': series}, index=df_res.index)
        
        for d_str, dt_obj in zip(str_dates, dates):
            try:
                hist = temp.loc[:dt_obj].tail(tail + 1)
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
# 5. 主程序渲染
# ==========================================
if st.button("🚀 开始分析", type="primary"):
    start_date = "2021-01-01"
    raw_df, fails = load_data_v23(current_pool, benchmark_code, start_date, force_update, is_us, period_code)
    
    if fails: st.toast(f"缺失: {len(fails)}", icon="⚠️")
    
    if raw_df is None:
        st.error("❌ 基准数据获取失败！请检查网络。")
    else:
        worms, dates, msg = calculate_rrg(raw_df, period_code, window, tail_len)
        
        if worms.empty:
            st.error(f"❌ 错误: {msg}")
        else:
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

            last_d = dates[-1]
            init = worms[worms['Frame'] == last_d]
            
            for name in worms['Name'].unique():
                row = init[init['Name'] == name]
                x, y = (row.iloc[0]['X'], row.iloc[0]['Y']) if not row.empty else ([],[])
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='lines+markers', name=name, 
                    marker=dict(size=[4]*(len(x)-1)+[14], line=dict(width=1,color='white')),
                    line=dict(width=2, shape='spline', smoothing=1.3)
                ))
            
            frames = []
            for d in dates:
                fd = []
                frm = worms[worms['Frame'] == d]
                for name in worms['Name'].unique():
                    r = frm[frm['Name'] == name]
                    fd.append(go.Scatter(x=r.iloc[0]['X'], y=r.iloc[0]['Y']) if not r.empty else go.Scatter(x=[],y=[]))
                frames.append(go.Frame(data=fd, name=d))
            fig.frames = frames
            
            fig.update_layout(
                title=f"RRG 轮动图 ({last_d})", 
                template="plotly_dark", 
                height=850, margin=dict(t=100),
                xaxis=dict(range=[94,106], title="RS-Ratio (趋势)"), 
                yaxis=dict(range=[94,106], title="RS-Mom (动能)"),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
                updatemenus=[dict(
                    type="buttons", direction="left",
                    buttons=[
                        dict(label="▶️ 播放", method="animate", args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True)]),
                        dict(label="⏸️ 暂停", method="animate", args=[[None], dict(mode="immediate")])
                    ],
                    pad={"r": 10, "t": 10}, showactive=True, x=0.0, xanchor="left", y=1.15, yanchor="bottom"
                )],
                sliders=[dict(steps=[dict(method='animate', args=[[d], dict(mode='immediate')], label=d) for d in dates])]
            )
            
            st.plotly_chart(fig, use_container_width=True, config={'displaylogo': False, 'scrollZoom': True, 'modeBarButtonsToRemove': ['autoScale2d']})
            
            st.divider()
            st.subheader("🚨 雷达监控区：主升浪与底部抢筹发现器")
            
            burst_list = []
            for _, row in init.iterrows():
                x_t, y_t = row['X'], row['Y']
                if len(x_t) >= 2:
                    dx, dy = x_t[-1] - x_t[-2], y_t[-1] - y_t[-2]
                    cx, cy = x_t[-1], y_t[-1]
                    sig = None
                    if cx > 100 and cy > 100 and dx > 0.1 and dy > 0.1: sig = "🔥 强者恒强 (右侧主升浪)"
                    elif dy > 0.8 and abs(dx) < 0.5 and cx < 101: sig = "🚀 底部抢筹 (垂直爆发)"
                    if sig: burst_list.append({'标的': row['Name'], '信号类型': sig, '最新价': row['P'], '动能(ΔY)': dy, '趋势(ΔX)': dx, '当前X': cx, '当前Y': cy})
            
            if burst_list:
                b_df = pd.DataFrame(burst_list).sort_values(by=['信号类型', '动能(ΔY)'], ascending=[True, False])
                st.success(f"雷达发现 {len(b_df)} 只异动标的 👇")
                st.dataframe(b_df.set_index('标的'), use_container_width=True, column_config={"动能(ΔY)": st.column_config.NumberColumn(format="%+.2f"), "趋势(ΔX)": st.column_config.NumberColumn(format="%+.2f")})
            else: st.info("🟢 当前扫描无异动标的。")