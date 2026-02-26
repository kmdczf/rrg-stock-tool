import streamlit as st
import akshare as ak
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import datetime
import os
import json
import warnings
import requests
import re

warnings.filterwarnings('ignore')

# ==========================================
# 1. 云端/本地 环境适配 (防崩溃单线版)
# ==========================================
os.environ['NO_PROXY'] = '*'

st.set_page_config(layout="wide", page_title="全球 RRG (V62 极致定型版)")
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

st.title("🚀 全球 RRG 极速分析系统 (V62 完美闭环版)")

# ==========================================
# 2. 核心本地数据库与 31 大 ETF 兵器库
# ==========================================
DATA_DIR = "rrg_data_warehouse"
if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)

# 满血 31 大核心 ETF 池
A_ETF_CONFIG = {
    "sh515220": "煤炭ETF", "sh515210": "钢铁ETF", "sh512400": "有色ETF",
    "sh561360": "石油ETF", "sh561560": "电力ETF", "sh516020": "化工ETF",
    "sh512800": "银行ETF", "sh512880": "证券ETF", "sh515050": "保险ETF",
    "sh512200": "房地产ETF", "sh512480": "半导体ETF", "sh515790": "光伏ETF",
    "sh515030": "新能车ETF", "sz159755": "电池ETF", "sh512690": "酒ETF",
    "sh512010": "医药ETF", "sh512170": "医疗ETF", "sh561120": "家电ETF",
    "sh516770": "游戏ETF", "sh512980": "传媒ETF", "sz159865": "养殖ETF",
    "sh515880": "通信ETF", "sz159998": "计算机ETF", "sh512660": "军工ETF",
    "sh516530": "物流ETF", "sh562510": "旅游ETF", "sh512580": "环保ETF",
    "sh512530": "建材ETF", "sh516100": "基建ETF", "sz159886": "机械ETF",
    "sh510150": "消费ETF"
}

# 🌟 V62 核心闭环：将新浪细分行业反向映射到 31 大 ETF
ETF_KEYWORD_MAP = {
    "煤炭": "sh515220", "钢铁": "sh515210", "有色": "sh512400", "金属": "sh512400",
    "石油": "sh561360", "电力": "sh561560", "发电": "sh515790", "化工": "sh516020", 
    "化纤": "sh516020", "塑料": "sh516020", "橡胶": "sh516020",
    "银行": "sh512800", "金融": "sh512800", "证券": "sh512880", "保险": "sh515050", 
    "房地": "sh512200", "园区": "sh512200", 
    "半导": "sh512480", "元器件": "sh512480", "光伏": "sh515790", 
    "汽车": "sh515030", "电池": "sz159755", "酿酒": "sh512690", "酒": "sh512690", 
    "医药": "sh512010", "制药": "sh512010", "生物": "sh512010", "医疗": "sh512170", 
    "家电": "sh561120", "电器": "sh561120", "游戏": "sh516770", "传媒": "sh512980",
    "娱乐": "sh516770", "农牧": "sz159865", "养殖": "sz159865", "农业": "sz159865",
    "通信": "sh515880", "计算": "sz159998", "软件": "sz159998", "互联网": "sz159998",
    "军工": "sh512660", "船舶": "sh512660", "航天": "sh512660", "飞机": "sh512660",
    # 细分板块兜底
    "交运": "sh516530", "交通": "sh516530", "物流": "sh516530", "公路": "sh516530", 
    "桥梁": "sh516530", "机场": "sh516530", "港口": "sh516530", "航空": "sh516530",
    "旅游": "sh562510", "酒店": "sh562510", "餐饮": "sh562510",
    "环保": "sh512580", "供水": "sh512580", "水务": "sh512580", "供气": "sh561560",
    "建材": "sh512530", "水泥": "sh512530", "玻璃": "sh512530",
    "建筑": "sh516100", "工程": "sh516100", "机械": "sz159886", "机床": "sz159886", 
    "仪器": "sz159886", "消费": "sh510150", "百货": "sh510150", "商贸": "sh510150", 
    "商业": "sh510150", "纺织": "sh510150", "服装": "sh510150", "轻工": "sh510150", 
    "造纸": "sh510150", "家具": "sh510150", "包装": "sh510150"
}

ETF_TO_KEYWORDS = {}
for kw, etf in ETF_KEYWORD_MAP.items():
    ETF_TO_KEYWORDS.setdefault(etf, []).append(kw)

# 满血复原：美股 12 大板块 100% 找回
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
    "本土基建 (AIRR)": {"etf": "AIRR", "stocks": {"STRL":"Sterling", "MTZ":"MasTec", "EME":"EMCOR", "FIX":"Comfort", "PRIM":"Primoris", "DY":"Dycom", "PWR":"Quanta", "URI":"UnitedRent", "PAVE":"PAVE基建", "XHB":"建筑商"}}
}

# ==========================================
# 3. 极速本地化抓取引擎
# ==========================================
@st.cache_data(ttl=86400)
def get_sina_board_mapping():
    local_file = os.path.join(DATA_DIR, "sina_boards.json")
    try:
        df = ak.stock_sector_spot(indicator="新浪行业")
        if not df.empty:
            mapping = dict(zip(df['板块'], df['label']))
            with open(local_file, 'w', encoding='utf-8') as f: json.dump(mapping, f, ensure_ascii=False)
            return mapping
    except: pass
    if os.path.exists(local_file):
        with open(local_file, 'r', encoding='utf-8') as f: return json.load(f)
    return {"半导体": "new_bdt", "交通运输": "new_jtys"}

@st.cache_data(ttl=3600)
def get_constituents_safe(sina_label, limit):
    local_file = os.path.join(DATA_DIR, f"cons_{sina_label}.json")
    try:
        url = f"http://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData?page=1&num={limit}&sort=amount&asc=0&node={sina_label}"
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
        text = re.sub(r'([{,])([a-zA-Z0-9_]+):', r'\1"\2":', resp.text)
        data = json.loads(text)
        df = pd.DataFrame(data)
        
        if not df.empty:
            res = {}
            for _, row in df.iterrows():
                code = str(row['symbol']).strip() 
                name = str(row['name']).strip()
                clean_code = code.replace('sh', '').replace('sz', '')
                fc = f"sh{clean_code}" if clean_code.startswith(('6','9','5')) else f"sz{clean_code}"
                res[fc] = name
            with open(local_file, 'w', encoding='utf-8') as f: json.dump(res, f, ensure_ascii=False)
            return res
    except: pass
    if os.path.exists(local_file):
        with open(local_file, 'r', encoding='utf-8') as f: return json.load(f)
    return {}

# ==========================================
# 4. 单线安全 K 线下载与深度缓存
# ==========================================
def fetch_kline_safe(code, start_date, is_us, p_code):
    if is_us:
        try:
            if p_code in ['1h', '60m']: start = (datetime.datetime.now() - datetime.timedelta(days=700)).strftime("%Y-%m-%d")
            elif p_code in ['15m']: start = (datetime.datetime.now() - datetime.timedelta(days=50)).strftime("%Y-%m-%d")
            else: start = "2023-01-01"
            df = yf.Ticker(code).history(interval=p_code, start=start)
            if not df.empty:
                df = df.reset_index()
                df['date'] = pd.to_datetime(df[df.columns[0]]).dt.tz_localize(None)
                return df.set_index('date')[['Close']].rename(columns={'Close':'close'})
        except: return None
    else:
        if any(x in code for x in ['51','15','56']):
            try:
                df = ak.fund_etf_hist_sina(symbol=code)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    return df.set_index('date')[['close']]
            except: pass
        try:
            url = f"http://money.finance.sina.com.cn/quotes_service/api/json_v2.php/CN_MarketData.getKLineData?symbol={code}&scale=240&ma=no&datalen=800"
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
            text = re.sub(r'([{,])([a-zA-Z0-9_]+):', r'\1"\2":', resp.text)
            df = pd.DataFrame(json.loads(text))
            if not df.empty:
                df['date'] = pd.to_datetime(df['day'])
                df = df[df['date'] >= pd.to_datetime(start_date)]
                return df.set_index('date')[['close']].astype(float)
        except: pass
        try:
            yf_code = code.replace("sh", "") + ".SS" if "sh" in code else code.replace("sz", "") + ".SZ"
            df = yf.Ticker(yf_code).history(start=start_date)
            if not df.empty:
                df = df.reset_index()
                df['date'] = pd.to_datetime(df[df.columns[0]]).dt.tz_localize(None)
                return df.set_index('date')[['Close']].rename(columns={'Close':'close'})
        except: pass
    return None

def get_data_smart(code, start_date, force_refresh, is_us, p_code):
    file_path = os.path.join(DATA_DIR, f"{code}_{p_code}.csv" if is_us else f"{code}.csv")
    if os.path.exists(file_path) and not force_refresh:
        mtime = datetime.date.fromtimestamp(os.path.getmtime(file_path))
        if mtime == datetime.date.today():
            try: return pd.read_csv(file_path, index_col=0, parse_dates=True)['close']
            except: pass
            
    df_new = fetch_kline_safe(code, start_date, is_us, p_code)
    if df_new is not None and not df_new.empty:
        try: df_new.to_csv(file_path) 
        except: pass
        return df_new['close']
        
    if os.path.exists(file_path): 
        try: return pd.read_csv(file_path, index_col=0, parse_dates=True)['close']
        except: pass
    return None

# ==========================================
# 🌟 核心黑科技：自建等权板块走势引擎
# ==========================================
@st.cache_data(ttl=3600)
def load_data_stable(pool, board_mapping, bench, start, _force, is_us, p_code):
    data, fails = {}, []
    status = st.empty(); bar = st.progress(0)
    
    status.text(f"读取基准: {bench}...")
    b_s = get_data_smart(bench, start, _force, is_us, p_code)
    if b_s is None: return None, ["基准"]
    data['__BENCH__'] = b_s
    full_idx = b_s.index
    
    # 第一步：下载并提取所有正股 (避开以 BK_ 开头的伪代码)
    normal_items = {k: v for k, v in pool.items() if not str(k).startswith("BK_")}
    bk_items = {k: v for k, v in pool.items() if str(k).startswith("BK_")}
    
    total = len(normal_items)
    for i, (k, v) in enumerate(normal_items.items()):
        status.text(f"安全同步数据 (固化后次日秒开) ({i+1}/{total}): {v}...")
        bar.progress((i+1)/total if total > 0 else 1.0)
        s = get_data_smart(k, start, _force, is_us, p_code)
        if s is not None:
            s = s[~s.index.duplicated(keep='last')]
            data[v] = s.reindex(full_idx).ffill()
        else:
            fails.append(v)
            
    # 第二步：本地合成各细分板块的等权走势 (完全不联网！)
    if bk_items:
        status.text("正在本地合成等权板块走势...")
        for k, v in bk_items.items():
            board_name = k[3:] # 剥离 "BK_"
            codes = board_mapping.get(board_name, [])
            valid_series = []
            
            for c in codes:
                name = pool.get(c)
                if name and name in data:
                    s = data[name]
                    first_valid = s.first_valid_index()
                    # 归一化为 100 进行等权计算
                    if first_valid is not None and s.loc[first_valid] != 0:
                        norm_s = s / s.loc[first_valid] * 100
                        valid_series.append(norm_s)
            
            if valid_series:
                df_board = pd.concat(valid_series, axis=1)
                data[v] = df_board.mean(axis=1) # 合成出板块的走势线！
            else:
                fails.append(v)

    status.empty(); bar.empty()
    return pd.DataFrame(data), fails

# ==========================================
# 5. 侧边栏交互 (无缝连通底层细分)
# ==========================================
board_mapping = {} # 用于向引擎传递“哪个板块包含哪些股票”的字典

with st.sidebar:
    st.header("1️⃣ 视角选择")
    market = st.selectbox("🌍 市场环境", ["🇨🇳 A股 (核心ETF透视底层)", "🇺🇸 美股 (网络直连)"], index=0)
    is_us = "美股" in market
    level = st.radio("模式", ["Level 1: 核心 ETF 轮动", "Level 2: 宏观赛道透视底层细分"])
    
    if is_us: 
        st.info("💡 提示: 美股数据首次加载需排队，已开启本地缓存机制！")
        BENCHMARK_DICT = {"标普500 (SPY)": "SPY", "纳斯达克 (QQQ)": "QQQ", "罗素2000 (IWM)": "IWM"}
    else: 
        st.info("🛡️ 自动防封锁: 采用单线安全连接+本地自建指数合成。")
        BENCHMARK_DICT = {"沪深300 (机构)": "sh510300", "红利ETF (避险)": "sh510880", "中证2000 (游资)": "sh563300"}
        
    bench_choice = st.selectbox("🎯 参照系基准", list(BENCHMARK_DICT.keys()) + ["自定义"])
    if bench_choice == "自定义": benchmark_code = st.text_input("代码", "SPY" if is_us else "sh510300").strip().upper()
    else: benchmark_code = BENCHMARK_DICT[bench_choice]
        
    st.caption(f"当前生效基准: {benchmark_code}")
    current_pool = {}
    
    if is_us:
        if "Level 1" in level:
            current_pool = {v['etf']: k for k, v in US_SECTOR_CONFIG.items()}
            if benchmark_code in current_pool: del current_pool[benchmark_code]
        else:
            sector_key = st.selectbox("选择美股板块", list(US_SECTOR_CONFIG.keys()))
            benchmark_code = US_SECTOR_CONFIG[sector_key]['etf'] 
            st.caption(f"自动切换基准: {benchmark_code}")
            current_pool = US_SECTOR_CONFIG[sector_key]['stocks']
    else:
        if "Level 1" in level:
            current_pool = A_ETF_CONFIG.copy()
            if benchmark_code in current_pool: del current_pool[benchmark_code]
        else:
            # 🌟 V62 终极闭环：选赛道 -> 找细分 -> 删主ETF -> 合成细分走势
            etf_options = {f"{name} ({code})": code for code, name in A_ETF_CONFIG.items()}
            selected_label = st.selectbox("选择宏观赛道 (自动透视底层相关细分行业)", list(etf_options.keys()))
            selected_etf_code = etf_options[selected_label]
            
            # 设置该宏观 ETF 为轮动中心系
            benchmark_code = selected_etf_code
            st.caption(f"🎯 中心基准已锁定为: {selected_label}")
            
            keywords = ETF_TO_KEYWORDS.get(selected_etf_code, [A_ETF_CONFIG[selected_etf_code].replace("ETF", "")])
            sina_mapping = get_sina_board_mapping()
            matched_boards = [board for board in sina_mapping.keys() if any(kw in board for kw in keywords)]
            
            st.caption(f"🔗 已穿透抓取新浪底层细分: {', '.join(matched_boards) if matched_boards else '宽基综合提取'}")
            
            top_n = st.slider("各细分板块截取前 N 只龙头", 5, 50, 15)
            
            with st.spinner("正在提取并准备本地合成算法..."):
                # 注意：这里我们明确删除了主 ETF 的走势，转而为每个细分板块生成一条趋势线
                for board in matched_boards:
                    label = sina_mapping[board]
                    board_stocks = get_constituents_safe(label, top_n)
                    if board_stocks:
                        # 加入底层龙头个股
                        current_pool.update(board_stocks)
                        # 记录此板块包含哪些个股代码，供引擎本地合成指数
                        board_mapping[board] = list(board_stocks.keys())
                        # 下达合成指令：加上“交运、公路”等板块自身的走势
                        current_pool[f"BK_{board}"] = f"🌟 {board} (等权走势)"
                
                if not current_pool:
                    st.error("🚨 提取失败或无对应细分，请检查网络。")
            
    extra = st.text_input("➕ 搅局者 (代码,名称)", "")
    if extra:
        p = extra.split(',')
        if is_us: current_pool[p[0].strip().upper()] = p[1].strip() if len(p)>1 else p[0].strip()
        else: current_pool[p[0].strip()] = p[1].strip() if len(p)>1 else p[0].strip()

    st.divider()
    st.header("2️⃣ 参数 (引擎与尾气)")
    if is_us:
        period_name = st.radio("时间周期", ["日线 (1d)", "周线 (1wk)", "1小时 (1h)", "15分钟 (15m)"], index=0)
        period_code = period_name.split('(')[1].replace(')', '')
    else:
        period = st.radio("时间周期", ["日线", "周线"], index=0)
        period_code = 'W-FRI' if "周" in period else 'D'
        
    col1, col2 = st.columns(2)
    with col1: window = st.number_input("RS窗口", 5, 60, 14)
    with col2: tail_len = st.number_input("拖尾长度", 1, 30, 8)

    st.divider()
    force_update = st.button("🔄 强制穿透刷新今日数据")

# ==========================================
# 6. 计算逻辑 (还原黄金平滑算法)
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
        
        rs = series / bench
        
        rs_smooth = rs.ewm(span=5, adjust=False).mean()
        rs_mean = rs_smooth.rolling(window).mean()
        rs_std = rs_smooth.rolling(window).std()
        
        ratio = 100 + ((rs_smooth - rs_mean) / rs_std) * 1.5
        mom = 100 + ((ratio - ratio.rolling(window).mean()) / ratio.rolling(window).std()) * 1.5
        
        ratio = ratio.ewm(span=3, adjust=False).mean()
        mom = mom.ewm(span=3, adjust=False).mean()
        
        temp = pd.DataFrame({'R': ratio, 'M': mom, 'P': series}, index=df_res.index)
        
        for d_str, dt_obj in zip(str_dates, dates):
            try:
                hist = temp.loc[:dt_obj].tail(tail + 1)
                if len(hist) > 0 and not np.isnan(hist.iloc[-1]['R']):
                    worm_data.append({'Frame': d_str, 'Name': col, 'X': hist['R'].tolist(), 'Y': hist['M'].tolist(), 'P': hist['P'].iloc[-1]})
            except: pass
            
    return pd.DataFrame(worm_data), str_dates, "OK"

# ==========================================
# 7. 主程序渲染
# ==========================================
if st.button("🚀 开始分析", type="primary"):
    start_date = "2021-01-01"
    
    # 将 board_mapping 传入，供底层合成指数使用
    raw_df, fails = load_data_stable(current_pool, board_mapping, benchmark_code, start_date, force_update, is_us, period_code)
    
    if fails: st.toast(f"已自动过滤 {len(fails)} 只停牌或无数据资产", icon="✅")
    
    if raw_df is None:
        st.error("❌ 基准数据获取失败！")
    elif not current_pool:
        st.error("❌ 股票池为空！")
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
            
            # --- 🌟 完美居中的艺术 Logo ---
            logo_x, logo_y = 0.5, 0.96
            fig.add_annotation(
                text="◯", x=logo_x, y=logo_y, xref="paper", yref="paper",
                xanchor="center", yanchor="middle",
                showarrow=False, font=dict(size=120, color="rgba(0, 204, 150, 0.15)")
            )
            fig.add_annotation(
                text="<span style='font-family: \"Arial Black\", Impact, sans-serif; font-style: italic; letter-spacing: 2px;'>ZF</span>", 
                x=logo_x, y=logo_y, xref="paper", yref="paper",
                xanchor="center", yanchor="middle",
                showarrow=False, font=dict(size=45, color="rgba(0, 204, 150, 0.25)")
            )

            fig.add_annotation(x=105,y=105,text="领先",showarrow=False,font=dict(color="green",size=16))
            fig.add_annotation(x=95,y=105,text="改善",showarrow=False,font=dict(color="cyan",size=16))
            fig.add_annotation(x=95,y=95,text="落后",showarrow=False,font=dict(color="red",size=16))
            fig.add_annotation(x=105,y=95,text="转弱",showarrow=False,font=dict(color="yellow",size=16))

            last_d = dates[-1]
            init = worms[worms['Frame'] == last_d]
            
            for name in worms['Name'].unique():
                row = init[init['Name'] == name]
                x, y = (row.iloc[0]['X'], row.iloc[0]['Y']) if not row.empty else ([],[])
                
                # 特殊高亮处理：对于自建的细分板块走势，加粗加亮
                is_sector_line = "🌟" in name
                line_width = 4 if is_sector_line else 2
                marker_size = [6]*(len(x)-1)+[18] if is_sector_line else [4]*(len(x)-1)+[14]
                
                fig.add_trace(go.Scatter(
                    x=x, y=y, mode='lines+markers', name=name, 
                    marker=dict(size=marker_size, line=dict(width=1,color='white')),
                    line=dict(width=line_width, shape='spline', smoothing=1.3)
                ))
            
            frames = []
            for d in dates:
                fd = []
                frm = worms[worms['Frame'] == d]
                for name in worms['Name'].unique():
                    r = frm[frm['Name'] == name]
                    x_fd = r.iloc[0]['X'] if not r.empty else []
                    y_fd = r.iloc[0]['Y'] if not r.empty else []
                    fd.append(go.Scatter(x=x_fd, y=y_fd))
                frames.append(go.Frame(data=fd, name=d))
            fig.frames = frames
            
            btn_play = dict(label="▶️ 播放", method="animate", args=[None, dict(frame=dict(duration=150, redraw=True), fromcurrent=True)])
            btn_pause = dict(label="⏸️ 暂停", method="animate", args=[[None], dict(mode="immediate")])
            menu_dict = dict(type="buttons", direction="left", buttons=[btn_play, btn_pause], pad={"r": 10, "t": 10}, showactive=True, x=0.0, xanchor="left", y=1.15, yanchor="bottom")
            
            fig.update_layout(
                title=f"RRG 轮动图 ({last_d})", template="plotly_dark", 
                height=880, 
                margin=dict(t=100, b=180),
                xaxis=dict(range=[94,106], title="RS-Ratio (趋势)"), 
                yaxis=dict(range=[94,106], title="RS-Mom (动能)"),
                legend=dict(orientation="h", yanchor="top", y=-0.28, xanchor="center", x=0.5),
                updatemenus=[menu_dict], sliders=[dict(steps=[dict(method='animate', args=[[d], dict(mode='immediate')], label=d) for d in dates])]
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
                    if sig: burst_list.append({'标的': row['Name'], '信号': sig, '最新价': row['P'], '动能ΔY': dy, '趋势ΔX': dx})
            
            if burst_list:
                b_df = pd.DataFrame(burst_list).sort_values(by=['信号', '动能ΔY'], ascending=[True, False])
                st.success(f"发现 {len(b_df)} 只异动标的 👇")
                col_cfg = {"动能ΔY": st.column_config.NumberColumn(format="%+.2f"), "趋势ΔX": st.column_config.NumberColumn(format="%+.2f")}
                st.dataframe(b_df.set_index('标的'), use_container_width=True, column_config=col_cfg)
            else: st.info("🟢 当前扫描无异动标的。")