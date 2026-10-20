import numpy as np
import pandas as pd
import streamlit as st
from FinMind.data import DataLoader

st.set_page_config(page_title="台股訊號查詢", page_icon="📈", layout="wide")

# ==================== 取得 FinMind Token（可選）====================
def _secret_token():
    try:
        return st.secrets.get("FINMIND_TOKEN", "")
    except Exception:
        return ""


# ==================== 營收檢核（月份對齊）====================
def check_revenue_accurate(rev_df):
    """近 3 個月營收合計 vs 去年同期 3 個月合計。
    以「年+月」對齊，不靠 index，避免資料缺月造成錯位。
    回傳 (是否通過, 說明文字)。"""
    if rev_df is None or rev_df.empty or len(rev_df) < 12:
        return False, "營收資料不足"

    rev_df = rev_df.copy()
    rev_df["revenue"] = rev_df["revenue"].astype(float)

    # 優先使用 FinMind 的 revenue_year / revenue_month（營收所屬月份）；
    # 沒有這兩欄才退回用 date。
    if {"revenue_year", "revenue_month"}.issubset(rev_df.columns):
        rev_df["_y"] = rev_df["revenue_year"].astype(int)
        rev_df["_m"] = rev_df["revenue_month"].astype(int)
    else:
        d = pd.to_datetime(rev_df["date"])
        rev_df["_y"], rev_df["_m"] = d.dt.year, d.dt.month

    rev_df = (
        rev_df.sort_values(["_y", "_m"])
        .drop_duplicates(["_y", "_m"], keep="last")
        .reset_index(drop=True)
    )
    lookup = {(int(r._y), int(r._m)): r.revenue for r in rev_df.itertuples()}

    latest_y, latest_m = int(rev_df.iloc[-1]["_y"]), int(rev_df.iloc[-1]["_m"])
    recent_periods = []
    for i in range(3):
        m, yr = latest_m - i, latest_y
        if m <= 0:
            m += 12
            yr -= 1
        recent_periods.append((yr, m))
    last_year_periods = [(yr - 1, m) for (yr, m) in recent_periods]

    for yr, m in recent_periods:
        if (yr, m) not in lookup:
            return False, f"缺少 {yr}-{m:02d} 營收資料"
    for yr, m in last_year_periods:
        if (yr, m) not in lookup:
            return False, f"缺少去年同期 {yr}-{m:02d} 營收資料"

    recent_sum = sum(lookup[p] for p in recent_periods)
    last_year_sum = sum(lookup[p] for p in last_year_periods)
    return bool(recent_sum > last_year_sum), (
        f"近3月合計 {recent_sum:,.0f} vs 去年同期 {last_year_sum:,.0f}"
    )


# ==================== 資料抓取（含快取）====================
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_fundamental(tk: str, token: str) -> dict:
    """單一股票基本面：營收 / ROE / PE"""
    dl = DataLoader()
    if token:
        dl.login_by_token(api_token=token)
    start_f_date = "2023-01-01"
    out = {"rev_pass": False, "rev_note": "", "roe_pass": False, "roe_val": 0.0, "pe_val": np.nan}
    try:
        rev_df = dl.taiwan_stock_month_revenue(stock_id=tk, start_date=start_f_date)
        out["rev_pass"], out["rev_note"] = check_revenue_accurate(rev_df)

        roe_df = dl.taiwan_stock_financial_statement(stock_id=tk, start_date=start_f_date)
        if not roe_df.empty:
            roe_data = roe_df[roe_df["type"] == "ROE"]
            if not roe_data.empty:
                out["roe_val"] = float(roe_data.iloc[-1]["value"])
                out["roe_pass"] = out["roe_val"] > 0

        per_df = dl.taiwan_stock_per_pbr(stock_id=tk, start_date="2024-01-01")
        if not per_df.empty:
            valid_pe = per_df[per_df["PER"] > 0]
            if not valid_pe.empty:
                out["pe_val"] = float(valid_pe.iloc[-1]["PER"])
    except Exception:
        pass
    return out


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_prices(tk: str, token: str):
    dl = DataLoader()
    if token:
        dl.login_by_token(api_token=token)
    df = dl.taiwan_stock_daily(stock_id=tk, start_date="2023-01-01")
    inst = dl.taiwan_stock_institutional_investors(stock_id=tk, start_date="2023-01-01")
    return df, inst


# ==================== 指標與訊號計算 ====================
def compute(target, peers, token, P):
    # ---- 基本面 ----
    fund = {tk: fetch_fundamental(tk, token) for tk in peers}
    valid_pes = [v["pe_val"] for v in fund.values() if not np.isnan(v["pe_val"])]
    pe_threshold = np.percentile(valid_pes, 30) if valid_pes else np.nan
    f = fund[target]
    pe_pass = (not np.isnan(f["pe_val"])) and (not np.isnan(pe_threshold)) and f["pe_val"] <= pe_threshold

    fundamental_score, fundamental_reasons = 0, []
    if f["rev_pass"]:
        fundamental_score += 2
        fundamental_reasons.append(f"近3月營收高於去年同期(+2分)：{f['rev_note']}")
    if f["roe_pass"]:
        fundamental_score += 2
        fundamental_reasons.append(f"最新ROE為正({f['roe_val']:.2f}%)(+2分)")
    if pe_pass:
        fundamental_score += 2
        fundamental_reasons.append(f"PE在族群最低30%內({f['pe_val']:.1f})(+2分)")

    # ---- 股價 ----
    raw, inst = fetch_prices(target, token)
    if raw is None or raw.empty:
        raise ValueError("查無股價資料，請確認股票代號。")
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    df = df[["open", "max", "min", "close", "Trading_Volume"]].copy()
    df.columns = ["Open", "High", "Low", "Close", "Volume"]

    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA60"] = df["Close"].rolling(60).mean()
    df["MA200"] = df["Close"].rolling(200).mean()

    low9 = df["Low"].rolling(9).min()
    high9 = df["High"].rolling(9).max()
    rsv = (df["Close"] - low9) / (high9 - low9) * 100
    df["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    df["D"] = df["K"].ewm(alpha=1 / 3, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - 100 / (1 + rs)

    df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
    df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = df["EMA12"] - df["EMA26"]
    df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["Histogram"] = df["MACD"] - df["Signal"]
    df["MACD_Buy"] = (df["MACD"] > df["Signal"]) & (df["MACD"].shift(1) <= df["Signal"].shift(1))
    df["MACD_Sell"] = (df["MACD"] < df["Signal"]) & (df["MACD"].shift(1) >= df["Signal"].shift(1))

    # 高潮走勢指標
    df["ROC"] = df["Close"] / df["Close"].shift(P["roc_days"]) - 1
    df["Vol_MA20"] = df["Volume"].rolling(20).mean()
    df["Vol_Ratio"] = df["Volume"] / df["Vol_MA20"]

    # 三大法人
    inst = inst.copy()
    inst["date"] = pd.to_datetime(inst["date"])
    inst["net"] = inst["buy"] - inst["sell"]
    inst_daily = inst.groupby("date")["net"].sum().to_frame("Total_Net")
    inst_daily["Total_Net"] = inst_daily["Total_Net"] / 1000
    df = df.join(inst_daily, how="left")
    df["Total_Net"] = df["Total_Net"].fillna(0)
    df["Net_3d"] = df["Total_Net"].rolling(3).sum()

    df = df.loc["2024-01-01":]
    if len(df) < 3:
        raise ValueError("資料筆數不足。")

    t, y = df.iloc[-1], df.iloc[-2]
    vol_ma10 = df["Volume"].rolling(10).mean().iloc[-1]
    avg_vol_10_lots = vol_ma10 / 1000

    # ---- 高潮走勢 ----
    roc_hit = (not np.isnan(t["ROC"])) and t["ROC"] > P["roc_th"]
    vol_hit = (not np.isnan(t["Vol_Ratio"])) and t["Vol_Ratio"] > P["vr_th"]
    climax_hit = roc_hit and vol_hit

    # ---- 買進過濾器 ----
    buy_blocked, block_reasons, soft_notes = False, [], []
    if avg_vol_10_lots < P["min_vol"]:
        buy_blocked = True
        block_reasons.append(f"流動性不足(10日均量 {avg_vol_10_lots:.0f}張 < {P['min_vol']}張)")
    bias_60 = (t["Close"] - t["MA60"]) / t["MA60"] * 100
    if not np.isnan(bias_60) and bias_60 > P["bias_limit"]:
        buy_blocked = True
        block_reasons.append(f"季線乖離過大({bias_60:.1f}% > {P['bias_limit']}%)，慎防追高")
    if climax_hit:
        buy_blocked = True
        block_reasons.append("高潮走勢（急漲+爆量），不追價")
    below_ma200 = (not np.isnan(t["MA200"])) and t["Close"] < t["MA200"]
    if below_ma200:
        soft_notes.append(f"收盤在年線下方（買進分數 -{P['ma200_penalty']}）")

    # ---- 買進得分 ----
    kd_score, kd_reason = 0, []
    if (y["K"] <= y["D"]) and (t["K"] > t["D"]) and t["D"] < 35:
        kd_score += 3; kd_reason.append("KD金叉（低檔）")
    if t["RSI"] < 35:
        kd_score += 2; kd_reason.append(f"RSI超賣 {t['RSI']:.1f}")
    if t["Close"] > t["MA5"]:
        kd_score += 1; kd_reason.append("站上MA5")
    if t["Volume"] > vol_ma10 * 1.0:
        kd_score += 1; kd_reason.append("放量")

    macd_score, macd_reason = 0, []
    if t["MACD"] > t["Signal"]:
        macd_score += 1; macd_reason.append("MACD多頭")
    if t["Histogram"] > 0:
        macd_score += 1; macd_reason.append("柱狀正值")
    if df["MACD_Buy"].iloc[-1]:
        macd_score += 2; macd_reason.append("★MACD金叉")
    if t["Histogram"] > y["Histogram"] and t["Histogram"] < 0:
        macd_score += 1; macd_reason.append("柱狀負值收斂(尚未轉正)")

    chip_score, chip_reason = 0, []
    if t["Total_Net"] > 0:
        chip_score += 1; chip_reason.append(f"法人買超 {t['Total_Net']:.0f}張")
    if t["Net_3d"] > 0:
        chip_score += 2; chip_reason.append("近3日連續吃貨")
    if t["Total_Net"] > df["Total_Net"].rolling(5).mean().iloc[-1] * 1.5 and t["Total_Net"] > 0:
        chip_score += 1; chip_reason.append("法人明顯放量買超")

    buy_raw = kd_score + macd_score + chip_score + fundamental_score
    buy_total = max(buy_raw - (P["ma200_penalty"] if below_ma200 else 0), 0)
    tech_buy = kd_score + macd_score
    tech_confirmed = tech_buy >= P["tech_min"]

    # ---- 賣出得分 ----
    sell_score, sell_reason = 0, []
    if (y["K"] >= y["D"]) and (t["K"] < t["D"]) and t["D"] > 65:
        sell_score += 3; sell_reason.append("KD高檔死叉")
    if t["RSI"] > 70:
        sell_score += 2; sell_reason.append(f"RSI超買 {t['RSI']:.1f}")
    if t["Close"] < t["MA5"]:
        sell_score += 1; sell_reason.append("跌破MA5")
    if t["Volume"] < vol_ma10 * 0.7:
        sell_score += 1; sell_reason.append("量縮嚴重")
    if t["MACD"] < t["Signal"]:
        sell_score += 1; sell_reason.append("MACD空頭")
    if t["Histogram"] < 0:
        sell_score += 1; sell_reason.append("柱狀負值")
    if df["MACD_Sell"].iloc[-1]:
        sell_score += 2; sell_reason.append("★MACD死叉")
    if t["Histogram"] < y["Histogram"] and t["Histogram"] > 0:
        sell_score += 1; sell_reason.append("柱狀由正轉負")
    if t["Total_Net"] < 0:
        sell_score += 1; sell_reason.append(f"法人賣超 {t['Total_Net']:.0f}張")
    if t["Net_3d"] < 0:
        sell_score += 2; sell_reason.append("近3日連續出貨")
    if roc_hit:
        sell_score += 2; sell_reason.append(f"{P['roc_days']}日急漲 {t['ROC']:.1%}（>{P['roc_th']:.0%}）")
    if vol_hit:
        sell_score += 2; sell_reason.append(f"爆量 {t['Vol_Ratio']:.1f}倍20日均量（>{P['vr_th']:.0f}倍）")
    if climax_hit:
        sell_score += 2; sell_reason.append("★高潮走勢：急漲+爆量同時出現")

    return dict(
        df=df, t=t, y=y, f=f, pe_threshold=pe_threshold,
        fundamental_score=fundamental_score, fundamental_reasons=fundamental_reasons,
        kd_reason=kd_reason, macd_reason=macd_reason, chip_reason=chip_reason,
        buy_raw=buy_raw, buy_total=buy_total, tech_buy=tech_buy, tech_confirmed=tech_confirmed,
        sell_score=sell_score, sell_reason=sell_reason,
        roc_hit=roc_hit, vol_hit=vol_hit, climax_hit=climax_hit,
        buy_blocked=buy_blocked, block_reasons=block_reasons, soft_notes=soft_notes,
        bias_60=bias_60, avg_vol_10_lots=avg_vol_10_lots,
    )


# ==================== 操作建議 ====================
def advice(R, P):
    buy_total, sell_score = R["buy_total"], R["sell_score"]
    climax, blocked = R["climax_hit"], R["buy_blocked"]

    # 買進建議
    if climax:
        buy = ("error", "🚫 高潮走勢（急漲+爆量），不追價、不加碼")
    elif blocked and buy_total >= 4:
        buy = ("warning", f"⚪ 買進訊號（調整後 {buy_total} 分）被過濾器攔截，暫不進場")
    elif blocked:
        buy = ("warning", "⚪ 無買進訊號，且不符合安全進場條件，暫不進場")
    elif buy_total >= 7 and not R["tech_confirmed"]:
        buy = ("warning",
               f"🟡 籌碼/基本面偏多，但技術面尚未確認（技術分 {R['tech_buy']}/12，需≥{P['tech_min']}）"
               "→ 僅小量試單，等站上MA5、MACD金叉再加碼")
    elif buy_total >= 11:
        buy = ("success", "🟢🟢🟢 極度強勢！基本面護航 + 技術籌碼同步，可積極布局或加碼")
    elif buy_total >= 7:
        buy = ("success", "🟢🟢 強烈買進：多頭與基本面訊號明確，適合進場")
    elif buy_total >= 4:
        buy = ("info", "🟢 可以買進：基本面或技術面初現跡象，可小量布局")
    else:
        buy = ("info", "⚪ 無明確買進訊號，等待")

    # 賣出建議
    if climax:
        sell = ("error",
                "🚨 高潮走勢警示！短線急漲 + 爆量，情緒極度亢奮\n\n"
                "- **風險管理**：切勿追價，不要因為買進分數高而加碼\n"
                "- **實現利潤**：建議分批獲利了結，至少減碼一半以上\n"
                "- **落袋為安**：剩餘部位設移動停利（如跌破MA5或前一日低點即出場）")
    elif sell_score >= 8:
        sell = ("error", "🔴🔴🔴 極度危險！建議立刻減碼或出場避險")
    elif sell_score >= 5:
        sell = ("error", "🔴🔴 強烈賣出：空頭訊號明顯，準備出場")
    elif sell_score >= 2:
        sell = ("warning", "🔴 建議減碼或出場：空頭訊號浮現")
    else:
        sell = ("success", "⚪ 無明顯賣訊，持有者續抱")

    # 綜合結論
    buy_ok = (not blocked) and buy_total >= 4
    sell_hit = sell_score >= 2
    if climax or sell_score >= 5:
        concl = ("error", "**以【賣出建議】為準**\n\n- 空手者：不進場\n- 持有者：依賣出建議減碼／停利")
    elif buy_ok and sell_hit:
        concl = ("warning", "**多空交雜（買進與賣出訊號同時存在）**\n\n"
                 "- 空手者：最多小量試單，等技術面轉強（站上MA5、MACD金叉）再加碼\n"
                 "- 持有者：不急著大幅減碼，但若再轉弱（賣出得分升到5以上）就減碼")
    elif buy_ok:
        concl = ("success", "**偏多**\n\n- 空手者：依買進建議進場\n- 持有者：續抱，可依買進建議加碼")
    elif sell_hit:
        concl = ("warning", "**偏空**\n\n- 空手者：不進場\n- 持有者：依賣出建議減碼")
    else:
        concl = ("info", "**觀望**：空手者等待訊號，持有者續抱")
    return buy, sell, concl


def show(box):
    level, msg = box
    getattr(st, level)(msg)


# ==================== 介面 ====================
st.title("📈 台股訊號查詢")
st.caption("基本面 + 技術面 + 籌碼面評分，含高潮走勢（急漲爆量）風險管理。資料來源：FinMind。")

with st.sidebar:
    st.header("查詢設定")
    with st.form("query"):
        target = st.text_input("股票代號", value="3026").strip()
        peers_text = st.text_input(
            "同業比較清單（逗號分隔）",
            value="3026, 2492, 2327, 2478, 2375",
            help="用來計算本益比最低 30% 門檻。請填同族群股票，系統會自動把查詢股票加入。",
        )
        token = st.text_input(
            "FinMind Token（選填）", type="password",
            help="未登入有每小時請求上限；有 token 額度較高。",
        ).strip() or _secret_token()

        st.subheader("高潮走勢")
        roc_days = st.number_input("急漲觀察天數", 2, 20, 5)
        roc_th = st.number_input("累積漲幅門檻（%）", 5.0, 100.0, 25.0, step=1.0) / 100
        vr_th = st.number_input("量比門檻（倍，相對20日均量）", 1.0, 10.0, 3.0, step=0.5)

        st.subheader("買進過濾")
        min_vol = st.number_input("10日均量下限（張）", 0, 100000, 500, step=100)
        bias_limit = st.number_input("季線乖離上限（%）", 5, 100, 25)
        ma200_penalty = st.number_input("年線下方扣分", 0, 10, 3)
        tech_min = st.number_input("技術確認門檻（KD+MACD 分）", 0, 12, 4)

        submitted = st.form_submit_button("查詢", use_container_width=True)

P = dict(roc_days=int(roc_days), roc_th=roc_th, vr_th=vr_th, min_vol=int(min_vol),
         bias_limit=bias_limit, ma200_penalty=int(ma200_penalty), tech_min=int(tech_min))

if not submitted:
    st.info("在左側輸入股票代號後按「查詢」。")
    st.stop()

if not target:
    st.error("請輸入股票代號。")
    st.stop()

peers = [p.strip() for p in peers_text.replace("，", ",").split(",") if p.strip()]
if target not in peers:
    peers.insert(0, target)

try:
    with st.spinner("抓取資料與計算中..."):
        R = compute(target, peers, token, P)
except Exception as e:
    st.error(f"查詢失敗：{e}")
    st.stop()

t, y = R["t"], R["y"]
st.subheader(f"{target}　{t.name.strftime('%Y/%m/%d')}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("收盤", f"{t['Close']:.2f}", f"{t['Close']/y['Close']-1:+.2%}")
c2.metric("成交量（張）", f"{t['Volume']/1000:,.0f}")
c3.metric("法人淨買賣（張）", f"{t['Total_Net']:+,.0f}", f"近3日 {t['Net_3d']:+,.0f}", delta_color="off")
c4.metric("RSI", f"{t['RSI']:.1f}")

c5, c6, c7, c8 = st.columns(4)
c5.metric("KD（K/D）", f"{t['K']:.1f} / {t['D']:.1f}")
c6.metric("MACD / Signal", f"{t['MACD']:.2f} / {t['Signal']:.2f}")
roc_txt = f"{t['ROC']:+.1%}" if not np.isnan(t["ROC"]) else "N/A"
vr_txt = f"{t['Vol_Ratio']:.2f}倍" if not np.isnan(t["Vol_Ratio"]) else "N/A"
c7.metric(f"{P['roc_days']}日漲幅", roc_txt, f"門檻 >{P['roc_th']:.0%}", delta_color="off")
c8.metric("量比（20日）", vr_txt, f"門檻 >{P['vr_th']:.0f}倍", delta_color="off")

st.divider()
buy, sell, concl = advice(R, P)

st.markdown("### 🎯 綜合結論")
show(concl)

colb, cols = st.columns(2)
with colb:
    st.markdown("### 🔵 買進建議")
    show(buy)
    st.write(f"買進得分：原始 **{R['buy_raw']}** / 調整後 **{R['buy_total']}**　（技術分 {R['tech_buy']}/12）")
    reasons = R["fundamental_reasons"] + R["kd_reason"] + R["macd_reason"] + R["chip_reason"]
    st.caption(" ｜ ".join(reasons) if reasons else "無買訊")
with cols:
    st.markdown("### 🔴 賣出建議")
    show(sell)
    st.write(f"賣出得分：**{R['sell_score']}**")
    st.caption(" ｜ ".join(R["sell_reason"]) if R["sell_reason"] else "無賣訊")

st.divider()
st.markdown("### 🛡️ 買進過濾器")
if R["buy_blocked"]:
    st.warning("攔截：" + " ｜ ".join(R["block_reasons"]))
else:
    st.success("通過：流動性、乖離率、高潮檢核")
for n in R["soft_notes"]:
    st.caption("↳ " + n)
bias_txt = f"{R['bias_60']:+.1f}%" if not np.isnan(R["bias_60"]) else "N/A"
st.caption(f"10日均量 {R['avg_vol_10_lots']:,.0f} 張　季線乖離 {bias_txt}")

st.markdown("### 🧱 基本面")
st.write(f"基本面得分：**{R['fundamental_score']}/6**")
st.caption(" ｜ ".join(R["fundamental_reasons"]) if R["fundamental_reasons"] else "基本面條件未滿足")
st.caption(f"營收檢核：{R['f'].get('rev_note', '')}")
pe_t = R["pe_threshold"]
st.caption(
    f"本益比 {R['f']['pe_val']:.1f}　族群30%分位門檻 {pe_t:.1f}"
    if not (np.isnan(R["f"]["pe_val"]) or np.isnan(pe_t)) else "本益比資料不足"
)

st.divider()
df = R["df"]
tab1, tab2 = st.tabs(["股價與均線", "MACD"])
with tab1:
    st.line_chart(df[["Close", "MA5", "MA60", "MA200"]].tail(250))
with tab2:
    st.line_chart(df[["MACD", "Signal"]].tail(250))
    st.bar_chart(df["Histogram"].tail(250))

st.caption("⚠️ 本工具僅為程式化訊號整理，不構成投資建議。投資有風險，請自行判斷並控管部位。")
