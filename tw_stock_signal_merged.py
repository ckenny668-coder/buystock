from datetime import datetime
from FinMind.data import DataLoader
import numpy as np
import pandas as pd

# 初始化資料載入器
dl = DataLoader()


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
    # 注意：itertuples 會把底線開頭的欄位改名，所以改用 zip 取值
    lookup = {
        (int(y_), int(m_)): float(v)
        for y_, m_, v in zip(rev_df["_y"], rev_df["_m"], rev_df["revenue"])
    }

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

# 定義要比對的族群股票清單（用於計算本益比前 30% 低位）
peer_tickers = ["5347", "2330", "2303", "6770", "2342", "3707"]
target_ticker = "5347"

# ==================== 參數設定 ====================
# --- 高潮走勢（Climax）：風險管理 / 實現利潤 / 落袋為安 ---
ROC_DAYS = 5               # 短天期漲幅觀察天數
ROC_THRESHOLD = 0.25       # N 日累積漲幅 > 25% → 非理性垂直飆升
VOL_MA_DAYS = 20           # 成交量均線天數
VOL_RATIO_THRESHOLD = 3.0  # 當日量 > 20 日均量 3 倍 → 資金全面湧入

# --- 買進過濾器（只攔買進，不影響賣出判斷）---
MIN_AVG_VOL_LOTS = 500     # 流動性：10 日均量下限（張）
BIAS60_LIMIT = 25          # 季線乖離率上限（%），過高暫緩買進
MA200_PENALTY = 3          # 收盤在年線下方：買進分數扣分（軟性過濾）
TECH_CONFIRM_MIN = 4       # 技術面(KD+MACD)至少幾分，才算「技術確認」可給強烈買進


# ==================== 基本面三條件計算函式 ====================
def get_fundamental_scores(tickers, target):
    fund_scores = {}
    pe_list = {}

    start_f_date = "2023-01-01"  # 抓取足夠算年增率與ROE的長度

    print("🔍 正在抓取基本面數據（營收、ROE、本益比）...")

    for tk in tickers:
        # 三項各自獨立 try：任何一項失敗，不會連累其他項
        rev_pass, rev_note = False, ""
        roe_pass, latest_roe = False, 0
        latest_pe = np.nan

        # 1. 抓月營收計算：近 3 個月營收合計 vs 去年同期
        try:
            rev_df = dl.taiwan_stock_month_revenue(
                stock_id=tk, start_date=start_f_date
            )
            rev_pass, rev_note = check_revenue_accurate(rev_df)
        except Exception as e:
            rev_note = f"營收檢核失敗：{type(e).__name__}: {e}"
            print(f"⚠️ {tk} {rev_note}")

        # 2. 抓 ROE：最新一季 ROE > 0
        try:
            roe_df = dl.taiwan_stock_financial_statement(
                stock_id=tk, start_date=start_f_date
            )
            if not roe_df.empty:
                roe_data = roe_df[roe_df["type"] == "ROE"]
                if not roe_data.empty:
                    latest_roe = float(roe_data.iloc[-1]["value"])
                    roe_pass = latest_roe > 0
        except Exception as e:
            print(f"⚠️ {tk} ROE 抓取失敗：{type(e).__name__}: {e}")

        # 3. 抓 PE (本益比)
        try:
            per_df = dl.taiwan_stock_per_pbr(
                stock_id=tk, start_date="2024-01-01"
            )
            if not per_df.empty:
                valid_pe = per_df[per_df["PER"] > 0]
                if not valid_pe.empty:
                    latest_pe = float(valid_pe.iloc[-1]["PER"])
        except Exception as e:
            print(f"⚠️ {tk} 本益比抓取失敗：{type(e).__name__}: {e}")

        pe_list[tk] = latest_pe
        fund_scores[tk] = {
            "rev_pass": rev_pass,
            "rev_note": rev_note,
            "roe_pass": roe_pass,
            "roe_val": latest_roe,
            "pe_val": latest_pe,
        }

    # 計算族群本益比 30% 分位數 (低價門檻)
    valid_pes = [v for v in pe_list.values() if not np.isnan(v)]
    pe_threshold = (
        np.percentile(valid_pes, 30) if len(valid_pes) > 0 else np.nan
    )

    # 判定目標股 PE 是否低於 30% 門檻
    target_pe = fund_scores[target]["pe_val"]
    pe_pass = False
    if not np.isnan(target_pe) and not np.isnan(pe_threshold):
        pe_pass = target_pe <= pe_threshold

    fund_scores[target]["pe_pass"] = pe_pass
    return fund_scores[target], pe_threshold


# ==================== 執行基本面評估 ====================
fund_res, pe_cutoff = get_fundamental_scores(peer_tickers, target_ticker)

fundamental_score = 0
fundamental_reasons = []

if fund_res["rev_pass"]:
    fundamental_score += 2
    fundamental_reasons.append(f"近3月營收高於去年同期(+2分)：{fund_res['rev_note']}")

if fund_res["roe_pass"]:
    fundamental_score += 2
    fundamental_reasons.append(f"最新ROE為正({fund_res['roe_val']:.2f}%)(+2分)")

if fund_res["pe_pass"]:
    fundamental_score += 2
    pe_str = (
        f"{fund_res['pe_val']:.1f}" if not np.isnan(fund_res['pe_val']) else "N/A"
    )
    fundamental_reasons.append(f"PE在族群最低30%內({pe_str})(+2分)")


# ==================== 股價資料（多抓一段供 MA200 暖機）====================
df = dl.taiwan_stock_daily(stock_id=target_ticker, start_date="2023-01-01")
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df = df[["open", "max", "min", "close", "Trading_Volume"]].copy()
df.columns = ["Open", "High", "Low", "Close", "Volume"]

# ==================== 技術指標 ====================
df["MA5"] = df["Close"].rolling(5).mean()
df["MA60"] = df["Close"].rolling(60).mean()    # 季線
df["MA200"] = df["Close"].rolling(200).mean()  # 年線（長期趨勢）

# KD
low9 = df["Low"].rolling(9).min()
high9 = df["High"].rolling(9).max()
rsv = (df["Close"] - low9) / (high9 - low9) * 100
df["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
df["D"] = df["K"].ewm(alpha=1 / 3, adjust=False).mean()

# RSI
delta = df["Close"].diff()
gain = delta.where(delta > 0, 0).rolling(14).mean()
loss = -delta.where(delta < 0, 0).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df["RSI"] = 100 - 100 / (1 + rs)

# MACD
df["EMA12"] = df["Close"].ewm(span=12, adjust=False).mean()
df["EMA26"] = df["Close"].ewm(span=26, adjust=False).mean()
df["MACD"] = df["EMA12"] - df["EMA26"]
df["Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
df["Histogram"] = df["MACD"] - df["Signal"]

df["MACD_Buy"] = (df["MACD"] > df["Signal"]) & (
    df["MACD"].shift(1) <= df["Signal"].shift(1)
)
df["MACD_Sell"] = (df["MACD"] < df["Signal"]) & (
    df["MACD"].shift(1) >= df["Signal"].shift(1)
)

# ==================== 高潮走勢指標：ROC + 量比 ====================
# 【風險管理】短天期急漲 + 爆量，是情緒過熱的典型特徵，
# 此時重點不是追價，而是「實現利潤、落袋為安」。
df["ROC"] = df["Close"] / df["Close"].shift(ROC_DAYS) - 1
df["Vol_MA20"] = df["Volume"].rolling(VOL_MA_DAYS).mean()
df["Vol_Ratio"] = df["Volume"] / df["Vol_MA20"]

# ==================== 三大法人 ====================
inst = dl.taiwan_stock_institutional_investors(
    stock_id=target_ticker, start_date="2023-01-01"
)
inst["date"] = pd.to_datetime(inst["date"])
inst["net"] = inst["buy"] - inst["sell"]

inst_daily = inst.groupby("date")["net"].sum().to_frame("Total_Net")
inst_daily["Total_Net"] = inst_daily["Total_Net"] / 1000

df = df.join(inst_daily, how="left")
df["Total_Net"] = df["Total_Net"].fillna(0)
df["Net_3d"] = df["Total_Net"].rolling(3).sum()

# 指標全部算完後，才切掉暖機區間
df = df.loc["2024-01-01":]

# ==================== 取今日、昨日 ====================
t = df.iloc[-1]
y = df.iloc[-2]
vol_ma10 = df["Volume"].rolling(10).mean().iloc[-1]
avg_vol_10_lots = vol_ma10 / 1000  # 10日均量（張）

# ==================== 高潮走勢判定 ====================
roc_hit = (not np.isnan(t["ROC"])) and t["ROC"] > ROC_THRESHOLD
vol_hit = (not np.isnan(t["Vol_Ratio"])) and t["Vol_Ratio"] > VOL_RATIO_THRESHOLD
climax_hit = roc_hit and vol_hit

# ==================== 買進過濾器（只攔買進，不影響賣出）====================
buy_blocked = False        # 硬性攔截
block_reasons = []
soft_notes = []            # 軟性扣分說明

# 1. 流動性
if avg_vol_10_lots < MIN_AVG_VOL_LOTS:
    buy_blocked = True
    block_reasons.append(
        f"流動性不足(10日均量 {avg_vol_10_lots:.0f}張 < {MIN_AVG_VOL_LOTS}張)"
    )

# 2. 季線乖離過大（防追高）
bias_60 = (t["Close"] - t["MA60"]) / t["MA60"] * 100
if not np.isnan(bias_60) and bias_60 > BIAS60_LIMIT:
    buy_blocked = True
    block_reasons.append(f"季線乖離過大({bias_60:.1f}% > {BIAS60_LIMIT}%)，慎防追高")

# 3. 高潮走勢：急漲爆量時不追價
if climax_hit:
    buy_blocked = True
    block_reasons.append("高潮走勢（急漲+爆量），不追價")

# 4. 年線（軟性）：年線下方扣分，不直接歸零，保留低檔反彈訊號的可能
below_ma200 = (not np.isnan(t["MA200"])) and t["Close"] < t["MA200"]
if below_ma200:
    soft_notes.append(f"收盤在年線下方(買進分數 -{MA200_PENALTY})")

# ==================== 買進得分 ====================
kd_score, kd_reason = 0, []
if (y["K"] <= y["D"]) and (t["K"] > t["D"]) and t["D"] < 35:
    kd_score += 3
    kd_reason.append("KD金叉（低檔）")
if t["RSI"] < 35:
    kd_score += 2
    kd_reason.append(f"RSI超賣 {t['RSI']:.1f}")
if t["Close"] > t["MA5"]:
    kd_score += 1
    kd_reason.append("站上MA5")
if t["Volume"] > vol_ma10 * 1.0:
    kd_score += 1
    kd_reason.append("放量")

macd_score, macd_reason = 0, []
if t["MACD"] > t["Signal"]:
    macd_score += 1
    macd_reason.append("MACD多頭")
if t["Histogram"] > 0:
    macd_score += 1
    macd_reason.append("柱狀正值")
if df["MACD_Buy"].iloc[-1]:
    macd_score += 2
    macd_reason.append("★MACD金叉")
if t["Histogram"] > y["Histogram"] and t["Histogram"] < 0:
    macd_score += 1
    macd_reason.append("柱狀負值收斂(尚未轉正)")

chip_score, chip_reason = 0, []
if t["Total_Net"] > 0:
    chip_score += 1
    chip_reason.append(f"法人買超 {t['Total_Net']:.0f}張")
if t["Net_3d"] > 0:
    chip_score += 2
    chip_reason.append("近3日連續吃貨")
if (
    t["Total_Net"] > df["Total_Net"].rolling(5).mean().iloc[-1] * 1.5
    and t["Total_Net"] > 0
):
    chip_score += 1
    chip_reason.append("法人明顯放量買超")

# 原始買進得分（含基本面滿分 6 分），保留不覆蓋
buy_raw = kd_score + macd_score + chip_score + fundamental_score
# 軟性扣分後的買進得分（用於分級）
buy_total = max(buy_raw - (MA200_PENALTY if below_ma200 else 0), 0)

# 技術面分數（KD + MACD，滿分 12）：用來確認「籌碼/基本面偏多」是否有技術面配合
tech_buy = kd_score + macd_score
tech_confirmed = tech_buy >= TECH_CONFIRM_MIN

# ==================== 賣出得分 ====================
sell_score, sell_reason = 0, []
if (y["K"] >= y["D"]) and (t["K"] < t["D"]) and t["D"] > 65:
    sell_score += 3
    sell_reason.append("KD高檔死叉")
if t["RSI"] > 70:
    sell_score += 2
    sell_reason.append(f"RSI超買 {t['RSI']:.1f}")
if t["Close"] < t["MA5"]:
    sell_score += 1
    sell_reason.append("跌破MA5")
if t["Volume"] < vol_ma10 * 0.7:
    sell_score += 1
    sell_reason.append("量縮嚴重")
if t["MACD"] < t["Signal"]:
    sell_score += 1
    sell_reason.append("MACD空頭")
if t["Histogram"] < 0:
    sell_score += 1
    sell_reason.append("柱狀負值")
if df["MACD_Sell"].iloc[-1]:
    sell_score += 2
    sell_reason.append("★MACD死叉")
if t["Histogram"] < y["Histogram"] and t["Histogram"] > 0:
    sell_score += 1
    sell_reason.append("柱狀由正轉負")
if t["Total_Net"] < 0:
    sell_score += 1
    sell_reason.append(f"法人賣超 {t['Total_Net']:.0f}張")
if t["Net_3d"] < 0:
    sell_score += 2
    sell_reason.append("近3日連續出貨")

# ---- 高潮走勢賣出加分（風險管理 / 實現利潤 / 落袋為安）----
if roc_hit:
    sell_score += 2
    sell_reason.append(f"{ROC_DAYS}日急漲 {t['ROC']:.1%}（>{ROC_THRESHOLD:.0%}）")
if vol_hit:
    sell_score += 2
    sell_reason.append(
        f"爆量 {t['Vol_Ratio']:.1f}倍{VOL_MA_DAYS}日均量（>{VOL_RATIO_THRESHOLD:.0f}倍）"
    )
if climax_hit:
    sell_score += 2
    sell_reason.append("★高潮走勢：急漲+爆量同時出現")

# ==================== 輸出結果 ====================
print("=" * 100)
print(
    f"         🚨 {target_ticker} {t.name.strftime('%Y/%m/%d')} 收盤 {t['Close']:.2f} 漲跌 {t['Close']/y['Close']-1:+.2%}"
)
print(
    f"成交量 {t['Volume']/1000:,.0f}張  KD {t['K']:.1f}/{t['D']:.1f}  RSI {t['RSI']:.1f}  MA5 {t['MA5']:.2f}"
)
print(
    f"MACD {t['MACD']:.3f} Signal {t['Signal']:.3f} 柱狀圖 {t['Histogram']:.3f}"
)
print(f"法人淨買賣 {t['Total_Net']:+.0f}張 近3日累計 {t['Net_3d']:+.0f}張")
roc_str = f"{t['ROC']:+.1%}" if not np.isnan(t["ROC"]) else "N/A"
vr_str = f"{t['Vol_Ratio']:.2f}倍" if not np.isnan(t["Vol_Ratio"]) else "N/A"
bias_str = f"{bias_60:+.1f}%" if not np.isnan(bias_60) else "N/A"
print(
    f"{ROC_DAYS}日漲幅 {roc_str}（門檻 >{ROC_THRESHOLD:.0%}）  "
    f"量比 {vr_str}（門檻 >{VOL_RATIO_THRESHOLD:.0f}倍）  季線乖離 {bias_str}"
)
print("-" * 100)

# 過濾器狀態（只針對買進）
if buy_blocked:
    print(f"🛡️ 買進過濾器：【攔截】{' | '.join(block_reasons)}")
else:
    print("🛡️ 買進過濾器：【通過】流動性、乖離率、高潮檢核")
if soft_notes:
    print(f"   ↳ 軟性提醒：{' | '.join(soft_notes)}")

print("-" * 100)
print(
    f"🧱 基本面得分：{fundamental_score:2d}/6分 → {' | '.join(fundamental_reasons) if fundamental_reasons else '基本面條件未滿足'}"
)
print(f"   營收檢核：{fund_res.get('rev_note', '')}")
buy_detail = ' | '.join(fundamental_reasons + kd_reason + macd_reason + chip_reason)
print(
    f"🔵 買進得分  ：原始 {buy_raw:2d} / 調整後 {buy_total:2d}"
    f"{'（被過濾器攔截）' if buy_blocked else ''} → {buy_detail if buy_raw > 0 else '無買訊'}"
)
print(
    f"🔴 賣出得分  ：{sell_score:2d} → {' | '.join(sell_reason) if sell_reason else '無賣訊'}"
)
print("=" * 100)

# ==================== 操作建議 ====================
# 買進建議與賣出建議各自獨立判斷、各自輸出，互不覆蓋。
print("【明日操作建議】".center(60))

# ---------- 1. 買進建議（適用：空手 / 想加碼）----------
print("🔵 1. 買進建議")
if climax_hit:
    print("   🚫 高潮走勢（急漲+爆量），不追價、不加碼")
elif buy_blocked and buy_total >= 4:
    print(f"   ⚪ 買進訊號（調整後 {buy_total} 分）被過濾器攔截，暫不進場")
elif buy_blocked:
    print("   ⚪ 無買進訊號，且不符合安全進場條件，暫不進場")
elif buy_total >= 7 and not tech_confirmed:
    print(
        f"   🟡 籌碼/基本面偏多，但技術面尚未確認（技術分 {tech_buy}/12，需≥{TECH_CONFIRM_MIN}）"
        "→ 僅小量試單，等站上MA5、MACD金叉再加碼"
    )
elif buy_total >= 11:
    print("   🟢🟢🟢 極度強勢！基本面護航 + 技術籌碼同步，可積極布局或加碼")
elif buy_total >= 7:
    print("   🟢🟢 強烈買進 多頭與基本面訊號明確，適合進場")
elif buy_total >= 4:
    print("   🟢 可以買進 基本面或技術面初現跡象，可小量布局")
else:
    print("   ⚪ 無明確買進訊號，等待")

# ---------- 2. 賣出建議（適用：已持有）----------
print("🔴 2. 賣出建議")
if climax_hit:
    print("   🚨🚨🚨 高潮走勢警示！短線急漲 + 爆量，情緒極度亢奮")
    print("   ▶ 風險管理：切勿追價，不要因為買進分數高而加碼")
    print("   ▶ 實現利潤：建議分批獲利了結，至少減碼一半以上")
    print("   ▶ 落袋為安：剩餘部位設移動停利（如跌破MA5或前一日低點即出場）")
elif sell_score >= 8:
    print("   🔴🔴🔴 極度危險！建議立刻減碼或出場避險")
elif sell_score >= 5:
    print("   🔴🔴 強烈賣出 空頭訊號明顯，準備跑路")
elif sell_score >= 2:
    print("   🔴 建議減碼或出場 空頭訊號浮現")
else:
    print("   ⚪ 無明顯賣訊，持有者續抱")

# ---------- 3. 綜合結論（買賣建議矛盾時，照這一行執行）----------
buy_ok = (not buy_blocked) and buy_total >= 4
sell_hit = sell_score >= 2
print("🎯 3. 綜合結論")
if climax_hit or sell_score >= 5:
    print("   以【賣出建議】為準：")
    print("   ▶ 空手者：不進場")
    print("   ▶ 持有者：依賣出建議減碼／停利")
elif buy_ok and sell_hit:
    print("   多空交雜（買進與賣出訊號同時存在）：")
    print("   ▶ 空手者：最多小量試單，等技術面轉強（站上MA5、MACD金叉）再加碼")
    print("   ▶ 持有者：不急著大幅減碼，但若再轉弱（賣出得分升到5以上）就減碼")
elif buy_ok:
    print("   偏多：")
    print("   ▶ 空手者：依買進建議進場")
    print("   ▶ 持有者：續抱，可依買進建議加碼")
elif sell_hit:
    print("   偏空：")
    print("   ▶ 空手者：不進場")
    print("   ▶ 持有者：依賣出建議減碼")
else:
    print("   觀望：空手者等待訊號，持有者續抱")

# ---------- 附註提醒 ----------
if not climax_hit and (roc_hit or vol_hit):
    print("⚠️ 注意：高潮條件已成立其一，請留意風險管理，已有獲利者可考慮部分落袋為安")
print("=" * 100)
