import sys
import streamlit as st

st.set_page_config(page_title="診斷", page_icon="🩺")
st.title("🩺 部署診斷頁")
st.write("如果你看得到這一頁，代表 Streamlit 本身與主程式檔案設定都正常。")

st.write(f"**Python**：{sys.version.split()[0]}")
st.write(f"**Streamlit**：{st.__version__}")

for name in ["numpy", "pandas", "requests"]:
    try:
        mod = __import__(name)
        st.success(f"{name} {getattr(mod, '__version__', '?')} ✅")
    except Exception as e:
        st.error(f"{name} 載入失敗")
        st.exception(e)

st.subheader("FinMind 連線測試")
if st.button("測試連線"):
    try:
        import requests
        r = requests.get(
            "https://api.finmindtrade.com/api/v4/data",
            params={"dataset": "TaiwanStockPrice", "data_id": "3026", "start_date": "2026-10-14"},
            timeout=30,
        )
        st.write("HTTP 狀態：", r.status_code)
        j = r.json()
        st.write("API status：", j.get("status"), "｜msg：", j.get("msg"))
        st.write("資料筆數：", len(j.get("data", [])))
        if j.get("data"):
            st.write("欄位：", list(j["data"][0].keys()))
    except Exception as e:
        st.error("連線測試失敗")
        st.exception(e)
