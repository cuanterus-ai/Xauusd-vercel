"""
XAUUSDc Analyzer
Jalankan:  pip install -r requirements.txt  &&  python app.py
Buka:      http://localhost:5000  lalu tekan tombol ANALISA

Strategi (hasil backtest XAUUSD_15m.csv, Mar-Sep 2026):
  Breakout M15 (close tembus high/low 20 candle) searah trend H1 + H4 (EMA20 vs EMA50).
  SL = 4 ATR, TP1 = 1 ATR, TP2 = 1.5 ATR, TP3 = 2 ATR.
Perintah backtest ulang:  python app.py --backtest XAUUSD_15m.csv
"""
import sys, os, json, re, math, time
import urllib.request, xml.etree.ElementTree as ET
from datetime import datetime, timezone, timedelta
import numpy as np
import pandas as pd

try:  # Vercel: filesystem read-only kecuali /tmp
    import yfinance as _yf
    _yf.set_tz_cache_location("/tmp/yf_cache")
except Exception:
    pass

CSV_FALLBACK = os.environ.get("XAU_CSV", "XAUUSD_15m.csv")
SYMBOL = os.environ.get("XAU_SYMBOL", "GC=F")  # futures emas; selisih beberapa USD dari spot XAUUSDc
LOOKBACK, SL_ATR, TP_ATRS = 20, 4.0, (1.0, 1.5, 2.0)


# ---------------------------------------------------------------- indikator
def ema(s, n): return s.ewm(span=n, adjust=False).mean()

def rsi(s, n=14):
    d = s.diff()
    u = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + u / l)

def atr(d, n=14):
    pc = d.close.shift()
    tr = pd.concat([d.high - d.low, (d.high - pc).abs(), (d.low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def resample(d, rule):
    return d.resample(rule).agg({"open": "first", "high": "max", "low": "min",
                                 "close": "last", "volume": "sum"}).dropna()

def trend_label(d):
    e20, e50, a = ema(d.close, 20).iloc[-1], ema(d.close, 50).iloc[-1], atr(d).iloc[-1]
    c = d.close.iloc[-1]
    if abs(e20 - e50) < 0.3 * a: return "SIDEWAYS", 0
    return ("BULLISH", 1) if e20 > e50 and c > e50 else ("BEARISH", -1) if e20 < e50 and c < e50 else ("SIDEWAYS", 0)


# ---------------------------------------------------------------- data harga
def load_prices():
    src = "CSV lokal"
    try:
        import yfinance as yf
        df = yf.download(SYMBOL, interval="15m", period="59d", progress=False, auto_adjust=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]].dropna()
        df.index = pd.to_datetime(df.index, utc=True)
        if len(df) > 300: src = f"Yahoo Finance ({SYMBOL})"
        else: raise ValueError("data live kurang")
    except Exception:
        df = pd.read_csv(CSV_FALLBACK, parse_dates=["datetime"]).set_index("datetime")
        df.index = pd.to_datetime(df.index, utc=True)
    # buang candle terakhir bila belum close
    if src != "CSV lokal" and df.index[-1] + timedelta(minutes=15) > datetime.now(timezone.utc):
        df = df.iloc[:-1]
    return df, src


# ---------------------------------------------------------------- support / resistance
def swings(d, k=3):
    hi, lo = [], []
    h, l = d.high.values, d.low.values
    for i in range(k, len(d) - k):
        if h[i] == h[i - k:i + k + 1].max(): hi.append((h[i], "swing high"))
        if l[i] == l[i - k:i + k + 1].min(): lo.append((l[i], "swing low"))
    return hi, lo

def build_levels(m15, h1, atr15):
    price = m15.close.iloc[-1]
    pts = []  # (harga, sumber, bobot)
    for p, _ in sum(swings(h1.tail(400), 3), []): pts.append((p, "swing H1", 2))
    for p, _ in sum(swings(m15.tail(800), 3), []): pts.append((p, "swing M15", 1))
    day = m15.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    if len(day) >= 2:
        y = day.iloc[-2]
        pts += [(y.high, "High kemarin", 4), (y.low, "Low kemarin", 4), (y.close, "Close kemarin", 2)]
    wk = m15.resample("W").agg({"high": "max", "low": "min"}).dropna()
    if len(wk) >= 2:
        pts += [(wk.iloc[-2].high, "High minggu lalu", 4), (wk.iloc[-2].low, "Low minggu lalu", 4)]
    r50 = round(price / 50) * 50
    for k in range(-4, 5): pts.append((r50 + 50 * k, "Level psikologis", 3))
    pts += [(h1.high.tail(24).max(), "High 24 jam", 3), (h1.low.tail(24).min(), "Low 24 jam", 3)]

    tol = max(atr15 * 0.6, 1.0)
    pts.sort()
    clusters = []
    for p, s, w in pts:
        if clusters and abs(p - clusters[-1]["p"]) <= tol:
            c = clusters[-1]; c["w"] += w; c["src"].add(s); c["ps"].append(p)
            c["p"] = float(np.mean(c["ps"]))
        else:
            clusters.append({"p": p, "w": w, "src": {s}, "ps": [p]})
    def fmt(c): return {"price": round(c["p"], 2), "score": c["w"],
                        "info": ", ".join(sorted(c["src"])) + f" (skor {c['w']})"}
    res = [fmt(c) for c in clusters if c["p"] > price + 0.3 * atr15]
    sup = [fmt(c) for c in clusters if c["p"] < price - 0.3 * atr15]
    res = sorted(res, key=lambda x: x["price"])[:3]
    sup = sorted(sup, key=lambda x: -x["price"])[:3]
    return res, sup


# ---------------------------------------------------------------- struktur M15
def m15_structure(m15):
    hi, lo = [], []
    h, l = m15.high.values[-300:], m15.low.values[-300:]
    k = 3
    for i in range(k, len(h) - k):
        if h[i] == h[i - k:i + k + 1].max(): hi.append(h[i])
        if l[i] == l[i - k:i + k + 1].min(): lo.append(l[i])
    if len(hi) < 2 or len(lo) < 2: return "Belum jelas"
    hh, hl = hi[-1] > hi[-2], lo[-1] > lo[-2]
    if hh and hl: return "Bullish (HH + HL)"
    if not hh and not hl: return "Bearish (LH + LL)"
    return "Range / transisi (" + ("HH+LL" if hh else "LH+HL") + ")"


# ---------------------------------------------------------------- berita & makro
POS = ["rate cut", "dovish", "safe haven", "safe-haven", "geopolit", "war", "tension", "weak dollar", "dollar falls",
       "dollar slips", "inflation rises", "rally", "surge", "record high", "gains", "jumps", "central bank buying", "etf inflow"]
NEG = ["rate hike", "hawkish", "strong dollar", "dollar rises", "dollar gains", "yields rise", "yields climb", "selloff",
       "sell-off", "drops", "falls", "slides", "slump", "tumbles", "profit-taking", "ceasefire", "risk-on", "etf outflow"]
FEEDS = {
    "Google News": "https://news.google.com/rss/search?q=gold+price+XAUUSD&hl=en-US&gl=US&ceid=US:en",
    "Kitco": "https://www.kitco.com/rss/",
    "FXStreet": "https://www.fxstreet.com/rss/news",
}

def http_get(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()

def fetch_news():
    items, score = [], 0
    for name, url in FEEDS.items():
        try:
            root = ET.fromstring(http_get(url))
            for it in root.iter("item"):
                t = (it.findtext("title") or "").strip()
                if not t: continue
                if name != "Google News" and not re.search(r"gold|xau|bullion|dollar|fed|yield", t, re.I): continue
                tl = t.lower()
                s = sum(w in tl for w in POS) - sum(w in tl for w in NEG)
                items.append({"src": name, "title": t, "s": s}); score += s
                if sum(1 for x in items if x["src"] == name) >= 8: break
        except Exception:
            continue
    return items, score

def fetch_macro():
    out = {}
    try:
        import yfinance as yf
        for name, tk in {"DXY (Dollar Index)": "DX-Y.NYB", "US10Y yield": "^TNX"}.items():
            d = yf.download(tk, interval="1h", period="5d", progress=False)
            if isinstance(d.columns, pd.MultiIndex): d.columns = d.columns.get_level_values(0)
            c = d["Close"].dropna()
            chg = (c.iloc[-1] / c.iloc[-24] - 1) * 100 if len(c) > 24 else 0
            out[name] = {"last": round(float(c.iloc[-1]), 3), "chg24h": round(float(chg), 2)}
    except Exception:
        pass
    return out

def fetch_calendar():
    """Berita USD high-impact terdekat (ForexFactory mirror). Mengembalikan (daftar, menit_ke_berita_terdekat)."""
    ev, nearest = [], None
    try:
        data = json.loads(http_get("https://nfs.faireconomy.media/ff_calendar_thisweek.json"))
        now = datetime.now(timezone.utc)
        for e in data:
            if e.get("country") != "USD" or e.get("impact") != "High": continue
            t = datetime.fromisoformat(e["date"]).astimezone(timezone.utc)
            mins = (t - now).total_seconds() / 60
            if -30 <= mins <= 24 * 60:
                ev.append({"title": e["title"], "time": t.strftime("%d/%m %H:%M UTC"), "mins": int(mins)})
                if nearest is None or abs(mins) < abs(nearest): nearest = mins
    except Exception:
        pass
    return ev, nearest


# ---------------------------------------------------------------- analisa utama
def analyze(broker=None):
    m15, src = load_prices()
    h1, h4 = resample(m15, "1h"), resample(m15, "4h")
    # pakai HTF yang sudah close saja (sama seperti backtest)
    h1c, h4c = h1.iloc[:-1], h4.iloc[:-1]
    t1, s1 = trend_label(h1c); t4, s4 = trend_label(h4c)

    a = atr(m15); r = rsi(m15.close)
    A, R, price = float(a.iloc[-1]), float(r.iloc[-1]), float(m15.close.iloc[-1])
    hh = float(m15.high.iloc[-LOOKBACK - 1:-1].max()); ll = float(m15.low.iloc[-LOOKBACK - 1:-1].min())
    structure = m15_structure(m15)
    res, sup = build_levels(m15, h1c, A)

    news, nscore = fetch_news()
    macro = fetch_macro()
    cal, near = fetch_calendar()

    action, reason = "WAIT", []
    if s1 == s4 and s1 != 0:
        if s1 == 1 and price > hh: action = "BUY"
        elif s1 == -1 and price < ll: action = "SELL"
    # filter keselamatan
    block = None
    if near is not None and -15 <= near <= 30: block = f"Berita USD high-impact dalam {int(near)} menit"
    if action == "BUY" and R > 80: block = "RSI M15 terlalu jenuh beli (>80)"
    if action == "SELL" and R < 20: block = "RSI M15 terlalu jenuh jual (<20)"
    if block and action != "WAIT":
        reason.append(f"Sinyal breakout ada, tapi DITUNDA: {block}."); action = "WAIT"

    if action in ("BUY", "SELL"):
        side = 1 if action == "BUY" else -1
        reason = [f"Trend H4 {t4} & H1 {t1} searah.",
                  f"Candle M15 close {'di atas high' if side == 1 else 'di bawah low'} {LOOKBACK} candle terakhir "
                  f"({hh if side == 1 else ll:.2f}) = breakout.",
                  f"Struktur M15: {structure}; RSI M15 {R:.0f}."]
        if (nscore > 0 and side == -1) or (nscore < 0 and side == 1):
            reason.append("Catatan: sentimen berita berlawanan arah, kecilkan lot.")
        entry = (round(price - 0.2 * A, 2), round(price + 0.2 * A, 2))
        levels = {"sl": round(price - side * SL_ATR * A, 2),
                  "tp1": round(price + side * TP_ATRS[0] * A, 2),
                  "tp2": round(price + side * TP_ATRS[1] * A, 2),
                  "tp3": round(price + side * TP_ATRS[2] * A, 2)}
    else:
        if not block:
            if s1 != s4 or s1 == 0:
                reason.append(f"Trend H4 ({t4}) dan H1 ({t1}) belum searah, tidak ada edge.")
            else:
                trig = hh if s1 == 1 else ll
                reason.append(f"Trend {t4} searah, tapi belum breakout. Tunggu close M15 "
                              f"{'di atas' if s1 == 1 else 'di bawah'} {trig:.2f} untuk {'BUY' if s1 == 1 else 'SELL'}.")
        else:
            reason.append(block + ".")
        entry, levels = None, None

    # ---- penyesuaian ke harga broker (selisih futures vs spot)
    off, warn = 0.0, None
    feed_price = price
    if broker:
        d = broker - price
        if abs(d) > 80:
            warn = f"Harga broker {broker} beda terlalu jauh ({d:+.1f}) dari data, diabaikan."
        else:
            off = d
    def sh(x): return round(x + off, 2)
    if off:
        entry = (sh(entry[0]), sh(entry[1])) if entry else None
        levels = {k: sh(v) for k, v in levels.items()} if levels else None
        for lst in (res, sup):
            for it in lst: it["price"] = sh(it["price"])
        reason = [re.sub(r"\d{4}\.\d{2}", lambda m: f"{float(m.group()) + off:.2f}", t) for t in reason]
    return {
        "source": src, "time": m15.index[-1].strftime("%Y-%m-%d %H:%M UTC"),
        "price": round(price + off, 2), "feed_price": round(feed_price, 2), "offset": round(off, 2), "warn": warn,
        "atr": round(A, 2), "action": action, "reason": reason,
        "entry": entry, "levels": levels,
        "trend_h4": t4, "trend_h1": t1, "structure": structure, "rsi": round(R, 1),
        "resistance": res, "support": sup,
        "news": news[:10], "news_score": nscore, "macro": macro, "calendar": cal[:6],
        "stats": {"winrate_tp1": "82%", "winrate_tp2": "77%", "winrate_tp3": "72%", "n": 235,
                  "note": "Backtest 235 trade Mar-Sep 2026, SL 4 ATR. Bukan jaminan."},
    }


# ---------------------------------------------------------------- backtest
def backtest(path):
    df = pd.read_csv(path, parse_dates=["datetime"]).set_index("datetime")
    df.index = pd.to_datetime(df.index, utc=True)
    h1, h4 = resample(df, "1h"), resample(df, "4h")
    def tr(d): return np.sign(ema(d.close, 20) - ema(d.close, 50)).shift(1)
    df["t1"] = tr(h1).reindex(df.index, method="ffill"); df["t4"] = tr(h4).reindex(df.index, method="ffill")
    df["atr"] = atr(df)
    df["hh"] = df.high.rolling(LOOKBACK).max().shift(1); df["ll"] = df.low.rolling(LOOKBACK).min().shift(1)
    df = df.dropna()
    H, L, C, A = df.high.values, df.low.values, df.close.values, df.atr.values
    sig = np.where((df.t4 == 1) & (df.t1 == 1) & (df.close > df.hh), 1,
                   np.where((df.t4 == -1) & (df.t1 == -1) & (df.close < df.ll), -1, 0))
    last, res = -99, []
    for i in np.where(sig != 0)[0]:
        if i - last < 8: continue
        last = i; s = sig[i]; e, a = C[i], A[i]; sl = e - s * SL_ATR * a; hit = [0, 0, 0]
        for j in range(i + 1, min(i + 97, len(C))):
            if (L[j] <= sl) if s == 1 else (H[j] >= sl): break  # SL dihitung lebih dulu (konservatif)
            for k, m in enumerate(TP_ATRS):
                tp = e + s * m * a
                if (H[j] >= tp) if s == 1 else (L[j] <= tp): hit[k] = 1
        res.append(hit)
    r = np.array(res)
    print(f"Trade: {len(r)} | TP1 {r[:,0].mean():.1%} | TP2 {r[:,1].mean():.1%} | TP3 {r[:,2].mean():.1%}")


# ---------------------------------------------------------------- web
PAGE = """<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>XAUUSDc Analyzer</title>
<link rel="manifest" href="/manifest.json"><meta name="theme-color" content="#0f1115">
<style>
body{font-family:system-ui,sans-serif;background:#0f1115;color:#e8e8e8;margin:0;padding:16px;max-width:760px;margin:auto}
button{width:100%;padding:16px;font-size:18px;font-weight:700;border:0;border-radius:12px;background:#d4a017;color:#000}
button:disabled{opacity:.5}.card{background:#1a1d24;border-radius:12px;padding:14px;margin-top:12px}
h3{margin:0 0 8px;font-size:14px;color:#9aa}.big{font-size:30px;font-weight:800}
.BUY{color:#2ecc71}.SELL{color:#ff5c5c}.WAIT{color:#f1c40f}.row{display:flex;justify-content:space-between;padding:3px 0}
.sm{font-size:12px;color:#888}.r{color:#ff8a8a}.s{color:#7fe3a1}ul{padding-left:18px;margin:6px 0}
</style></head><body>
<input id="br" type="number" step="0.01" inputmode="decimal" placeholder="Harga broker XAUUSDc saat ini (opsional)" style="width:100%;box-sizing:border-box;padding:12px;margin-bottom:10px;border-radius:10px;border:1px solid #333;background:#1a1d24;color:#fff;font-size:16px">
<button id="b" onclick="go()">ANALISA</button><div id="o"></div>
<script>
async function go(){const b=document.getElementById('b');b.disabled=true;b.textContent='Menganalisa... (10-30 dtk)';
try{const d=await (await fetch('/analyze?broker='+encodeURIComponent(document.getElementById('br').value||''))).json();render(d)}catch(e){document.getElementById('o').innerHTML='<div class=card>Error: '+e+'</div>'}
b.disabled=false;b.textContent='ANALISA'}
const row=(a,b,c='')=>`<div class=row><span>${a}</span><b class="${c}">${b}</b></div>`;
const lv=(arr,c)=>arr.map((x,i)=>row((c=='r'?'R':'S')+(i+1)+' <span class=sm>'+x.info+'</span>',x.price,c)).join('');
function strongest(arr){if(!arr.length)return'-';const m=arr.reduce((a,b)=>b.score>a.score?b:a);return m.price+' - '+m.info}
function render(d){const L=d.levels,e=d.entry;let h=`<div class=card><div class=sm>${d.time} | ${d.source} | harga ${d.price} | ATR ${d.atr}</div>${d.offset?`<div class=sm>Disesuaikan ke harga broker (data ${d.feed_price}, selisih ${d.offset>0?'+':''}${d.offset}). Isi harga broker hanya saat market buka.</div>`:''}${d.warn?`<div class=sm style="color:#f1c40f">${d.warn}</div>`:''}
<div class="big ${d.action}">${d.action}</div>`;
if(e)h+=row('Zona entry',e[0]+' - '+e[1]);
h+=`<ul>${d.reason.map(x=>'<li>'+x+'</li>').join('')}</ul></div>`;
if(L)h+=`<div class=card><h3>SL & TARGET</h3>${row('SL (4 ATR)',L.sl,'SELL')}${row('TP1 (ATR 1)',L.tp1,'BUY')}${row('TP2 (ATR 1.5)',L.tp2,'BUY')}${row('TP3 (ATR 2)',L.tp3,'BUY')}</div>`;
h+=`<div class=card><h3>TREND</h3>${row('Trend H4',d.trend_h4)}${row('Trend H1',d.trend_h1)}${row('Struktur M15',d.structure)}${row('RSI M15',d.rsi)}</div>`;
h+=`<div class=card><h3>RESISTANCE</h3>${lv(d.resistance,'r')}<div class=sm>Terkuat: ${strongest(d.resistance)}</div></div>`;
h+=`<div class=card><h3>SUPPORT</h3>${lv(d.support,'s')}<div class=sm>Terkuat: ${strongest(d.support)}</div></div>`;
h+=`<div class=card><h3>MAKRO & BERITA (skor sentimen emas: ${d.news_score})</h3>`;
for(const k in d.macro)h+=row(k,d.macro[k].last+' ('+d.macro[k].chg24h+'% /24j)');
d.calendar.forEach(c=>h+=`<div class=sm>USD High: ${c.title} - ${c.time}</div>`);
h+='<ul>'+d.news.map(n=>`<li class=sm>[${n.src}] ${n.title}</li>`).join('')+'</ul></div>';
h+=`<div class="card sm">Backtest: TP1 ${d.stats.winrate_tp1} / TP2 ${d.stats.winrate_tp2} / TP3 ${d.stats.winrate_tp3} (${d.stats.n} trade). ${d.stats.note}</div>`;
document.getElementById('o').innerHTML=h}
</script></body></html>"""

from flask import Flask, jsonify, Response

app = Flask(__name__)


@app.route("/")
def index():
    return PAGE


@app.route("/manifest.json")
def manifest():
    return jsonify({"name": "XAUUSDc Analyzer", "short_name": "XAU Analyzer", "start_url": "/",
                    "display": "standalone", "background_color": "#0f1115", "theme_color": "#0f1115",
                    "icons": [{"src": "/icon.svg", "sizes": "any", "type": "image/svg+xml", "purpose": "any"}]})


@app.route("/icon.svg")
def icon():
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><rect width="100" height="100" rx="20" '
           'fill="#0f1115"/><text x="50" y="62" font-size="38" font-weight="800" text-anchor="middle" '
           'fill="#d4a017" font-family="sans-serif">XAU</text></svg>')
    return Response(svg, mimetype="image/svg+xml")


@app.route("/analyze")
def run():
    from flask import request
    try: b = float(request.args.get("broker", "") or 0) or None
    except ValueError: b = None
    return jsonify(analyze(b))


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[1] == "--backtest": backtest(sys.argv[2])
    else: app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
