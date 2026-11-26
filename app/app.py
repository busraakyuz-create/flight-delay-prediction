"""Uçuş Gecikmesi Tahmini — Streamlit demosu.

Çalıştırma:  streamlit run app.py   (assets/ klasörü bu dosyanın yanında olmalı)
"""
import gzip
import json
from datetime import date, time
from pathlib import Path

import altair as alt
import lightgbm as lgb
import numpy as np
import pandas as pd
import streamlit as st
from pandas.tseries.holiday import USFederalHolidayCalendar

ASSETS = Path(__file__).parent / 'assets'
st.set_page_config(page_title='Uçuş Gecikmesi Tahmini', page_icon='✈️', layout='wide')

BLUE, ORANGE, INK2 = '#2a78d6', '#eb6834', '#52514e'
AIRLINES = {'WN': 'Southwest', 'DL': 'Delta', 'AA': 'American', 'OO': 'SkyWest', 'UA': 'United', 'YX': 'Republic',
            'MQ': 'Envoy', 'OH': 'PSA', 'AS': 'Alaska', 'B6': 'JetBlue', 'F9': 'Frontier', 'NK': 'Spirit',
            'G4': 'Allegiant', 'HA': 'Hawaiian', '9E': 'Endeavor', 'QX': 'Horizon', 'ZW': 'Air Wisconsin'}
WX = ['tmax_c', 'tmin_c', 'prcp_mm', 'snow_mm', 'snwd_mm', 'awnd_ms', 'wsf2_ms', 'fog', 'thunder', 'hail',
      'freezing', 'heavy_rain', 'snow_day']
GROUPS = {  # SHAP katkılarını okunur gruplara topluyoruz
    'Kalkış saati': ['dep_minute', 'dep_hour'], 'Varış saati': ['arr_hour'], 'Havayolu': ['airline'],
    'Kalkış havalimanı': ['origin'], 'Varış havalimanı': ['dest'], 'Haftanın günü': ['day_of_week'],
    'Tatile yakınlık': ['days_to_holiday', 'is_holiday'], 'Uçuş süresi / mesafe': ['crs_elapsed_min', 'distance_mi'],
    'Havalimanı / havayolu yoğunluğu': ['origin_sched_dep_hour', 'origin_sched_dep_day', 'dest_sched_arr_hour', 'airline_sched_day'],
    'Kalkışta hava durumu': [f'o_{c}' for c in WX], 'Varışta hava durumu': [f'd_{c}' for c in WX],
}


@st.cache_resource
def load():
    meta = json.loads((ASSETS / 'model_meta_final.json').read_text())
    # Model gzip ile sıkıştırılmış olarak saklanıyor (GitHub yükleme sınırı için); bellekte açılıyor
    model = lgb.Booster(model_str=gzip.decompress((ASSETS / 'lgbm_final.txt.gz').read_bytes()).decode())
    a = {n: pd.read_parquet(ASSETS / f'{n}.parquet') for n in
         ('airports', 'routes', 'weather', 'sched_origin_hour', 'sched_origin_day', 'sched_dest_hour', 'sched_airline_day')}
    a['weather']['date'] = pd.to_datetime(a['weather']['date']).dt.date
    for k in ('sched_origin_hour', 'sched_origin_day', 'sched_dest_hour', 'sched_airline_day'):
        a[k]['flight_date'] = pd.to_datetime(a[k]['flight_date']).dt.date
    hol = USFederalHolidayCalendar().holidays('2024-11-01', '2026-02-28')
    return meta, model, a, hol


meta, model, A, HOL = load()
FEATURES, CATS = meta['features'], meta['categories']
BASE_RATE = meta['base_rate_all']
ap = A['airports'].set_index('code')
routes = A['routes']


def ap_label(code):
    if code in ap.index:
        return f"{code} — {ap.at[code, 'city']}"
    return code


def lookup(df, keys, col='n', default=1):
    m = df
    for k, v in keys.items():
        m = m[m[k] == v]
    return int(m[col].iloc[0]) if len(m) else default


def build_rows(airline, origin, dest, d, dep_minutes, wx_o, wx_d, route):
    """Verilen uçuş için modele girecek satır(lar)ı kurar. dep_minutes: kalkış dakikaları listesi (saat senaryosu için)."""
    rows = []
    dtd = min(abs((pd.Timestamp(d) - h).days) for h in HOL)
    for dm in dep_minutes:
        dep_hour = dm // 60
        arr_hour = int(((dm + route['arr_offset_min']) % 1440) // 60)
        r = {
            'airline': airline, 'origin': origin, 'dest': dest, 'day_of_week': d.isoweekday(),
            'dep_hour': dep_hour, 'arr_hour': arr_hour, 'dep_minute': dm,
            'crs_elapsed_min': route['crs_elapsed_min'], 'distance_mi': route['distance_mi'],
            'origin_sched_dep_hour': lookup(A['sched_origin_hour'], {'origin': origin, 'flight_date': d, 'dep_hour': dep_hour}),
            'origin_sched_dep_day': lookup(A['sched_origin_day'], {'origin': origin, 'flight_date': d}),
            'dest_sched_arr_hour': lookup(A['sched_dest_hour'], {'dest': dest, 'flight_date': d, 'arr_hour': arr_hour}),
            'airline_sched_day': lookup(A['sched_airline_day'], {'airline': airline, 'flight_date': d}),
            'days_to_holiday': min(dtd, 14), 'is_holiday': int(dtd == 0),
        }
        r.update({f'o_{k}': v for k, v in wx_o.items()})
        r.update({f'd_{k}': v for k, v in wx_d.items()})
        rows.append(r)
    X = pd.DataFrame(rows)[FEATURES]
    for c in meta['categorical']:
        X[c] = pd.Categorical(X[c], categories=CATS[c])
    for c in X.columns:
        if c not in meta['categorical']:
            X[c] = X[c].astype('float32')
    return X


def real_weather(code, d):
    w = A['weather']
    m = w[(w['airport'] == code) & (w['date'] == d)]
    if not len(m):
        return {k: np.nan for k in WX}, False
    return {k: (float(m.iloc[0][k]) if pd.notna(m.iloc[0][k]) else np.nan) for k in WX}, True


def scenario_weather(prefix, defaults):
    c1, c2 = st.columns(2)
    tmax = c1.slider('En yüksek sıcaklık (°C)', -20, 45, int(defaults.get('tmax_c', 20) if pd.notna(defaults.get('tmax_c')) else 20), key=f'{prefix}tmax')
    tmin = c2.slider('En düşük sıcaklık (°C)', -30, 35, int(defaults.get('tmin_c', 10) if pd.notna(defaults.get('tmin_c')) else 10), key=f'{prefix}tmin')
    prcp = c1.slider('Yağış (mm)', 0, 100, 0, key=f'{prefix}prcp')
    snow = c2.slider('Yağan kar (cm)', 0, 30, 0, key=f'{prefix}snow')
    wind = c1.slider('Ortalama rüzgar (m/s)', 0, 20, 4, key=f'{prefix}wind')
    thunder = c2.checkbox('Gök gürültülü fırtına', key=f'{prefix}thunder')
    fog = c1.checkbox('Sis', key=f'{prefix}fog')
    hail = c2.checkbox('Dolu', key=f'{prefix}hail')
    snow_mm = snow * 10.0
    return {'tmax_c': float(tmax), 'tmin_c': float(min(tmin, tmax)), 'prcp_mm': float(prcp), 'snow_mm': snow_mm,
            'snwd_mm': snow_mm, 'awnd_ms': float(wind), 'wsf2_ms': float(wind) * 1.8, 'fog': float(fog),
            'thunder': float(thunder), 'hail': float(hail), 'freezing': float(min(tmin, tmax) <= 0),
            'heavy_rain': float(prcp >= 10), 'snow_day': float(snow_mm > 0)}


def describe_wx(w):
    if all(pd.isna(v) for v in w.values()):
        return 'veri yok'
    parts = [f"{w['tmin_c']:.0f}…{w['tmax_c']:.0f} °C"] if pd.notna(w['tmax_c']) and pd.notna(w['tmin_c']) else []
    if pd.notna(w['prcp_mm']) and w['prcp_mm'] > 0:
        parts.append(f"yağış {w['prcp_mm']:.0f} mm")
    if pd.notna(w['snow_mm']) and w['snow_mm'] > 0:
        parts.append(f"kar {w['snow_mm'] / 10:.1f} cm".replace('.', ','))
    if w.get('thunder') == 1:
        parts.append('fırtına')
    if w.get('fog') == 1:
        parts.append('sis')
    if pd.notna(w.get('awnd_ms')):
        parts.append(f"rüzgar {w['awnd_ms']:.0f} m/s")
    return ' · '.join(parts)


# ───────────────────────────── ARAYÜZ ─────────────────────────────
st.title('✈️ Uçuş Gecikmesi Tahmini')
st.caption('Bir uçuşun varışta **15 dakikadan fazla** gecikme olasılığını, kalkıştan önce bilinen bilgilerle tahmin eder. '
           'Model: LightGBM · Veri: ABD 2025 iç hat uçuşları (BTS, 7 milyon uçuş) + NOAA günlük hava durumu.')

with st.sidebar:
    st.header('Uçuş')
    al_opts = sorted(routes['airline'].unique(), key=lambda x: AIRLINES.get(x, x))
    airline = st.selectbox('Havayolu', al_opts, format_func=lambda x: f'{AIRLINES.get(x, x)} ({x})',
                           index=al_opts.index('AA') if 'AA' in al_opts else 0)
    r_al = routes[routes['airline'] == airline]
    o_opts = r_al.groupby('origin')['n'].sum().sort_values(ascending=False).index.tolist()
    origin = st.selectbox('Kalkış havalimanı', o_opts, format_func=ap_label)
    d_opts = r_al[r_al['origin'] == origin].sort_values('n', ascending=False)['dest'].tolist()
    dest = st.selectbox('Varış havalimanı', d_opts, format_func=ap_label)
    d = st.date_input('Tarih (2025)', value=date(2025, 7, 18), min_value=date(2025, 1, 1), max_value=date(2025, 12, 31), format='DD.MM.YYYY')
    TIMES = [time(h, m) for h in range(5, 24) for m in (0, 15, 30, 45)]
    t = st.selectbox('Planlanan kalkış saati (yerel)', TIMES, index=TIMES.index(time(18, 0)), format_func=lambda x: x.strftime('%H:%M'))
    st.header('Hava durumu')
    mode = st.radio('Kaynak', ['O günün gerçek hava durumu (NOAA)', 'Senaryo: kendim ayarlayacağım'], index=0)

route = r_al[(r_al['origin'] == origin) & (r_al['dest'] == dest)].iloc[0]
wo_real, ok_o = real_weather(origin, d)
wd_real, ok_d = real_weather(dest, d)
if mode.startswith('Senaryo'):
    with st.sidebar.expander(f'Kalkış: {origin}', expanded=True):
        wx_o = scenario_weather('o_', wo_real)
    with st.sidebar.expander(f'Varış: {dest}', expanded=False):
        wx_d = scenario_weather('d_', wd_real)
else:
    wx_o, wx_d = wo_real, wd_real

dep_min = t.hour * 60 + t.minute
X = build_rows(airline, origin, dest, d, [dep_min], wx_o, wx_d, route)
p = float(model.predict(X)[0])
level, color = (('Düşük', BLUE) if p < 0.15 else ('Orta', '#b07a00') if p < 0.30 else ('Yüksek', ORANGE))

c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    st.markdown(f"<div style='font-size:15px;color:{INK2}'>15 dk+ gecikme olasılığı</div>"
                f"<div style='font-size:64px;font-weight:700;line-height:1.1'>%{p * 100:.0f}</div>"
                f"<div style='font-size:20px;font-weight:600;color:{color}'>Risk: {level}</div>", unsafe_allow_html=True)
with c2:
    st.metric('Ortalama uçuşa göre', f'{p / BASE_RATE:.1f} kat'.replace('.', ','), help=f'2025 ortalaması: %{BASE_RATE * 100:.1f}')
    st.metric('Modelin uyarı eşiği', f"%{meta['threshold'] * 100:.0f}",
              delta='uyarı verilir' if p >= meta['threshold'] else 'uyarı yok', delta_color='inverse' if p >= meta['threshold'] else 'off')
with c3:
    st.metric(f'Bu rotada 2025 gerçek oranı ({airline})', f"%{route['delay_rate'] * 100:.0f}", help=f"{int(route['n']):,} uçuş")
    st.caption(f"**Kalkış ({origin}):** {describe_wx(wx_o)}  \n**Varış ({dest}):** {describe_wx(wx_d)}")
    if not mode.startswith('Senaryo') and not (ok_o and ok_d):
        st.caption('⚠️ Bu havalimanı/gün için hava durumu verisi yok; model bu bilgiyi "bilinmiyor" sayıyor.')

st.divider()
left, right = st.columns(2)

with left:
    st.subheader('Model bu kararı neden verdi?')
    contrib = model.predict(X, pred_contrib=True)[0]
    cs = pd.Series(contrib[:-1], index=FEATURES)
    g = pd.Series({k: cs[v].sum() for k, v in GROUPS.items()}).sort_values(key=np.abs, ascending=False).head(7)
    gdf = pd.DataFrame({'etken': g.index, 'katki': g.values, 'yon': np.where(g.values > 0, 'Riski artırıyor', 'Riski azaltıyor')})
    chart = alt.Chart(gdf).mark_bar(cornerRadius=3).encode(
        x=alt.X('katki:Q', title='Etki (log-odds; sağ = daha riskli)'),
        y=alt.Y('etken:N', sort=list(gdf['etken']), title=None, axis=alt.Axis(labelLimit=260)),
        color=alt.Color('yon:N', scale=alt.Scale(domain=['Riski artırıyor', 'Riski azaltıyor'], range=[ORANGE, BLUE]),
                        legend=alt.Legend(title=None, orient='bottom')),
        tooltip=[alt.Tooltip('etken:N', title='Etken'), alt.Tooltip('katki:Q', title='Etki', format='+.2f')],
    ).properties(height=alt.Step(34))
    st.altair_chart(chart, width='stretch')
    st.caption('SHAP katkıları: her etkenin bu uçuşun tahminini ortalama bir uçuşa göre ne kadar yukarı/aşağı ittiğini gösterir.')

with right:
    st.subheader('Kalkış saatini değiştirsem?')
    hours = list(range(5, 24))
    Xh = build_rows(airline, origin, dest, d, [h * 60 for h in hours], wx_o, wx_d, route)
    ph = model.predict(Xh)
    hdf = pd.DataFrame({'saat': [f'{h:02d}:00' for h in hours], 'olasilik': ph,
                        'secili': ['Seçili saat' if h == t.hour else 'Diğer saatler' for h in hours]})
    bar = alt.Chart(hdf).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=alt.X('saat:N', title='Planlanan kalkış saati', sort=None),
        y=alt.Y('olasilik:Q', title='Olasılık', axis=alt.Axis(format='%')),
        color=alt.Color('secili:N', scale=alt.Scale(domain=['Seçili saat', 'Diğer saatler'], range=[ORANGE, BLUE]),
                        legend=alt.Legend(title=None, orient='bottom')),
        tooltip=[alt.Tooltip('saat:N', title='Saat'), alt.Tooltip('olasilik:Q', title='Olasılık', format='.0%')],
    ).properties(height=300)
    st.altair_chart(bar, width='stretch')
    best = hdf.loc[hdf['olasilik'].idxmin()]
    st.caption(f"Aynı gün ve rotada en düşük risk **{best['saat']}** kalkışında: **%{best['olasilik'] * 100:.0f}**.")

with st.expander('Model hakkında'):
    m = meta['test_metrics']
    st.markdown(f"""
- **Soru:** Uçuş varışta 15 dakikadan fazla gecikecek mi? (ikili sınıflandırma)
- **Veri:** BTS 2025 — 7.001.619 uçuş, 14 havayolu, 352 havalimanı · NOAA GHCN-Daily hava durumu (kalkış ve varış havalimanı)
- **Değerlendirme (Ocak–Eylül ile eğitilip Kasım–Aralık'ta test edildi):** ROC-AUC **{m['ROC-AUC (test)']:.3f}**, PR-AUC **{m['PR-AUC (test)']:.3f}** (rastgele: {meta['base_rate_test']:.3f})
- En riskli %10'luk uçuşların **%57'si** gerçekten gecikiyor (ortalama %24).
- **Bu demodaki model:** aynı ayarlarla 2025'in tamamıyla yeniden eğitilmiş son sürüm.
- **Sınırlamalar:** gerçekleşen hava durumu kullanılıyor (gerçek hayatta hava tahmini kullanılmalı); uçağın önceki seferinin rötarı bilinmiyor; 2025 dışındaki tarihler desteklenmiyor.
""")
