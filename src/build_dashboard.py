"""Sesi 5 - HCI & Data Visualization: membangun dashboard HTML mandiri (tanpa dependensi).

Prinsip desain yang diterapkan: overview first -> filter -> details on demand;
palet Okabe-Ito (ramah buta warna); label langsung (bukan legenda terpisah);
KPI di atas; tooltip pada hover; kontras teks memadai; mendukung mode gelap.
"""
import json

import pandas as pd

from .config import DRIVER_COLOR, DRIVER_LABEL, DRIVERS, KABUPATEN, PRED_CSV, REPORT_DIR

TEMPLATE = r"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Dashboard Deforestasi Bengkulu (data sintetis)</title>
<style>
:root{--bg:#f7f7f5;--card:#fff;--tx:#1d1d1b;--mu:#6b6b66;--bd:#dcdcd6;--map:#eef2ee}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--card:#1f1f1f;--tx:#ececea;--mu:#9a9a94;--bd:#333;--map:#232823}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--tx);padding:16px;max-width:1200px;margin:auto}
h1{font-size:20px;margin:0 0 2px}.sub{color:var(--mu);margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:12px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin-bottom:12px}
.kpi b{display:block;font-size:22px}.kpi span{color:var(--mu);font-size:12px}
.grid{display:grid;grid-template-columns:1.6fr 1fr;gap:12px}@media(max-width:800px){.grid{grid-template-columns:1fr}}
.filters{display:flex;flex-wrap:wrap;gap:8px 16px;align-items:center;margin-bottom:12px}
select{padding:4px 8px;border-radius:6px;border:1px solid var(--bd);background:var(--card);color:var(--tx)}
label.chip{display:inline-flex;gap:6px;align-items:center;cursor:pointer;font-size:13px}
.dot{width:12px;height:12px;border-radius:50%;display:inline-block;border:1px solid #0004}
canvas{width:100%;height:auto;border-radius:8px;background:var(--map);display:block}
.bar{display:flex;align-items:center;gap:8px;margin:5px 0;font-size:13px}.bar .n{width:120px}.bar .t{flex:1;background:var(--bd);height:14px;border-radius:7px;overflow:hidden}
.bar .t i{display:block;height:100%}.bar .v{width:48px;text-align:right;color:var(--mu)}
#tip{position:fixed;pointer-events:none;background:#000d;color:#fff;padding:6px 8px;border-radius:6px;font-size:12px;display:none;z-index:9}
h2{font-size:14px;margin:0 0 8px}.note{color:var(--mu);font-size:12px;margin-top:10px}
</style></head><body>
<h1>Dashboard Alert Deforestasi - Bengkulu</h1>
<div class="sub">Studi kasus dengan <b>data sintetis</b>. Prediksi penyebab dari model out-of-fold (akurasi __ACC__).</div>
<div class="card filters" role="group" aria-label="Filter"><span id="chips"></span>
<label>Tahun <select id="year"><option value="">Semua</option></select></label>
<label>Kabupaten <select id="kab"><option value="">Semua</option></select></label>
<label>Tampilkan <select id="mode"><option value="obs">Label anotator</option><option value="pred">Prediksi model</option></select></label></div>
<div class="kpis" id="kpis"></div>
<div class="grid"><div class="card"><h2>Peta sebaran alert (hover untuk detail)</h2><canvas id="map" width="760" height="620"></canvas></div>
<div><div class="card" style="margin-bottom:12px"><h2>Luas kehilangan hutan per penyebab (ha)</h2><div id="bars"></div></div>
<div class="card"><h2>Tren bulanan jumlah alert</h2><svg id="trend" viewBox="0 0 360 150" width="100%"></svg></div></div></div>
<div class="note">Catatan: koordinat dan pola dibangkitkan secara sintetis untuk tujuan pembelajaran; bukan data resmi.</div>
<div id="tip"></div>
<script>
const D=__DATA__,DR=__DRIVERS__,K=__KABS__,KC=__KCENT__;
const css=getComputedStyle(document.documentElement);
const on=Object.fromEntries(DR.map(d=>[d.id,true]));
const $=id=>document.getElementById(id);
const years=[...new Set(D.map(r=>r.d.slice(0,4)))].sort();
years.forEach(y=>$('year').add(new Option(y,y)));K.forEach(k=>$('kab').add(new Option(k,k)));
DR.forEach(d=>{const l=document.createElement('label');l.className='chip';
l.innerHTML=`<input type=checkbox checked><span class=dot style="background:${d.c}"></span>${d.l}`;
l.firstChild.onchange=e=>{on[d.id]=e.target.checked;draw()};$('chips').appendChild(l)});
['year','kab','mode'].forEach(i=>$(i).onchange=draw);
const LAT=[-5.05,-2.3],LON=[100.9,103.9],cv=$('map'),cx=cv.getContext('2d');
const px=(lo,la)=>[(lo-LON[0])/(LON[1]-LON[0])*cv.width,(LAT[1]-la)/(LAT[1]-LAT[0])*cv.height];
let vis=[];
function draw(){
 const y=$('year').value,k=$('kab').value,m=$('mode').value;
 vis=D.filter(r=>(!y||r.d.startsWith(y))&&(!k||r.k===k)&&on[m==='obs'?r.o:r.p]).map(r=>({...r,g:m==='obs'?r.o:r.p}));
 const ha=vis.reduce((s,r)=>s+r.a,0),pr=vis.filter(r=>r.pr).length;
 const by={};vis.forEach(r=>by[r.g]=(by[r.g]||0)+r.a);
 const top=Object.entries(by).sort((a,b)=>b[1]-a[1])[0];
 $('kpis').innerHTML=[[vis.length.toLocaleString('id'),'Jumlah alert'],[Math.round(ha).toLocaleString('id')+' ha','Total luas'],
  [vis.length?Math.round(pr/vis.length*100)+'%':'-','Di kawasan lindung'],[top?DR.find(d=>d.id===top[0]).l:'-','Penyebab dominan (luas)']]
  .map(([v,l])=>`<div class="card kpi"><b>${v}</b><span>${l}</span></div>`).join('');
 cx.clearRect(0,0,cv.width,cv.height);
 vis.forEach(r=>{const[x,yy]=px(r.lo,r.la);cx.beginPath();cx.arc(x,yy,2.6,0,6.3);cx.fillStyle=DR.find(d=>d.id===r.g).c;cx.globalAlpha=.7;cx.fill()});
 cx.globalAlpha=1;cx.fillStyle=css.getPropertyValue('--mu');cx.font='13px system-ui';cx.textAlign='center';
 Object.entries(KC).forEach(([n,[la,lo]])=>{const[x,yy]=px(lo,la);cx.fillText(n,x,yy)});
 const mx=Math.max(1,...DR.map(d=>by[d.id]||0));
 $('bars').innerHTML=DR.map(d=>`<div class=bar><span class=n>${d.l}</span><span class=t><i style="width:${(by[d.id]||0)/mx*100}%;background:${d.c}"></i></span><span class=v>${Math.round(by[d.id]||0)}</span></div>`).join('');
 const mo={};vis.forEach(r=>{const q=r.d.slice(0,7);mo[q]=(mo[q]||0)+1});
 const yrs=y?[y]:years,ks=[];for(const yr of yrs)for(let i=1;i<=12;i++)ks.push(yr+'-'+String(i).padStart(2,'0'));
 const vals=ks.map(q=>mo[q]||0),M=Math.max(1,...vals);
 const pts=vals.map((v,i)=>`${30+i*(320/(ks.length-1))},${130-v/M*110}`).join(' ');
 $('trend').innerHTML=`<polyline points="${pts}" fill="none" stroke="#0072B2" stroke-width="2"/><line x1=30 y1=130 x2=350 y2=130 stroke="#8886"/>`+
 `<text x=2 y=22 font-size=10 fill="#888">${M}</text><text x=2 y=130 font-size=10 fill="#888">0</text>`+
 yrs.map((yr,i)=>`<text x=${30+i*12*(320/(ks.length-1))} y=146 font-size=10 fill="#888">${yr}</text>`).join('');
}
const tip=$('tip');
cv.onmousemove=e=>{const b=cv.getBoundingClientRect(),s=cv.width/b.width,mx=(e.clientX-b.left)*s,my=(e.clientY-b.top)*s;
 let best=null,bd=100;for(const r of vis){const[x,y]=px(r.lo,r.la),d=(x-mx)**2+(y-my)**2;if(d<bd){bd=d;best=r}}
 if(!best){tip.style.display='none';return}
 const L=id=>DR.find(d=>d.id===id).l;
 tip.innerHTML=`<b>${best.i}</b> - ${best.k}<br>${best.d} | ${best.a} ha<br>Label: ${L(best.o)}<br>Prediksi: ${L(best.p)}${best.pr?'<br>Kawasan lindung':''}`;
 tip.style.display='block';tip.style.left=e.clientX+12+'px';tip.style.top=e.clientY+12+'px'};
cv.onmouseleave=()=>tip.style.display='none';
draw();
</script></body></html>"""


def main():
    df = pd.read_csv(PRED_CSV)
    metrics = json.loads((REPORT_DIR / "metrics.json").read_text())
    data = [{"i": r.alert_id, "d": r.date, "la": r.lat, "lo": r.lon, "k": r.kabupaten, "a": round(r.area_ha, 1),
             "pr": int(r.in_protected), "o": r.driver, "p": r.predicted_driver} for r in df.itertuples()]
    drivers = [{"id": d, "l": DRIVER_LABEL[d], "c": DRIVER_COLOR[d]} for d in DRIVERS]
    html = (TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
            .replace("__DRIVERS__", json.dumps(drivers)).replace("__KABS__", json.dumps(sorted(KABUPATEN)))
            .replace("__KCENT__", json.dumps(KABUPATEN)).replace("__ACC__", f"{metrics['oof_accuracy']:.0%}"))
    out = REPORT_DIR / "dashboard.html"
    out.write_text(html, encoding="utf-8")
    print(f"Dashboard -> {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
