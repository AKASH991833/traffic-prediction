/* FlowCast - seasonal time-series congestion forecasting on simulated sensor data. */
(()=>{
const $=s=>document.querySelector(s);
function mulberry(a){return()=>{a|=0;a=a+0x6D2B79F5|0;let t=Math.imul(a^a>>>15,1|a);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296}}
const DAYS=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'];
const WEEKS=12,HOURS=24;
// congestion index 0..1 per corridor/day/hour, with weekly patterns + incidents
function genCorridor(seed,profile){
  const r=mulberry(seed),data=[]; // [week][day][hour]
  for(let w=0;w<WEEKS;w++){const wk=[];
    const incidentDay=Math.floor(r()*7),incidentHour=Math.floor(6+r()*12),hasIncident=r()<.4;
    const rainWeek=r()<.25;
    for(let d=0;d<7;d++){const hrs=[];const weekend=d>=5;
      for(let h=0;h<24;h++){
        let c=0.06;
        if(!weekend){
          c+=profile.am*Math.exp(-((h-8.2)**2)/(2*1.1**2));
          c+=profile.pm*Math.exp(-((h-17.6)**2)/(2*1.4**2));
          c+=profile.mid*Math.exp(-((h-12.5)**2)/(2*1.8**2));
        }else{
          c+=profile.we*Math.exp(-((h-13)**2)/(2*2.6**2));
        }
        if(rainWeek)c*=1.12;
        if(hasIncident&&d===incidentDay&&h>=incidentHour&&h<incidentHour+2)c=Math.min(1,c+0.45*(1-(h-incidentHour)/2.2));
        c+= (r()*2-1)*0.035;
        hrs.push(Math.max(0.02,Math.min(1,c)));
      }
      wk.push(hrs);}
    data.push(wk);}
  return data;
}
const CORRIDORS=[
  {data:genCorridor(101,{am:.62,pm:.78,mid:.28,we:.30})},
  {data:genCorridor(202,{am:.35,pm:.55,mid:.18,we:.42})},
  {data:genCorridor(303,{am:.50,pm:.44,mid:.52,we:.48})},
  {data:genCorridor(404,{am:.58,pm:.40,mid:.46,we:.22})},
];
// model: per (day,hour) seasonal average + recent-week trend (damped), features real
function fitModel(hist){ // hist: [week][day][hour], weeks 0..n-1
  const n=hist.length,seasonal=[],trend=[];
  for(let d=0;d<7;d++){seasonal[d]=[];for(let h=0;h<24;h++){
    let s=0;for(let w=0;w<n;w++)s+=hist[w][d][h];seasonal[d][h]=s/n;}}
  for(let d=0;d<7;d++){trend[d]=[];for(let h=0;h<24;h++){
    const recent=(hist[n-1][d][h]+hist[n-2]?.[d][h]??hist[n-1][d][h])/2;
    const older=(hist[Math.max(0,n-4)][d][h]+hist[Math.max(0,n-3)][d][h])/2;
    trend[d][h]=(recent-older)*0.5;}}
  return{seasonal,trend};
}
function predict(model,d,h){return Math.max(0.02,Math.min(1,model.seasonal[d][h]+model.trend[d][h]));}
// backtest across corridors
let mape=0,mapeN=0;
const t0=performance.now();
CORRIDORS.forEach(c=>{
  const hist=c.data.slice(0,11),m=fitModel(hist);
  for(let d=0;d<7;d++)for(let h=0;h<24;h++){
    const p=predict(m,d,h),a=c.data[11][d][h];
    mape+=Math.abs(p-a)/Math.max(a,0.05);mapeN++;}});
mape/=mapeN;
const trainMs=Math.max(1,Math.round(performance.now()-t0));
$('#m-mape').textContent=(mape*100).toFixed(1)+'%';
$('#m-n').textContent=11*7*24;
$('#m-ms').textContent=trainMs+' ms';
const daySel=$('#day');DAYS.forEach((d,i)=>{const o=document.createElement('option');o.value=i;o.textContent=d;daySel.append(o);});
daySel.value=String((new Date().getDay()+6)%7);
function current(){return{ci:+$('#corridor').value,d:+daySel.value,trip:+$('#tripmin').value};}
function render(){
  const{ci,d,trip}=current();
  const c=CORRIDORS[ci],model=fitModel(c.data);
  const pred=Array.from({length:24},(_,h)=>predict(model,d,h));
  const actual=c.data[WEEKS-1][d];
  $('#chartTitle').textContent=DAYS[d];
  // best windows: minimize avg congestion over trip window (in hours, .25 steps)
  const hrs=trip/60,wins=[];
  for(let s=5;s<=22;s+=0.25){
    let sum=0;for(let h=s;h<s+hrs;h+=0.25){const hh=Math.floor(h)%24;sum+=pred[hh];}
    wins.push({s,avg:sum/(hrs*4)});
  }
  wins.sort((a,b)=>a.avg-b.avg);
  const fmt=t=>{const h=Math.floor(t),m=Math.round((t-h)*60);const ap=h>=12?'PM':'AM';const h12=((h+11)%12)+1;return h12+':'+String(m).padStart(2,'0')+' '+ap;};
  const dayW=wins.filter(x=>x.s>=5&&x.s<=21.25);
  const b=dayW[0]||wins[0];
  const bands=[['Morning',5,11],['Midday',11,16],['Evening',16,21.25]].map(([n,lo,hi])=>{
    const bw=dayW.filter(x=>x.s>=lo&&x.s<hi).sort((a,c)=>a.avg-c.avg)[0];
    return bw?n+': '+fmt(bw.s)+' - '+fmt(bw.s+hrs):null;}).filter(Boolean).join('  ·  ');
  const rush=dayW[Math.floor(dayW.length*0.92)]||wins[wins.length-1];
  $('#best').textContent=fmt(b.s)+' - '+fmt(b.s+hrs);
  $('#save').textContent='vs worst window: '+Math.round((1-b.avg/rush.avg)*100)+'% less congestion, ~'+Math.max(1,Math.round((rush.avg-b.avg)*trip*0.8))+' min saved';
  $('#bands')?.remove();
  $('#save').insertAdjacentHTML('afterend','<p id="bands" style="font-size:12.5px;color:var(--muted);margin-top:10px">'+bands+'</p>');
  drawChart(pred,actual);
  drawHeat(c,model);
}
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function drawChart(pred,actual){
  const cv=$('#dayChart'),dpr=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
  cv.width=w*dpr;cv.height=h*dpr;const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
  const L=34,R=w-8,T=10,B=h-20;
  const sx=i=>L+i/23*(R-L),sy=v=>B-v*(B-T);
  ctx.strokeStyle=css('--border');ctx.fillStyle=css('--muted');ctx.font='10px system-ui';
  [0,.25,.5,.75,1].forEach(g=>{ctx.beginPath();ctx.moveTo(L,sy(g));ctx.lineTo(R,sy(g));ctx.stroke();ctx.fillText(Math.round(g*100)+'%',6,sy(g)+3);});
  [0,4,8,12,16,20].forEach(hh=>ctx.fillText(hh+':00',sx(hh)-10,h-6));
  // threshold
  ctx.setLineDash([5,5]);ctx.strokeStyle='#f59e0b';ctx.lineWidth=1.4;
  ctx.beginPath();ctx.moveTo(L,sy(.6));ctx.lineTo(R,sy(.6));ctx.stroke();ctx.setLineDash([]);
  // actual (same weekday, latest week)
  ctx.strokeStyle='#8a97ad';ctx.lineWidth=1.6;ctx.beginPath();
  actual.forEach((v,i)=>i?ctx.lineTo(sx(i),sy(v)):ctx.moveTo(sx(i),sy(v)));ctx.stroke();
  // predicted area+line
  ctx.fillStyle=css('--accent');ctx.globalAlpha=.12;ctx.beginPath();
  ctx.moveTo(sx(0),sy(pred[0]));pred.forEach((v,i)=>ctx.lineTo(sx(i),sy(v)));
  ctx.lineTo(R,B);ctx.lineTo(L,B);ctx.closePath();ctx.fill();ctx.globalAlpha=1;
  ctx.strokeStyle=css('--accent');ctx.lineWidth=2.4;ctx.beginPath();
  pred.forEach((v,i)=>i?ctx.lineTo(sx(i),sy(v)):ctx.moveTo(sx(i),sy(v)));ctx.stroke();
}
function drawHeat(c,model){
  const el=$('#heatmap');
  let html='<table class="hmap"><tr><th></th>';
  for(let h=0;h<24;h++)html+=`<th>${h}</th>`;
  html+='</tr>';
  for(let d=0;d<7;d++){html+=`<tr><th style="text-align:left;padding-right:6px">${DAYS[d].slice(0,3)}</th>`;
    for(let h=0;h<24;h++){const v=predict(model,d,h);
      const col=v<.25?'#dcfce7':v<.45?'#fde68a':v<.65?'#fca5a5':'#dc2626';
      const fg=v<.25?'#166534':v<.45?'#92400e':v<.65?'#991b1b':'#fff';
      html+=`<td><div class="hcell" style="background:${col};color:${fg}" title="${DAYS[d]} ${h}:00 - ${(v*100).toFixed(0)}% congested">${Math.round(v*100)}</div></td>`;}
    html+='</tr>';}
  el.innerHTML=html+'</table>';
}
['corridor','day','tripmin'].forEach(id=>$('#'+id).onchange=render);
window.addEventListener('themechange',render);
let rz;window.addEventListener('resize',()=>{clearTimeout(rz);rz=setTimeout(render,150);});
render();
})();
