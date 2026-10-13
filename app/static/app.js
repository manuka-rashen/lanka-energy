'use strict';
const $ = id => document.getElementById(id);
let examples = [], exampleIndex = 0, latestResult = null;
const number = (value, digits = 1) => Number(value).toLocaleString('en-US', {maximumFractionDigits: digits});
const text = (id, value) => { $(id).textContent = value; };
async function api(path, options = {}) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) {
    const detail = typeof data.detail === 'string' ? data.detail : 'Check your inputs: six readings, a valid historical month, and non-negative appliance counts.';
    throw new Error(detail);
  }
  return data;
}
function profileEnabled() {
  $('profile-fields').hidden = !$('use-profile').checked;
  $('profile-fields').querySelectorAll('input').forEach(input => { input.disabled = !$('use-profile').checked; input.required = $('use-profile').checked; });
}
$('use-profile').addEventListener('change', profileEnabled);
profileEnabled();
function requestData() {
  return {history: [6,5,4,3,2,1].map(i => Number($(`reading-${i}`).value)),
    forecast_month: $('forecast-month').value,
    consumption_limit: Number($('consumption-limit').value),
    profile: $('use-profile').checked ? {
      household_members: Number($('members').value), electricity_meters: Number($('meters').value),
      refrigerators: Number($('fridges').value), washing_machines: Number($('washers').value),
      air_conditioners: Number($('acs').value), rice_cookers: Number($('rice').value)} : null};
}
$('forecast-form').addEventListener('submit', async event => {
  event.preventDefault();
  $('form-error').hidden = true;
  $('forecast-button').disabled = true;
  $('forecast-button').firstElementChild.textContent = 'Creating your forecast…';
  try {
    const input = requestData();
    const result = await api('/api/forecast', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(input)});
    latestResult = {input, result};
    showResult(input, result);
  } catch (error) { text('form-error', error.message); $('form-error').hidden = false; }
  finally { $('forecast-button').disabled = false; $('forecast-button').firstElementChild.textContent = 'Forecast consumption'; }
});
function showResult(input, result) {
  $('result-empty').hidden = true; $('result-content').hidden = false;
  text('result-month', new Date(result.forecast_month+'-01T12:00:00').toLocaleDateString('en-GB',{month:'long',year:'numeric'}));
  text('prediction-value', number(result.prediction_kwh));
  const change = result.change_from_last_month_percent;
  text('change-text', change === null ? 'Change percentage unavailable because the last reading was zero.' : `${change >= 0 ? '+' : ''}${number(change)}% compared with your last monthly reading`);
  text('interval-value', `${number(result.lower_kwh)}–${number(result.upper_kwh)} kWh`);
  const labels = {within:'Within your consumption limit', possible:'Higher usage is possible', above:'Estimate above your limit', likely_above:'Range above your limit'};
  text('alert-title', labels[result.alert.level]); text('alert-message', result.alert.message);
  $('alert-box').className = 'alert-box' + (result.alert.level === 'possible' ? ' caution' : ['above','likely_above'].includes(result.alert.level) ? ' above' : '');
  $('factors').replaceChildren();
  result.explanation.factors.slice(0,5).forEach(factor => {
    const row = document.createElement('div'); row.className = 'factor';
    const label = document.createElement('span'); label.className = 'factor-name'; label.textContent = factor.factor;
    const value = document.createElement('span'); value.className = 'factor-value'; value.textContent = `${factor.contribution_kwh >= 0 ? '+' : ''}${number(factor.contribution_kwh)} kWh`;
    row.append(label,value); $('factors').append(row);
  });
  if (result.explanation.reference_kwh !== undefined) {
    text('explanation-reference', `All contributions add to a ${number(result.explanation.reference_kwh)} kWh model reference to form the forecast. Only the five largest factors are shown.`);
  } else { text('explanation-reference', result.explanation.method); }
  text('model-label', `${result.model} · v${result.model_version}`);
  drawChart(input.history, result);
}
function drawChart(history, result) {
  const svg = $('usage-chart'); svg.replaceChildren();
  const NS = 'http://www.w3.org/2000/svg';
  const el = (name, attrs, value) => { const node=document.createElementNS(NS,name); Object.entries(attrs).forEach(([k,v])=>node.setAttribute(k,String(v))); if(value!==undefined)node.textContent=value; svg.append(node); return node; };
  const ceiling = Math.max(...history, result.upper_kwh, 20)*1.12;
  const x = i => 35+i*62, y = value => 135-value/ceiling*110;
  for (let i=0;i<3;i++) { const value=ceiling*i/2; el('line',{x1:35,y1:y(value),x2:408,y2:y(value),stroke:'#e7ece3'}); el('text',{x:0,y:y(value)+3,fill:'#9ba69a','font-size':8},number(value,0)); }
  const values = [...history,result.prediction_kwh];
  el('rect',{x:x(6)-12,y:y(result.upper_kwh),width:24,height:Math.max(2,y(result.lower_kwh)-y(result.upper_kwh)),fill:'#efe5ca',rx:4});
  el('polyline',{points:history.map((v,i)=>`${x(i)},${y(v)}`).join(' '),fill:'none',stroke:'#137f71','stroke-width':2.5});
  el('line',{x1:x(5),y1:y(history[5]),x2:x(6),y2:y(result.prediction_kwh),stroke:'#b8953f','stroke-width':2,'stroke-dasharray':'4 4'});
  values.forEach((value,i) => {el('circle',{cx:x(i),cy:y(value),r:i===6?5:3,fill:i===6?'#c4a24e':'#137f71'}); el('text',{x:x(i),y:159,'text-anchor':'middle',fill:'#919e8f','font-size':8},i===6?'Forecast':i===5?'Latest':`${6-i} mo ago`);});
}
$('load-example').addEventListener('click', () => {
  if (!examples.length) return;
  const data = examples[exampleIndex++ % examples.length];
  data.history.forEach((value,i)=>{$(`reading-${6-i}`).value=value;});
  $('forecast-month').value=data.forecast_month;
  $('consumption-limit').value=data.consumption_limit;
  $('use-profile').checked=Boolean(data.profile); profileEnabled();
  if(data.profile){ const map={members:'household_members',meters:'electricity_meters',fridges:'refrigerators',washers:'washing_machines',acs:'air_conditioners',rice:'rice_cookers'}; Object.entries(map).forEach(([id,key])=>{$(id).value=data.profile[key] ?? 0;}); }
  $('forecast-form').requestSubmit();
});
$('download-result').addEventListener('click',()=>{
  if(!latestResult)return;
  const blob=new Blob([JSON.stringify(latestResult,null,2)],{type:'application/json'});
  const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='lanka-energy-forecast.json';a.click();URL.revokeObjectURL(url);
});
async function initialize() {
  try {
    const [metadata, demo] = await Promise.all([api('/api/metadata'),api('/api/examples')]);
    examples=demo;
    text('mae', `${number(metadata.test.mae_kwh)} kWh`);
    const improvement=100*(metadata.test.last_month_baseline.mae_kwh-metadata.test.mae_kwh)/metadata.test.last_month_baseline.mae_kwh;
    text('improvement', `${improvement>=0?'+':''}${number(improvement)}%`);
    text('coverage',`${number(metadata.test.interval_coverage_percent)}%`);
    text('unseen-mae',`${number(metadata.unseen_households.mae_kwh)} kWh`);
    text('split-summary',`Training through March 2024; validation in April–June; final model fitted through June; interval calibration in July. Later-month test: ${number(metadata.test.rows,0)} observations. New-household test: ${number(metadata.unseen_households.rows,0)} observations.`);
    metadata.candidates.forEach(candidate=>{const tr=document.createElement('tr');[candidate.name,number(candidate.validation.mae_kwh,2),candidate.name===metadata.model?'✓ Selected':''].forEach(value=>{const td=document.createElement('td');td.textContent=value;tr.append(td);});$('candidate-table').append(tr);});
  } catch (error) { text('form-error','The model could not be loaded. '+error.message); $('form-error').hidden=false; $('forecast-button').disabled=true; }
}
initialize();
