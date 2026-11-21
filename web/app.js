import {fields, getValue, setValue} from './config-fields.js';
import {mapNavigation} from './map-navigation.js';
const $ = id => document.getElementById(id);
let scenario, config, worker, busy=false, latest=null, completed=null, timer;
let scenarioPath, choices=[], profiles=[], activeProfile, loading=false;
for(const section of [...new Set(fields.map(f=>f.section))]) {
 const group=document.createElement('fieldset');const title=document.createElement('legend');title.textContent=section;group.append(title);
 for(const field of fields.filter(f=>f.section===section)){
  const container=document.createElement('div');container.className='config-field';
  const row=document.createElement('div');row.className='config-label-row';
  const label=document.createElement('label');label.htmlFor=field.id;label.textContent=field.label;
  row.append(label);
  if(field.help){
    const help=document.createElement('a');help.className='help-button';help.textContent='?';
    help.setAttribute('href',`guide.html#${field.id}`);help.setAttribute('target','_blank');help.setAttribute('rel','noopener');
    help.setAttribute('aria-label',`Help for ${field.label} (opens in a new tab)`);help.setAttribute('title','Explain this parameter (opens in a new tab)');
    row.append(help);
  }
  const input=document.createElement('input');Object.assign(input,{id:field.id,type:'number',min:field.min,step:field.step});if(field.max!==undefined)input.max=field.max;
  container.append(row,input);group.append(container);
 }
 $('common-configs').append(group);
}
const navigation=mapNavigation($('map'));
$('zoom-in').onclick=()=>navigation.zoomIn();
$('zoom-out').onclick=()=>navigation.zoomOut();
$('map-reset').onclick=()=>navigation.fit();
const ns='http://www.w3.org/2000/svg';
function svg(tag, attrs, parent=$('map')) {const el=document.createElementNS(ns,tag); for(const [k,v] of Object.entries(attrs)) el.setAttribute(k,v); parent.append(el); return el;}
function draw(network) {
  const [xmin,xmax,ymin,ymax]=scenario.viewport, width=xmax-xmin,height=ymax-ymin;
  $('map').replaceChildren();
  svg('image',{href:scenarioPath+'background.png',width,height});
  const xy=p=>[p[0]-xmin,ymax-p[1]];
  for(const f of network?.features || []) if(f.geometry.type==='LineString') {
    const p=f.properties, points=scenario.paths[`${p.edge_id}:${p.track_type}`];
    const line=svg('polyline',{points:points.map(p=>xy(p).join(',')).join(' '),fill:'none',stroke:p.track_type==='HSR'?'#d45e37':'#147d79','stroke-width':2.5,'vector-effect':'non-scaling-stroke'});
    svg('title',{},line).textContent=`${p.track_type} · ${p.length_km.toFixed(1)} km · ${p.frequency_per_hour_direction.toFixed(2)} trains/h`;
  }
  scenario.nodes.forEach((point,i)=>{const [cx,cy]=xy(point);const c=svg('circle',{cx,cy,r:width/180,fill:'#143d50',stroke:'white','stroke-width':1,'vector-effect':'non-scaling-stroke'}); svg('title',{},c).textContent=scenario.cities[i]; const label=svg('text',{x:cx+width/100,y:cy-width/130,'font-size':width/65,fill:'#143d50'});label.textContent=scenario.cities[i];});
}
function status(message){$('status').textContent=message;}
function controls(){ $('run').disabled=busy||loading||!config;$('stop').disabled=!busy;$('cities').disabled=loading||!config;$('profile').disabled=loading||!config; }
function readConfig(){
 const c=JSON.parse($('config').value);
 if(!c || Array.isArray(c)||typeof c!=='object') throw Error('Config must be a JSON object.');
 for(const field of fields)setValue(c,field,$(field.id).value);
 if(c.prep.max_cities!==Number($('cities').value))throw Error('Use the number of cities selector to load a matching prepared scenario.');
 if(c.operations.target_load_factor>c.operations.maximum_load_factor)throw Error('Target load factor must not exceed maximum load factor.');
 c.search.workers=1;return c;
}
function cancel(){clearTimeout(timer);worker?.terminate();worker=null;busy=false;controls();status('Stopped. Latest output retained.');}
function run(){clearTimeout(timer);if(busy)cancel();try{config=readConfig();$('config').value=JSON.stringify(config,null,2);}catch(e){status(e.message);return;}busy=true;completed=null;$('log').textContent='';controls();status('Starting Python worker…');worker=new Worker('worker.js');worker.onerror=e=>{status(`Worker failed: ${e.message}`);worker?.terminate();busy=false;controls();};worker.onmessage=({data})=>{if(typeof data==='string')data=JSON.parse(data);if(data.type==='status')status(data.message);if(data.type==='log'){$('log').textContent=($('log').textContent+data.text+'\n').slice(-40000);}if(data.type==='update'){latest=data;draw(data.network);$('iteration').textContent=`Update ${data.iteration} · ${data.phase}`;const m=data.metrics;$('metrics').replaceChildren();for(const [label,value] of [['Annual trips',(m.passengers/1e6).toFixed(2)+'M'],['Active stations',m.active_stations],['Track length',m.track_km.toFixed(0)+' km'],['Upfront capital','€'+(m.upfront_eur/1e6).toFixed(0)+'M'],['Annual net','€'+(m.net_eur/1e6).toFixed(1)+'M'],['Need satisfaction',(m.need_satisfaction_ratio*100).toFixed(1)+'%']]){const box=document.createElement('div');box.className='metric';const strong=document.createElement('strong');strong.textContent=value;const span=document.createElement('span');span.textContent=label;box.append(strong,span);$('metrics').append(box);}$('download').disabled=false;status(`Running · ${data.seconds.toFixed(1)} seconds`);}if(data.type==='done'||data.type==='error'){busy=false;controls();if(data.type==='done'){completed=data;status(`Finished · ${data.status} · ${data.evaluations} evaluations`);}else status(data.message);worker.terminate();worker=null;}};worker.postMessage({config,scenarioPath});}
function schedule(){clearTimeout(timer);status('Settings changed. Run to update the network.');if($('auto').checked)timer=setTimeout(run,900);}
function download(name,value){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
$('run').onclick=run;$('stop').onclick=cancel;
for(const field of fields)$(field.id).addEventListener('input',()=>{try{const c=readConfig();$('config').value=JSON.stringify(c,null,2);schedule();}catch(e){clearTimeout(timer);status(e.message);}});
$('config').addEventListener('input',()=>{try{apply();}catch{}clearTimeout(timer);if($('auto').checked)timer=setTimeout(()=>{try{apply();run();}catch(e){status(e.message);}},1200);});
function apply(){const c=JSON.parse($('config').value);for(const field of fields)$(field.id).value=getValue(c,field);config=c;}
$('apply').onclick=()=>{try{apply();schedule();}catch(e){status(e.message);}};
$('download').onclick=()=>download('trainline-results.json',{...completed,latest});
$('export-config').onclick=()=>{try{download('trainline-config.json',readConfig());}catch(e){status(e.message);}};
async function loadScenario(choice, {preserve=false, profile=activeProfile, profileChoices=choices}={}) {
 let edits;
 if(preserve)edits=readConfig();
 if(busy)cancel();
 clearTimeout(timer);loading=true;controls();status('Loading prepared city routes…');
 try {
  const path=`scenarios/${profile.path}/${choice.path==='.'?'':choice.path+'/'}`;
  const response=await fetch(path+'scenario.json');if(!response.ok)throw Error(`HTTP ${response.status}`);
  const next=await response.json();
  if(edits){edits.prep.max_cities=next.config.prep.max_cities;config=edits;}else config=structuredClone(next.config);
  navigation.reset(next.viewport[1]-next.viewport[0],next.viewport[3]-next.viewport[2]);
  scenario=next;scenarioPath=path;activeProfile=profile;choices=profileChoices;
  $('profile').value=profile.id;$('cities').replaceChildren();
  for(const choice of choices){const option=document.createElement('option');option.value=choice.city_count;option.textContent=choice.actual_city_count<choice.city_count?`${choice.city_count} city limit (${choice.actual_city_count} available)`:`${choice.city_count} cities`;$('cities').append(option);}
  $('cities').value=choice.city_count;
  $('config').value=JSON.stringify(config,null,2);apply();latest=null;completed=null;
  $('download').disabled=true;$('metrics').textContent='Run the optimizer to see network metrics.';
  $('log').textContent='';$('iteration').textContent='Ready to explore';draw();
  $('scenario-description').textContent=`${profile.label} · ${scenario.cities.length} cities${scenario.cities.length<choice.city_count?` (limit ${choice.city_count})`:''} · prepared terrain routes`;
  status(`Ready · ${profile.label} · ${scenario.cities.length} cities · ${scenario.terrain} terrain prepared offline.`);
  $('export-config').disabled=false;
 } finally {loading=false;controls();}
}
async function loadProfile(profile) {
 if(busy)cancel();
 clearTimeout(timer);loading=true;controls();status(`Loading ${profile.label}…`);
 try {
  const response=await fetch(`scenarios/${profile.path}/index.json`);if(!response.ok)throw Error(`HTTP ${response.status}`);
  const index=await response.json();
  const count=Number($('cities').value)||index.default_city_count;
  const choice=index.scenarios.find(c=>c.city_count===count)||index.scenarios.find(c=>c.city_count===index.default_city_count)||index.scenarios[0];
  await loadScenario(choice,{profile,profileChoices:index.scenarios});
 } finally {loading=false;controls();if(activeProfile)$('profile').value=activeProfile.id;}
}
$('cities').addEventListener('change',async()=>{
 const count=Number($('cities').value);const choice=choices.find(c=>c.city_count===count);
 // Read the existing settings against the current graph before changing scenarios.
 $('cities').value=scenario.config.prep.max_cities;
 try{await loadScenario(choice,{preserve:true});if($('auto').checked)schedule();}catch(e){$('cities').value=scenario.config.prep.max_cities;status(e.message);}
});
$('profile').addEventListener('change',async()=>{
 const profile=profiles.find(p=>p.id===$('profile').value);
 try{await loadProfile(profile);if($('auto').checked)schedule();}catch(e){status(`Cannot load profile: ${e.message}`);}
});
try {
 const response=await fetch('scenarios/index.json');if(!response.ok)throw Error(`HTTP ${response.status}`);
 const catalog=await response.json();profiles=catalog.profiles;
 for(const profile of profiles){const option=document.createElement('option');option.value=profile.id;option.textContent=profile.label;$('profile').append(option);}
 await loadProfile(profiles.find(p=>p.id===catalog.default_profile)||profiles[0]);
}catch(e){status(`Cannot load scenario: ${e.message}. Serve the repository over HTTP.`);}
