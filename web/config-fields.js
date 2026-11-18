// Display units are converted to the Python configuration's native units.
export const fields = [
  {id:'iterations', label:'Iterations', section:'Search', path:['search','generations'], min:1, step:1, integer:true},
  {id:'population', label:'Proposals per update', section:'Search', path:['search','population'], min:4, step:1, integer:true},
  {id:'seed', label:'Random seed', section:'Search', path:['search','seed'], min:0, step:1, integer:true},
  {id:'time-limit', label:'Time limit (seconds; 0 = unlimited)', section:'Search', path:['search','time_limit_seconds'], min:0, step:1},
  {id:'budget', label:'Upfront budget (€ million; 0 = unlimited)', section:'Finance', path:['constraints','upfront_budget_eur'], min:0, step:1, scale:1e6},
  {id:'subsidy', label:'Maximum annual subsidy (€ million; 0 = disabled)', section:'Finance', path:['constraints','maximum_subsidy_eur'], min:0, step:1, scale:1e6},
  {id:'discount', label:'Discount rate (%)', section:'Finance', path:['finance','discount_rate'], min:0, step:.1, scale:.01},
  {id:'demand-k', help:true, label:'Demand multiplier (k)', section:'Demand', path:['demand','k'], min:0, step:'any'},
  {id:'demand-beta', help:true, label:'Distance decay (beta)', section:'Demand', path:['demand','beta'], min:.01, step:.05},
  {id:'hours', label:'Operating hours per day', section:'Operations', path:['operations','hours_per_day'], min:.01, max:24, step:.5},
  {id:'load', label:'Target load factor (%)', section:'Operations', path:['operations','target_load_factor'], min:.01, max:100, step:1, scale:.01},
  {id:'max-load', label:'Maximum load factor (%)', section:'Operations', path:['operations','maximum_load_factor'], min:.01, max:100, step:1, scale:.01},
  {id:'fare-cr', label:'Conventional fare (€ / passenger-km)', section:'Operations', path:['rolling_stock','CR','yield_eur_pkm'], min:.001, step:.005},
  {id:'fare-hsr', label:'High-speed fare (€ / passenger-km)', section:'Operations', path:['rolling_stock','HSR','yield_eur_pkm'], min:.001, step:.005},
];
export function getValue(config, field) {
  return Number((field.path.reduce((value,key)=>value[key],config)/(field.scale ?? 1)).toPrecision(12));
}
export function setValue(config, field, raw) {
  const value=Number(raw);
  if(String(raw).trim()==='' || !Number.isFinite(value) || value<field.min || (field.max!==undefined && value>field.max) || (field.integer&&!Number.isInteger(value))) {
    throw Error(`Invalid ${field.label.toLowerCase()}: enter ${field.integer?'an integer':'a number'} from ${field.min}${field.max===undefined?'':` to ${field.max}`}.`);
  }
  const parent=field.path.slice(0,-1).reduce((value,key)=>value[key],config);
  parent[field.path.at(-1)]=value*(field.scale ?? 1);
}
