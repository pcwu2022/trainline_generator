import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {fields,getValue,setValue} from '../config-fields.js';
const original=JSON.parse(readFileSync(new URL('../scenarios/france/cities-8/scenario.json',import.meta.url))).config;
test('common controls round-trip the native Python configuration',()=>{
 const config=structuredClone(original);
 for(const field of fields)setValue(config,field,String(getValue(original,field)));
 for(const field of fields)assert.ok(Math.abs(getValue(config,field)-getValue(original,field))<1e-10);
 assert.deepEqual(config.prep,original.prep);
});
test('finance and percentage controls convert display units',()=>{
 const config=structuredClone(original);
 setValue(config,fields.find(f=>f.id==='budget'),'250');
 setValue(config,fields.find(f=>f.id==='discount'),'4.5');
 setValue(config,fields.find(f=>f.id==='load'),'70');
 assert.equal(config.constraints.upfront_budget_eur,250e6);
 assert.equal(config.finance.discount_rate,.045);
 assert.ok(Math.abs(config.operations.target_load_factor-.7)<1e-12);
});
test('rejects blank, nonfinite, fractional integer and out-of-range values',()=>{
 for(const [id,value] of [['iterations',''],['population','3'],['seed','1.5'],['hours','25'],['load','101'],['discount','Infinity']]){
  assert.throws(()=>setValue(structuredClone(original),fields.find(f=>f.id===id),value),/Invalid/);
 }
});
