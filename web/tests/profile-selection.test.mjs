// Test the app's selector wiring with a small DOM fixture and local file responses.
import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';

test('profiles load regional defaults, cities preserve edits, and failed loads keep the current map',async()=>{
 const elements=new Map();
 class Element {
  constructor(tag='div'){this.tag=tag;this.children=[];this.attributes={};this.listeners={};this.value='';}
  set id(value){this._id=value;elements.set(value,this);} get id(){return this._id;}
  append(...children){this.children.push(...children);if(this.tag==='select'&&!this.value)this.value=String(children[0]?.value||'');}
  replaceChildren(...children){this.children=[];if(this.tag==='select')this.value='';this.append(...children);}
  setAttribute(key,value){this.attributes[key]=value;}
  addEventListener(event,listener){this.listeners[event]=listener;}
 }
 const html=readFileSync(new URL('../index.html',import.meta.url),'utf8');
 for(const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g)){const element=new Element(match[1]);element.id=match[2];}
 const doc={getElementById:id=>elements.get(id),createElement:tag=>new Element(tag),createElementNS:(_,tag)=>new Element(tag)};
 const originals={document:globalThis.document,fetch:globalThis.fetch,Worker:globalThis.Worker};
 const calls=[];let failProfile=null;
 Object.assign(globalThis,{document:doc,fetch:async path=>{
  if(failProfile&&path.includes(`/${failProfile}/`))return {ok:false,status:503};
  const body=readFileSync(new URL('../'+path,import.meta.url),'utf8');return {ok:true,json:async()=>JSON.parse(body)};
 },Worker:class {postMessage(data){calls.push(data);}terminate(){}}});
 const $=id=>elements.get(id);
 try {
  await import('../app.js?profile-selection-test');
  assert.equal($('profile').value,'france');assert.equal($('cities').value,8);
  assert.ok($('profile').children.length>=6);
  const descendants=element=>[element,...element.children.flatMap(descendants)];
  const helpLinks=descendants($('common-configs')).filter(el=>el.tag==='a');
  const guide=readFileSync(new URL('../guide.html',import.meta.url),'utf8');
  assert.equal(helpLinks.length,14);
  for(const help of helpLinks){
   assert.equal(help.textContent,'?');assert.equal(help.attributes.target,'_blank');
   assert.equal(help.attributes.rel,'noopener');
   assert.match(help.attributes['aria-label'],/opens in a new tab/);
   assert.ok(guide.includes(`id="${help.attributes.href.split('#')[1]}"`));
  }

  $('budget').value='250';$('budget').listeners.input();
  $('cities').value='12';await $('cities').listeners.change();
  assert.equal($('budget').value,250);
  assert.equal(JSON.parse($('config').value).prep.max_cities,12);
  for(const [profile,country] of [['taiwan','TW'],['usa','US'],['uk','GB'],['japan','JP'],['germany','DE']]){
   $('profile').value=profile;await $('profile').listeners.change();
   assert.equal($('profile').value,profile);assert.equal($('budget').value,0);
   assert.deepEqual(JSON.parse($('config').value).countries,[country]);
   assert.equal($('map').children[0].attributes.href,`scenarios/${profile}/cities-12/background.png`);
   $('run').onclick();
   assert.equal(calls.at(-1).scenarioPath,`scenarios/${profile}/cities-12/`);
   assert.deepEqual(calls.at(-1).config.countries,[country]);
  }
  const previous=$('map').children[0].attributes.href;
  failProfile='taiwan';$('profile').value='taiwan';await $('profile').listeners.change();
  assert.equal($('profile').value,'germany');assert.equal($('map').children[0].attributes.href,previous);
  assert.match($('status').textContent,/503/);assert.equal($('run').disabled,false);
 } finally {Object.assign(globalThis,originals);}
});
