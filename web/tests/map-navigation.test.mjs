import test from 'node:test';
import assert from 'node:assert/strict';
import {mapNavigation} from '../map-navigation.js';

test('map navigation zooms, pans, clamps scale, and resets for a new scenario',()=>{
 const handlers={};let box;
 const original=globalThis.DOMPoint;
 globalThis.DOMPoint=class {
  constructor(x,y){this.x=x;this.y=y;}
  matrixTransform(matrix){return matrix(this);}
 };
 const map={
  setAttribute(_,value){box=value.split(' ').map(Number);},
  addEventListener(name,handler){handlers[name]=handler;},
  setPointerCapture(){},
  getScreenCTM(){return {inverse:()=>p=>({x:box[0]+p.x*box[2]/100,y:box[1]+p.y*box[3]/100})};}
 };
 try {
  const nav=mapNavigation(map);nav.reset(100,100);
  let prevented=false;
  handlers.wheel({deltaY:-100,clientX:25,clientY:25,preventDefault(){prevented=true;}});
  assert.ok(prevented);assert.ok(box[2]<100);
  assert.ok(Math.abs(box[0]+.25*box[2]-25)<1e-10);
  const previous=box.slice();
  handlers.pointerdown({pointerId:1,clientX:50,clientY:50,pointerType:'mouse',button:0});
  handlers.pointermove({pointerId:1,clientX:60,clientY:50});
  assert.ok(box[0]<previous[0]);assert.equal(box[2],previous[2]);
  handlers.pointerup({pointerId:1});
  const released=box.slice();handlers.pointermove({pointerId:1,clientX:80,clientY:50});assert.deepEqual(box,released);
  for(let i=0;i<100;i++)nav.zoomIn();assert.equal(box[2],100/32);
  for(let i=0;i<100;i++)nav.zoomOut();assert.equal(box[2],200);
  nav.fit();assert.deepEqual(box,[0,0,100,100]);
  handlers.pointerdown({pointerId:1,clientX:25,clientY:50});
  handlers.pointerdown({pointerId:2,clientX:75,clientY:50});
  handlers.pointermove({pointerId:2,clientX:100,clientY:50});assert.ok(box[2]<100);
  handlers.pointercancel({pointerId:1});handlers.pointercancel({pointerId:2});
  nav.reset(200,150);assert.deepEqual(box,[0,0,200,150]);
 }finally{globalThis.DOMPoint=original;}
});
