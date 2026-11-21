/** Navigate the SVG viewBox independently of live network redraws. */
export function mapNavigation(map) {
 let home, view;
 const pointers=new Map();
 const render=()=>map.setAttribute('viewBox',`${view.x} ${view.y} ${view.width} ${view.height}`);
 function point(x,y) {
  return new DOMPoint(x,y).matrixTransform(map.getScreenCTM().inverse());
 }
 function zoom(factor, anchor={x:view.x+view.width/2,y:view.y+view.height/2}) {
  if(!view)return;
  const width=Math.max(home.width/32,Math.min(home.width*2,view.width*factor));
  const ratio=width/view.width;
  view={x:anchor.x+(view.x-anchor.x)*ratio,y:anchor.y+(view.y-anchor.y)*ratio,width,height:view.height*ratio};render();
 }
 map.addEventListener('wheel',event=>{
  if(!view)return;
  event.preventDefault();
  zoom(Math.exp(Math.max(-100,Math.min(100,event.deltaY))*.002),point(event.clientX,event.clientY));
 },{passive:false});
 map.addEventListener('pointerdown',event=>{
  if(!view || (event.pointerType==='mouse' && event.button!==0))return;
  map.setPointerCapture(event.pointerId);
  pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
 });
 map.addEventListener('pointermove',event=>{
  if(!pointers.has(event.pointerId))return;
  const before=[...pointers.values()];
  pointers.set(event.pointerId,{x:event.clientX,y:event.clientY});
  const after=[...pointers.values()];
  const center=points=>({x:points.reduce((s,p)=>s+p.x,0)/points.length,y:points.reduce((s,p)=>s+p.y,0)/points.length});
  const a=center(before),b=center(after),from=point(a.x,a.y),to=point(b.x,b.y);
  view.x+=from.x-to.x;view.y+=from.y-to.y;render();
  if(before.length===2){
   const distance=points=>Math.hypot(points[0].x-points[1].x,points[0].y-points[1].y);
   if(distance(after)>0)zoom(distance(before)/distance(after),point(b.x,b.y));
  }
 });
 for(const name of ['pointerup','pointercancel','lostpointercapture'])map.addEventListener(name,event=>pointers.delete(event.pointerId));
 return {
  reset(width,height){home={x:0,y:0,width,height};view={...home};pointers.clear();render();},
  fit(){if(home){view={...home};render();}},
  zoomIn(){if(view)zoom(1/1.5);},
  zoomOut(){if(view)zoom(1.5);}
 };
}
