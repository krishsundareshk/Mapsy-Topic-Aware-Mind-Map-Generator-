"""
pipeline/phase5_visualisation.py — v4
========================================
Changes from v3
---------------
1. "related to" predicate pills suppressed entirely — only real SPO
   verbs get a pill rendered on the edge.

2. Per-edge arrowhead markers — ensureMarker() registers one <marker>
   per colour in <defs> so every arrowhead matches its branch exactly.

3. Sticky tab strip — header has z-index:10 and flex-shrink:0.

4. Fit button — resets pan/zoom back to default view.
"""
from __future__ import annotations
import json, math, textwrap
from typing import List, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

_PALETTE = [
    "#2563eb","#16a34a","#dc2626","#d97706","#7c3aed",
    "#db2777","#0891b2","#65a30d",
]

def _colour(topic_index: int) -> str:
    return _PALETTE[topic_index % len(_PALETTE)]

# ─────────────────────────────────────────────────────────────────────────────
# Hierarchy → JSON-serialisable dict
# ─────────────────────────────────────────────────────────────────────────────

def _hierarchy_to_js(h: Dict[str, Any]) -> Dict[str, Any]:
    root = {"label": h.get("label","Topic"), "wef_score": h.get("wef_score",0), "children": []}
    for l2 in h.get("children", []):
        l2_node = {"label": l2.get("label",""), "wef_score": l2.get("wef_score",0), "children": []}
        for l3 in l2.get("children", []):
            l2_node["children"].append({
                "label":     l3.get("label",""),
                "wef_score": l3.get("wef_score",0),
                "predicate": l3.get("predicate","related to"),
                "direction": l3.get("direction","out"),
            })
        root["children"].append(l2_node)
    return root

# ─────────────────────────────────────────────────────────────────────────────
# SVG / layout config
# ─────────────────────────────────────────────────────────────────────────────

_SVG_CFG = {
    "CW": 1400, "CH": 900,
    "rRoot": 52, "rL2": 320, "rL3Base": 530, "rL3Extra": 60,
    "rL2Node": 38, "rL3Node": 28,
    "fontSize": 12, "lineH": 15,
}

# ─────────────────────────────────────────────────────────────────────────────
# HTML builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_html(hierarchies: List[Dict[str,Any]], title: str) -> str:
    topics_json  = json.dumps([_hierarchy_to_js(h) for h in hierarchies],
                               ensure_ascii=False)
    cfg_json     = json.dumps(_SVG_CFG)
    colours_json = json.dumps([_colour(i) for i in range(len(hierarchies))])

    js = r"""
const TOPICS  = """ + topics_json  + r""";
const CFG     = """ + cfg_json     + r""";
const COLOURS = """ + colours_json + r""";

let activeTopic=0, activeDepth=3;
let panX=0, panY=0, scale=1;
let dragging=false, lastX=0, lastY=0;

const SVG_NS="http://www.w3.org/2000/svg";
function el(tag,attrs={},children=[]){
  const e=document.createElementNS(SVG_NS,tag);
  for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);
  for(const c of children)e.appendChild(c);
  return e;
}

function wrapText(parent,text,cx,cy,maxW,lineH,fontSize){
  const words=text.split(" "),approxCW=fontSize*0.55;
  const maxCh=Math.floor(maxW/approxCW);
  const lines=[];let cur="";
  for(const w of words){
    if((cur+" "+w).trim().length>maxCh&&cur){lines.push(cur.trim());cur=w;}
    else{cur=(cur+" "+w).trim();}
  }
  if(cur)lines.push(cur);
  const totalH=lines.length*lineH;
  lines.forEach((line,i)=>{
    const t=el("text",{x:cx,y:cy-totalH/2+i*lineH+lineH*0.75,
      "text-anchor":"middle","font-size":fontSize,
      "font-family":"Inter,Segoe UI,sans-serif",fill:"#fff",
      "pointer-events":"none"});
    t.textContent=line;parent.appendChild(t);
  });
}

function pillLabel(parent,mx,my,text,colour){
  const pw=text.length*6.5+14,ph=16;
  const g=el("g",{"pointer-events":"none"});
  g.appendChild(el("rect",{x:mx-pw/2,y:my-ph/2,width:pw,height:ph,
    rx:7,fill:colour,opacity:0.9}));
  const t=el("text",{x:mx,y:my+4.5,"text-anchor":"middle",
    "font-size":9.5,fill:"#fff","font-family":"Inter,Segoe UI,sans-serif"});
  t.textContent=text;g.appendChild(t);parent.appendChild(g);
}

function ensureMarker(defs,colour){
  const id="arr-"+colour.replace("#","");
  if(document.getElementById(id))return id;
  const m=el("marker",{id,markerWidth:8,markerHeight:6,refX:6,refY:3,orient:"auto"});
  m.appendChild(el("path",{d:"M0,0 L0,6 L8,3 z",fill:colour}));
  defs.appendChild(m);return id;
}

function render(){
  const svg=document.getElementById("mindmap");
  while(svg.firstChild)svg.removeChild(svg.firstChild);
  const defs=el("defs");svg.appendChild(defs);
  const topic=TOPICS[activeTopic],colour=COLOURS[activeTopic];
  const CW=CFG.CW,CH=CFG.CH,cx=CW/2,cy=CH/2;
  const g=el("g",{transform:`translate(${panX},${panY}) scale(${scale})`});
  svg.appendChild(g);
  const rL2=CFG.rL2;
  const l2list=activeDepth===1?[]:topic.children;
  const nL2=l2list.length;

  l2list.forEach((l2,i)=>{
    const angL2=(2*Math.PI*i/nL2)-Math.PI/2;
    const l2x=cx+rL2*Math.cos(angL2),l2y=cy+rL2*Math.sin(angL2);

    g.appendChild(el("circle",{cx,cy,r:rL2,fill:"none",
      stroke:colour,"stroke-opacity":0.08,"stroke-width":1}));

    const mrkR=ensureMarker(defs,colour);
    g.appendChild(el("line",{x1:cx,y1:cy,x2:l2x,y2:l2y,
      stroke:colour,"stroke-width":2.5,"stroke-opacity":0.7,
      "marker-end":`url(#${mrkR})`}));

    const l3list=activeDepth===3?l2.children:[];
    const nL3=l3list.length;
    const rL3=CFG.rL3Base+(nL3>4?CFG.rL3Extra*Math.ceil((nL3-4)/2):0);
    const spanAng=nL3>1?Math.min(Math.PI*0.72,nL3*0.22):0;

    l3list.forEach((l3,j)=>{
      const angL3=angL2+(nL3===1?0:-spanAng/2+spanAng*j/(nL3-1));
      const l3x=cx+rL3*Math.cos(angL3),l3y=cy+rL3*Math.sin(angL3);
      const dx=l3x-l2x,dy=l3y-l2y;
      const cpx=l2x+dx*0.45+dy*0.15,cpy=l2y+dy*0.45-dx*0.15;
      const mrkL=ensureMarker(defs,colour);

      g.appendChild(el("path",{
        d:`M${l2x},${l2y} Q${cpx},${cpy} ${l3x},${l3y}`,
        fill:"none",stroke:colour,"stroke-width":1.6,
        "stroke-opacity":0.55,"marker-end":`url(#${mrkL})`}));

      // Only show pill for real SPO predicates — suppress "related to"
      const pred=l3.predicate||"related to";
      if(pred&&pred!=="related to"){
        pillLabel(g,(l2x+cpx+l3x)/3,(l2y+cpy+l3y)/3,pred,colour);
      }

      const l3c=el("circle",{cx:l3x,cy:l3y,r:CFG.rL3Node,
        fill:colour,"fill-opacity":0.72,stroke:"#fff","stroke-width":1.4,
        style:"cursor:pointer"});
      l3c.appendChild(el("title",{}));
      l3c.querySelector("title").textContent=
        `${l3.label}\nWEF: ${(l3.wef_score||0).toFixed(3)}\nPredicate: ${pred}`;
      g.appendChild(l3c);
      wrapText(g,l3.label,l3x,l3y,CFG.rL3Node*2-4,CFG.lineH,CFG.fontSize-1);
    });

    const l2c=el("circle",{cx:l2x,cy:l2y,r:CFG.rL2Node,
      fill:colour,stroke:"#fff","stroke-width":2,style:"cursor:pointer"});
    l2c.appendChild(el("title",{}));
    l2c.querySelector("title").textContent=
      `${l2.label}\nWEF: ${(l2.wef_score||0).toFixed(3)}`;
    g.appendChild(l2c);
    wrapText(g,l2.label,l2x,l2y,CFG.rL2Node*2-6,CFG.lineH,CFG.fontSize);
  });

  g.appendChild(el("circle",{cx,cy,r:CFG.rRoot,fill:colour,stroke:"#fff","stroke-width":3}));
  wrapText(g,topic.label,cx,cy,CFG.rRoot*2-10,CFG.lineH+1,CFG.fontSize+1);
}

function fitView(){panX=0;panY=0;scale=1;render();}

function initInteraction(){
  const svg=document.getElementById("mindmap");
  svg.addEventListener("wheel",e=>{
    e.preventDefault();
    scale=Math.min(4,Math.max(0.2,scale*(e.deltaY>0?0.9:1.1)));render();
  },{passive:false});
  svg.addEventListener("mousedown",e=>{dragging=true;lastX=e.clientX;lastY=e.clientY;});
  svg.addEventListener("mousemove",e=>{
    if(!dragging)return;
    panX+=e.clientX-lastX;panY+=e.clientY-lastY;
    lastX=e.clientX;lastY=e.clientY;render();
  });
  svg.addEventListener("mouseup",()=>{dragging=false;});
  svg.addEventListener("mouseleave",()=>{dragging=false;});
  let ltx=0,lty=0;
  svg.addEventListener("touchstart",e=>{ltx=e.touches[0].clientX;lty=e.touches[0].clientY;});
  svg.addEventListener("touchmove",e=>{
    e.preventDefault();
    panX+=e.touches[0].clientX-ltx;panY+=e.touches[0].clientY-lty;
    ltx=e.touches[0].clientX;lty=e.touches[0].clientY;render();
  },{passive:false});
}

function buildTabs(){
  const strip=document.getElementById("tabs");
  TOPICS.forEach((t,i)=>{
    const btn=document.createElement("button");
    btn.className="tab"+(i===0?" active":"");
    btn.style.setProperty("--tc",COLOURS[i]);
    btn.textContent=t.label.length>18?t.label.slice(0,16)+"…":t.label;
    btn.onclick=()=>{
      activeTopic=i;
      document.querySelectorAll(".tab").forEach((b,j)=>b.classList.toggle("active",j===i));
      panX=0;panY=0;scale=1;render();
    };
    strip.appendChild(btn);
  });
}

function buildDepthBtns(){
  document.querySelectorAll(".depth-btn").forEach(btn=>{
    btn.addEventListener("click",()=>{
      activeDepth=parseInt(btn.dataset.d);
      document.querySelectorAll(".depth-btn").forEach(b=>b.classList.toggle("active",b===btn));
      panX=0;panY=0;scale=1;render();
    });
  });
  document.getElementById("fit-btn").addEventListener("click",fitView);
}

window.addEventListener("DOMContentLoaded",()=>{
  buildTabs();buildDepthBtns();initInteraction();render();
});
"""

    css = textwrap.dedent("""
        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: Inter, Segoe UI, sans-serif;
            background: #0f1117; color: #e2e8f0;
            display: flex; flex-direction: column; height: 100vh; overflow: hidden;
        }
        header {
            display: flex; align-items: center; gap: 12px;
            padding: 10px 20px; background: #161b27;
            border-bottom: 1px solid #2d3748; flex-shrink: 0; z-index: 10;
        }
        header h1 { font-size: 15px; font-weight: 600; color: #e2e8f0; white-space: nowrap; }
        #tabs {
            display: flex; gap: 6px; overflow-x: auto; flex: 1;
            scrollbar-width: thin;
        }
        .tab {
            padding: 5px 13px; border-radius: 20px; border: 1.5px solid var(--tc);
            background: transparent; color: var(--tc);
            font-size: 12px; font-weight: 500; cursor: pointer; white-space: nowrap;
            transition: background .15s, color .15s;
        }
        .tab.active { background: var(--tc); color: #fff; }
        .controls { display: flex; gap: 6px; align-items: center; flex-shrink: 0; }
        .depth-btn, #fit-btn {
            padding: 5px 11px; border-radius: 6px; border: 1px solid #4a5568;
            background: #1a2236; color: #a0aec0;
            font-size: 12px; cursor: pointer; transition: background .15s, color .15s;
        }
        .depth-btn.active { background: #2d3748; color: #e2e8f0; border-color: #718096; }
        #fit-btn:hover { background: #2d3748; color: #e2e8f0; }
        #canvas { flex: 1; overflow: hidden; }
        #mindmap { width: 100%; height: 100%; display: block; cursor: grab; }
        #mindmap:active { cursor: grabbing; }
    """)

    html = textwrap.dedent(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} — Mind Map</title>
<style>{css}</style>
</head>
<body>
<header>
  <h1>🗺 {title}</h1>
  <div id="tabs"></div>
  <div class="controls">
    <button class="depth-btn active" data-d="1">L1</button>
    <button class="depth-btn" data-d="2">L2</button>
    <button class="depth-btn" data-d="3">L3</button>
    <button id="fit-btn">⊡ Fit</button>
  </div>
</header>
<div id="canvas">
  <svg id="mindmap"
       viewBox="0 0 {_SVG_CFG['CW']} {_SVG_CFG['CH']}"
       xmlns="http://www.w3.org/2000/svg">
  </svg>
</div>
<script>{js}</script>
</body>
</html>""")
    return html

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run(
    hierarchies: List[Dict[str, Any]],
    output_path: str = "mindmap.html",
    title:       str = "Mind Map",
    verbose:     bool = False,
) -> str:
    html = _build_html(hierarchies, title)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    if verbose:
        total_l2 = sum(len(h.get("children",[])) for h in hierarchies)
        total_l3 = sum(len(c.get("children",[]))
                       for h in hierarchies for c in h.get("children",[]))
        print(f"  [Phase 5] Wrote {output_path}  "
              f"({len(hierarchies)} topics, {total_l2} L2, {total_l3} L3 nodes, "
              f"{len(html):,} bytes)")
    return output_path