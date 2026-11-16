"""Flask app: document portal (live page + archive, at /) and the camera page (preview, Capture, gallery, at /camera)."""
from __future__ import annotations

from flask import Flask, Response, abort, jsonify, request, send_file

from .documents import DocumentStore

from .rectify import RectifyError
from .service import Busy, CaptureService
from .store import Store

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camera clipboard</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;padding:12px;max-width:760px;margin:auto;background:#fafafa;color:#222}
 #preview{width:100%;background:#000;aspect-ratio:16/9;object-fit:contain;display:block;border-radius:6px}
 button{font-size:1.2rem;padding:.7rem 1.2rem;border-radius:8px;border:0;background:#1a6fe0;color:#fff;margin:.6rem .4rem .6rem 0}
 button:disabled{background:#889}
 button.sec{background:#445;font-size:1rem;padding:.5rem .9rem}
 #status{min-height:1.4em}.warn{color:#a50}.err{color:#b00}.ok{color:#070}
 #gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(110px,1fr));gap:8px}
 .item{background:#fff;border:1px solid #ddd;border-radius:6px;padding:4px;font-size:.75rem}
 .item img{width:100%;display:block}.item label{display:block;margin:2px 0}
</style></head><body>
<p style="margin:0 0 4px"><a href="/">Documents</a> | <b>Camera</b></p>
<h2>Camera clipboard</h2>
<img id="preview" alt="live preview">
<button id="cap">Capture</button><span id="status"></span>
<h3>Recent captures</h3>
<div><button class="sec" id="pdf">PDF of ticked pages</button><span style="font-size:.8rem">(tick order = page order)</span></div>
<div id="gallery"></div>
<script>
const $=id=>document.getElementById(id);
let ticked=[];
function preview(){
  const im=$('preview');
  im.onload=im.onerror=()=>setTimeout(preview,400);
  im.src='/preview.jpg?'+Date.now();
}
function say(msg,cls){$('status').className=cls||'';$('status').textContent=msg;}
async function gallery(){
  const names=await (await fetch('/captures')).json();
  ticked=ticked.filter(n=>names.includes(n));
  const g=$('gallery');g.innerHTML='';
  for(const n of names){
    const d=document.createElement('div');d.className='item';
    d.innerHTML=`<a href="/captures/${n}.png" download><img src="/thumbs/${n}.jpg" loading="lazy"></a>
      <label><input type="checkbox" ${ticked.includes(n)?'checked':''}> ${n.slice(9,11)}:${n.slice(11,13)}:${n.slice(13,15)}</label>
      <a href="/captures/${n}.png" download="${n}.png">PNG</a>`;
    d.querySelector('input').onchange=e=>{ticked=e.target.checked?[...ticked,n]:ticked.filter(x=>x!==n);};
    g.appendChild(d);
  }
}
$('cap').onclick=async()=>{
  $('cap').disabled=true;say('Capturing... (about 10 s)');
  try{
    const r=await fetch('/capture',{method:'POST'});const j=await r.json();
    if(!r.ok){say(j.error||'failed','err');}
    else{ticked.push(j.name);
      say(`Saved ${j.name} - ${j.method}, ${j.seconds}s`+(j.warnings.length?' - '+j.warnings.join('; '):''),j.warnings.length?'warn':'ok');}
  }catch(e){say('Request failed: '+e,'err');}
  $('cap').disabled=false;gallery();
};
$('pdf').onclick=()=>{ if(!ticked.length){say('Tick at least one page','warn');return;}
  location.href='/pdf?names='+ticked.join(',');};
preview();gallery();
</script></body></html>"""

# Page text only reaches the browser as JSON and is inserted with textContent (never innerHTML).
_DOC_STYLE = """
 body{font-family:system-ui,sans-serif;margin:0 auto;padding:12px;max-width:900px;background:#fafafa;color:#222}
 nav{margin-bottom:10px} nav a{margin-right:10px}
 .live{border:2px solid #16a34a;border-radius:8px;padding:12px;background:#f0fdf4;margin-bottom:18px}
 .tag{display:inline-block;font-size:.7rem;font-weight:700;padding:1px 7px;border-radius:9px;color:#fff;background:#16a34a;margin-right:6px;vertical-align:middle}
 .tag.arch{background:#6b7280}.tag.warn{background:#d97706}
 .meta{color:#6b7280;font-size:.85rem}
 pre{white-space:pre-wrap;font-family:inherit;background:#fff;border:1px solid #e5e7eb;border-radius:6px;padding:10px;margin:8px 0;min-height:2.5em}
 .row{display:flex;gap:12px;align-items:flex-start}.row pre{flex:1}
 .thumb{width:170px;max-width:35%;border:1px solid #d1d5db;border-radius:4px}
 button,.btn{font-size:.9rem;padding:.35rem .8rem;border-radius:6px;border:1px solid #cbd5e1;background:#fff;color:#222;cursor:pointer;text-decoration:none;margin-right:6px}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));gap:10px}
 .card{display:block;background:#fff;border:1px solid #e5e7eb;border-radius:8px;padding:10px;color:inherit;text-decoration:none}
 .card:hover{border-color:#94a3b8}.card b{display:block;margin:4px 0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 .empty{color:#6b7280;font-style:italic}
 .revs li{margin:4px 0}
"""

_DOC_SCRIPT = """
const $=id=>document.getElementById(id);
function el(tag,cls,text){const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;}
function ago(iso){if(!iso)return '';const s=(Date.now()-new Date(iso).getTime())/1000;
  if(s<60)return Math.max(0,Math.round(s))+' s ago';if(s<3600)return Math.round(s/60)+' min ago';
  if(s<86400)return Math.round(s/3600)+' h ago';return iso.slice(0,10);}
function hhmm(iso){return iso?iso.slice(11,16):'';}
function copyText(t){
  if(window.isSecureContext&&navigator.clipboard){return navigator.clipboard.writeText(t);}
  const ta=document.createElement('textarea');ta.value=t;ta.style.position='fixed';ta.style.opacity='0';
  document.body.appendChild(ta);ta.select();try{document.execCommand('copy');}finally{ta.remove();}
  return Promise.resolve();
}
"""

DOCS_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clipboard documents</title><style>""" + _DOC_STYLE + """</style></head><body>
<nav><b>Documents</b> <a href="/camera">Camera</a></nav>
<section class="live" id="live"><span class="empty">Loading...</span></section>
<h3>Archive</h3>
<div class="grid" id="archive"></div>
<script>""" + _DOC_SCRIPT + """
let liveText='';
function renderLive(j){
  const box=$('live');box.replaceChildren();
  const st=j.status, p=j.page;
  if(!st){box.append(el('span','empty','Auto-capture is off.'));}
  if(!p){box.append(el('span','empty',st?'No live page yet: put a numbered sheet on the clipboard.':''));return;}
  const head=el('div');head.append(el('span','tag','LIVE'),el('b',null,'Page '+p.number+' '));
  head.append(el('span','meta','· updated '+ago(p.last_updated)+' · started '+hhmm(p.first_seen)));
  if(st&&st.qr_visible===false)head.append(' ',el('span','tag warn','QR not visible'));
  if(p.unsent||(st&&st.pending))head.append(' ',el('span','tag warn','unsent changes'));
  box.append(head);
  const row=el('div','row');const pre=el('pre',null,p.text||'(no text yet)');row.append(pre);
  if(p.image){const a=el('a');a.href='/pages/'+p.number+'/img/'+p.image;a.target='_blank';
    const im=el('img','thumb');im.src=a.href;im.alt='photo';a.append(im);row.append(a);}
  box.append(row);
  liveText=p.text||'';
  const cp=el('button',null,'Copy text');cp.onclick=()=>copyText(liveText).then(()=>{cp.textContent='Copied';setTimeout(()=>cp.textContent='Copy text',1200);});
  const dl=el('a','btn','Download .txt');dl.href='/pages/'+p.number+'.txt';
  const op=el('a','btn','Open page');op.href='/pages/'+p.number;
  box.append(cp,dl,op);
}
function renderArchive(list){
  const g=$('archive');g.replaceChildren();
  if(!list.length){g.append(el('span','empty','No archived pages yet.'));return;}
  for(const p of list){
    const a=el('a','card');a.href='/pages/'+p.number;
    const top=el('div');top.append(el('span','tag arch',String(p.number)));
    if(p.unsent)top.append(el('span','tag warn','unsent changes'));
    a.append(top,el('b',null,p.first_line||'(empty)'),el('span','meta',ago(p.last_updated)));
    g.append(a);
  }
}
async function pollLive(){try{renderLive(await (await fetch('/api/live')).json());}catch(e){}setTimeout(pollLive,2000);}
async function pollArchive(){try{renderArchive(await (await fetch('/api/pages')).json());}catch(e){}setTimeout(pollArchive,10000);}
pollLive();pollArchive();
</script></body></html>"""

NO_DOCS_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Clipboard</title></head>
<body style="font-family:system-ui,sans-serif;padding:12px">
<p>Documents are not available (no document store configured). <a href="/camera">Camera page</a></p></body></html>"""

PAGE_VIEW = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clipboard page</title><style>""" + _DOC_STYLE + """</style></head><body>
<nav><a href="/">Documents</a> <a href="/camera">Camera</a></nav>
<div id="page"><span class="empty">Loading...</span></div>
<script>""" + _DOC_SCRIPT + """
const N=__N__;
async function load(){
  const r=await fetch('/api/pages/__N__');const box=$('page');box.replaceChildren();
  if(!r.ok){box.append(el('span','empty','Page not found.'));return;}
  const p=await r.json();
  const h=el('h2',null,'Page '+p.number);box.append(h);
  box.append(el('div','meta','first seen '+(p.first_seen||'').replace('T',' ')+' · updated '+(p.last_updated||'').replace('T',' ')));
  if(p.unsent)box.append(el('span','tag warn','unsent changes'));
  const revs=p.revisions||[];const last=revs[revs.length-1];
  const row=el('div','row');row.append(el('pre',null,p.text||'(no text yet)'));
  if(last){const a=el('a');a.href='/pages/'+N+'/img/'+last.image;a.target='_blank';
    const im=el('img','thumb');im.src=a.href;im.style.width='320px';a.append(im);row.append(a);}
  box.append(row);
  const cp=el('button',null,'Copy text');cp.onclick=()=>copyText(p.text||'');
  const dl=el('a','btn','Download .txt');dl.href='/pages/'+N+'.txt';box.append(cp,dl);
  box.append(el('h3',null,'Revisions ('+revs.length+')'));
  const ul=el('ul','revs');
  for(const rv of revs.slice().reverse()){
    const li=el('li');const a=el('a',null,(rv.time||'').replace('T',' '));a.href='/pages/'+N+'/img/'+rv.image;a.target='_blank';
    li.append(a,el('span','meta',' '+((rv.text||'').split('\\n').find(l=>l.trim())||'(empty)').slice(0,80)));ul.append(li);
  }
  box.append(ul);
}
load();
</script></body></html>"""


def create_app(service: CaptureService, store: Store, gallery_size: int = 24,
               documents: DocumentStore | None = None, watcher=None) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return DOCS_PAGE if documents is not None else NO_DOCS_PAGE

    @app.get("/camera")
    def camera_page():
        return PAGE

    # --- document portal API ---
    def _live_summary():
        if documents is None:
            return None
        n = documents.live()
        return documents.summary(n) if n is not None else None

    @app.get("/api/live")
    def api_live():
        return jsonify(status=None if watcher is None else watcher.status(), page=_live_summary())

    @app.get("/api/pages")
    def api_pages():
        if documents is None:
            return jsonify([])
        live = documents.live()
        return jsonify([p for p in documents.summaries() if p["number"] != live])

    @app.get("/api/pages/<int:n>")
    def api_page(n):
        page = documents.get(n) if documents is not None else None
        return jsonify(page) if page is not None else abort(404)

    @app.get("/pages/<int:n>")
    def page_view(n):
        if documents is None or documents.get(n) is None:
            abort(404)
        return PAGE_VIEW.replace("__N__", str(n))

    @app.get("/pages/<int:n>.txt")
    def page_txt(n):
        page = documents.get(n) if documents is not None else None
        if page is None:
            abort(404)
        return Response(page.get("text", ""), mimetype="text/plain", content_type="text/plain; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="page-{n}.txt"'})

    @app.get("/pages/<int:n>/img/<name>")
    def page_img(n, name):
        p = documents.image_path(n, name) if documents is not None else None
        return send_file(p, mimetype="image/jpeg") if p else abort(404)

    @app.get("/preview.jpg")
    def preview():
        return Response(service.camera.preview_jpeg(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.post("/capture")
    def capture():
        try:
            return jsonify(service.capture())
        except Busy as e:
            return jsonify(error=str(e)), 409
        except RectifyError as e:
            return jsonify(error=str(e)), 422
        except Exception as e:  # camera failure etc.
            app.logger.exception("capture failed")
            return jsonify(error=f"capture failed: {e}"), 500

    @app.get("/captures")
    def captures():
        return jsonify(store.names()[:gallery_size])

    @app.get("/captures/<name>.png")
    def png(name):
        p = store.png_path(name)
        return send_file(p, mimetype="image/png", download_name=f"{name}.png") if p else abort(404)

    @app.get("/thumbs/<name>.jpg")
    def thumb(name):
        p = store.thumb_path(name)
        return send_file(p, mimetype="image/jpeg") if p else abort(404)

    @app.get("/pdf")
    def pdf():
        names = [n for n in request.args.get("names", "").split(",") if n]
        try:
            data = store.pdf(names)
        except KeyError:
            abort(404)
        return Response(data, mimetype="application/pdf",
                        headers={"Content-Disposition": 'attachment; filename="clipboard.pdf"'})

    return app
