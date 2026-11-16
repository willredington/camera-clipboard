"""Flask app: Capture button, live preview, gallery, PNG/PDF download."""
from __future__ import annotations

from flask import Flask, Response, abort, jsonify, request, send_file

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


def create_app(service: CaptureService, store: Store, gallery_size: int = 24) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index():
        return PAGE

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
