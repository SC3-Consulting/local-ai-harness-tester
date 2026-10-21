"""Loopback-only, trusted-user interface to the local benchmark runner."""

import argparse
import json
import secrets
import sqlite3
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import harness_tester as ht
from evaluations import basic_suite

HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Local AI Harness Tester</title>
<style>
body{font:16px system-ui;margin:0;background:#101827;color:#e5edf8}
main{max-width:1100px;margin:auto;padding:24px}
h1,h2{color:#93c5fd} section{background:#1b273b;padding:20px;border-radius:12px;margin:18px 0}
label{display:block;margin:12px 0}input,select,textarea,button{font:inherit;padding:8px;border-radius:5px}
input,select,textarea{background:#0d1523;color:#fff;border:1px solid #64748b}
option:disabled{color:#94a3b8}
textarea{width:calc(100% - 20px);font:14px monospace;min-height:90px}
button{background:#93c5fd;color:#101827;border:0;margin:5px;cursor:pointer}
button:disabled{opacity:.5;cursor:wait}.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:13px}table{width:100%;border-collapse:collapse}
td,th{text-align:left;border-bottom:1px solid #526078;padding:8px}#error{color:#fca5a5;white-space:pre-wrap}
small{color:#b3c3da}@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style></head><body><main>
<h1>Local AI Harness Tester</h1>
<p>Run local model benchmarks, compare stored results, and find the best measured settings.</p>
<p id="error" role="alert"></p>
<section><details><summary><h2 style="display:inline">Machine</h2></summary>
<button id="hardware">Refresh hardware</button>
<pre id="machine">Machine details load when expanded.</pre></details></section>
<section><h2>Configure a benchmark</h2>
<p><small>Servers must already be running. AirLLM requires its optional dependencies.
No automatic model downloads or server installation. Custom commands run with your local user's permissions.</small></p>
<div class="grid"><div>
<label>Runtime <select id="runtime"><option>ollama</option><option>llama.cpp</option>
<option>airllm</option><option>colibri</option><option>custom</option></select></label>
<button id="runtime-refresh" type="button">Refresh runtime detection</button>
<p id="runtime-status" role="status" style="white-space:pre-line;overflow-wrap:anywhere"></p>
<details><summary>Runtime discovery endpoints</summary>
<label>Ollama endpoint <input id="detect-ollama" value="http://127.0.0.1:11434"></label>
<label>llama.cpp endpoint <input id="detect-llama" value="http://127.0.0.1:8080"></label>
<label>Colibri endpoint <input id="detect-colibri" value="http://127.0.0.1:8000/v1"></label>
<small>Edit an endpoint here and refresh detection to enable a server on a non-default address.</small>
</details>
<label>Model identifier / path
<input id="model" placeholder="e.g. llama3.2:3b">
<select id="model-select" hidden aria-label="Available models"></select></label>
<button id="discover-models" type="button">Refresh model list</button><small id="model-status"></small>
<button id="model-manual" type="button" hidden>Enter model manually</button>
<label>Harness version <input id="harness" value="ollama-v1"></label>
<label>Server endpoint <input id="endpoint" value="http://127.0.0.1:11434"></label>
</div><div>
<label>Repetitions <input id="repetitions" type="number" min="1" value="3"></label>
<label>Timeout per trial (seconds) <input id="timeout" type="number" min="1" value="300"></label>
<label>Seed <input id="seed" type="number" value="42"></label>
<label>Working directory / media base <input id="cwd" value="."></label>
</div></div>
<label>Fixed parameters (JSON)<textarea id="parameters">{"temperature":0,"top_p":0.9}</textarea></label>
<label>Parameter sweep (JSON arrays)<textarea id="grid">{"context_size":[2048,4096]}</textarea></label>
<fieldset id="context-controls"><legend>Context sweep range</legend>
<label>Minimum context <input id="context-min" type="number" min="1" step="1" value="2048"></label>
<label>Maximum context <input id="context-max" type="number" min="1" step="1" value="8192"></label>
<label>Context step <input id="context-step" type="number" min="1" step="1" value="2048"></label>
<button id="context-defaults" type="button">Use model/runtime defaults</button>
<button id="context-apply" type="button">Apply context range to sweep</button>
</fieldset><p id="context-status" role="status"></p>
<p><small>Range controls generate the context_size array above, preserving other sweep parameters.
The maximum is included even if the step does not land exactly on it.
Trials remain shuffled to reduce ordering bias; the step controls context values, not execution order.</small></p>
<label>Custom command (JSON argument array; custom runtime only)
<textarea id="command">["python3","my_harness.py"]</textarea></label>
<label>Test cases (JSON; supports media parts and embeddings)
<textarea id="cases" style="min-height:220px"></textarea></label>
<button id="basic">Load basic / JSON tests</button><button id="embedding">Load embedding retrieval test</button>
<button id="start">Run benchmark</button>
<div id="benchmark-progress" hidden>
<label for="trial-progress">Benchmark progress</label>
<progress id="trial-progress" max="100" value="0" style="width:100%"></progress>
<p id="progress-label" role="status">0%</p>
<small>Progress counts finished trials, including failures; it is not an estimate of remaining time.</small>
</div><pre id="job"></pre>
</section>
<section><h2>Stored runs</h2><button id="refresh">Refresh runs</button>
<label>Recommendation objective <select id="objective">
<option value="quality">Highest quality, then lowest latency</option>
<option value="latency">Lowest latency meeting minimum quality</option></select></label>
<label>Minimum quality (0 to 1)
<input id="min-quality" type="number" min="0" max="1" step="0.01" value="0"></label>
<button id="compare">Compare selected runs</button>
<div style="overflow:auto"><table><thead><tr><th>Select</th><th>Run</th><th>Status</th><th>Actions</th></tr></thead>
<tbody id="runs"></tbody></table></div></section>
<section><h2>Results / comparison / recommendation</h2>
<p><small>Recommendations rank complete successful settings by your chosen objective within the same
model, harness, suite, and machine scope. They do not establish a global optimum or minimum hardware fit.</small></p>
<div id="charts"></div>
<pre id="result">Select a stored run to inspect its results.</pre></section>
</main><script>
const token="__TOKEN__";
const el=id=>document.getElementById(id);
async function api(path,body){
  const response=await fetch(path,{method:body===undefined?"GET":"POST",
    headers:{"X-Harness-Token":token,"Content-Type":"application/json"},
    body:body===undefined?undefined:JSON.stringify(body)});
  const data=await response.json();
  if(!response.ok)throw Error(data.error||response.statusText);
  return data;
}
async function action(fn){el("error").textContent="";try{await fn()}catch(e){el("error").textContent=e.message}}
function show(data){
  el("result").textContent=JSON.stringify(data,null,2);
  renderCharts(data);
}
function svgElement(tag,attrs,text){
  const node=document.createElementNS("http://www.w3.org/2000/svg",tag);
  for(const [key,value] of Object.entries(attrs))node.setAttribute(key,value);
  if(text!==undefined)node.textContent=text;
  return node;
}
function renderCharts(data){
  const container=el("charts");container.replaceChildren();
  const runs=data.runs||(data.configurations?[data]:[]);
  if(!runs.length)return;
  const keys=["context_size","context","num_ctx"];
  const points=[];
  for(const run of runs){
    for(const config of run.configurations){
      const contexts=keys.filter(key=>key in config.parameters);
      if(contexts.length!==1)continue;
      const x=config.parameters[contexts[0]];
      if(typeof x!=="number"||!Number.isFinite(x)||x<=0)continue;
      if(!config.eligible)continue;
      points.push({x,config,run});
    }
  }
  if(!points.length){
    const note=document.createElement("p");
    note.textContent="No complete successful results with a numeric context window to graph.";
    container.append(note);return;
  }
  const note=document.createElement("p");
  note.textContent="Each point is a configuration mean across cases/repetitions. "+
    "Hover or focus a point for run ID and parameters. Points are not joined: other parameters may differ."+
    (data.comparable===false?" Warning: selected runs have different comparison scopes.":"");
  container.append(note);
  for(const [metric,title] of [["mean_latency_seconds","Context window x latency (seconds)"],
                               ["mean_score","Context window x quality (0–1)"]]){
    const heading=document.createElement("h3");heading.textContent=title;container.append(heading);
    const svg=svgElement("svg",{viewBox:"0 0 700 300",role:"img","aria-label":title,
      style:"width:100%;max-width:900px;background:#0d1523"});
    const maxX=Math.max(...points.map(p=>p.x));
    const maxY=metric==="mean_score"?1:Math.max(...points.map(p=>p.config[metric]),0.001);
    svg.append(svgElement("path",{d:"M70 20 V250 H675",fill:"none",stroke:"#b3c3da"}));
    for(let i=0;i<=4;i++){
      const x=70+i*150,y=250-i*55;
      svg.append(svgElement("text",{x,y:270,fill:"#e5edf8","text-anchor":"middle"},
        String(Math.round(maxX*i/4))));
      svg.append(svgElement("text",{x:60,y:y+4,fill:"#e5edf8","text-anchor":"end"},
        (maxY*i/4).toFixed(3)));
    }
    svg.append(svgElement("text",{x:370,y:295,fill:"#e5edf8","text-anchor":"middle"},"Context window"));
    for(const point of points){
      const index=runs.indexOf(point.run);
      const colors=["#93c5fd","#fbbf24","#a7f3d0","#f9a8d4"];
      const description=point.run.run_id+" "+JSON.stringify(point.config.parameters)+
        " | context="+point.x+" | "+metric+"="+point.config[metric];
      const circle=svgElement("circle",{cx:70+point.x/maxX*600,
        cy:250-point.config[metric]/maxY*220,r:5,fill:colors[index%colors.length],
        tabindex:0,"aria-label":description});
      circle.append(svgElement("title",{},description));svg.append(circle);
    }
    container.append(svg);
  }
}
async function hardware(){el("machine").textContent=JSON.stringify(await api("/api/machine"),null,2)}
async function runs(){
  const saved=await api("/api/runs");el("runs").replaceChildren();
  for(const run of saved){
    const tr=document.createElement("tr");
    const td=document.createElement("td");const check=document.createElement("input");
    check.type="checkbox";check.value=run.id;td.append(check);tr.append(td);
    for(const text of [run.id+"\n"+run.created_at,run.status]){
      const cell=document.createElement("td");cell.textContent=text;tr.append(cell);
    }
    const cell=document.createElement("td");
    for(const [label,path] of [["Show","/api/run?id="],["Recommend","/api/recommend?id="]]){
      const button=document.createElement("button");button.textContent=label;
      button.onclick=()=>action(async()=>{
        const query=label==="Recommend"?"&objective="+encodeURIComponent(el("objective").value)+
          "&min_quality="+encodeURIComponent(el("min-quality").value):"";
        show(await api(path+encodeURIComponent(run.id)+query));
      });cell.append(button);
    }
    tr.append(cell);el("runs").append(tr);
  }
}
async function basic(){el("cases").value=JSON.stringify(await api("/api/suite"),null,2)}
let modelDiscoveryVersion=0;
let contextDefaultsVersion=0;
let runtimeDetectionVersion=0;
const discoveryInputs={"ollama":"detect-ollama","llama.cpp":"detect-llama","colibri":"detect-colibri"};
async function detectRuntimes(){
  const version=++runtimeDetectionVersion;
  const endpoints=Object.fromEntries(Object.entries(discoveryInputs).map(([runtime,id])=>
    [runtime,el(id).value]));
  el("runtime-status").textContent="Checking runtime availability...";
  try{
    const result=await api("/api/runtimes?"+Object.entries(endpoints).map(([runtime,endpoint])=>
      encodeURIComponent(runtime)+"="+encodeURIComponent(endpoint)).join("&"));
    if(version!==runtimeDetectionVersion)return;
    for(const option of el("runtime").options){
      const runtime=option.value;
      option.value=runtime;
      const state=result[runtime];
      option.disabled=!state.available;
      option.textContent=runtime+(state.available?"":" (not detected)");
      option.title=state.reason;
    }
    const selected=el("runtime").selectedOptions[0];
    if(selected.disabled){
      el("runtime").value=[...el("runtime").options].find(option=>!option.disabled).value;
      el("runtime").onchange();
    }
    el("runtime-status").textContent=Object.entries(result).map(([runtime,state])=>
      runtime+": "+state.reason).join("\n");
  }catch(error){
    el("runtime-status").textContent="Runtime detection failed: "+error.message;
  }
}
el("runtime-refresh").onclick=()=>action(detectRuntimes);
for(const id of Object.values(discoveryInputs)){
  el(id).addEventListener("change",()=>action(async()=>{
    const runtime=el("runtime").value;
    if(discoveryInputs[runtime]===id)el("endpoint").value=el(id).value;
    await detectRuntimes();await discoverModels();
  }));
}
function currentModel(){return el("model-select").hidden?el("model").value:el("model-select").value}
function applyContextRange(){
  const minimum=Number(el("context-min").value);
  const maximum=Number(el("context-max").value);
  const step=Number(el("context-step").value);
  if(![minimum,maximum,step].every(value=>Number.isSafeInteger(value)&&value>0)||minimum>maximum)
    throw Error("Context minimum, maximum and step must be positive integers, with minimum <= maximum.");
  const count=Math.floor((maximum-minimum)/step)+1;
  if(count+(minimum+(count-1)*step===maximum?0:1)>1000)
    throw Error("Context range exceeds 1,000 values. Increase the step or reduce the maximum.");
  const grid=JSON.parse(el("grid").value);
  if(!grid||Array.isArray(grid)||typeof grid!=="object")
    throw Error("Parameter sweep must be a JSON object.");
  const values=Array.from({length:count},(_,i)=>minimum+i*step);
  if(values[values.length-1]!==maximum)values.push(maximum);
  delete grid.context;delete grid.num_ctx;
  grid.context_size=values;
  el("grid").value=JSON.stringify(grid,null,2);
}
async function contextDefaults(){
  const version=++contextDefaultsVersion;
  const runtime=el("runtime").value;
  // Re-enable the defaults button even for runtimes with server-fixed context.
  el("context-controls").disabled=false;
  for(const id of ["context-min","context-max","context-step","context-apply"])
    el(id).disabled=true;
  el("context-status").textContent="Loading model/runtime context defaults...";
  try{
    const cases=el("cases").value.trim()?JSON.parse(el("cases").value):[];
    if(Array.isArray(cases)&&cases.some(test=>test.task==="embedding")){
      el("context-status").textContent=
        "Context sweeps are generation-only; embedding adapters do not accept context_size.";
      return;
    }
    let model=currentModel();
    if(runtime==="airllm"){
      const parameters=JSON.parse(el("parameters").value);
      model=parameters.model_path||model;
    }
    const result=await api("/api/context-defaults?runtime="+encodeURIComponent(runtime)+
      "&model="+encodeURIComponent(model)+"&endpoint="+encodeURIComponent(el("endpoint").value)+
      "&cwd="+encodeURIComponent(el("cwd").value));
    if(version!==contextDefaultsVersion)return;
    for(const id of ["context-min","context-max","context-step","context-apply"])
      el(id).disabled=!result.supported;
    el("context-status").textContent=result.source+": "+result.note;
    if(result.supported){
      el("context-min").value=result.minimum;
      el("context-max").value=result.maximum;
      el("context-step").value=result.step;
      applyContextRange();
    }
  }catch(error){
    if(version!==contextDefaultsVersion)return;
    el("context-status").textContent="Context defaults unavailable: "+error.message+
      ". Existing parameter sweep has not been changed.";
  }
}
el("context-apply").onclick=()=>action(applyContextRange);
el("context-defaults").onclick=()=>action(contextDefaults);
for(const id of ["context-min","context-max","context-step"])
  el(id).addEventListener("change",()=>action(applyContextRange));
function useManualModel(message){
  el("model").hidden=false;
  el("model-select").hidden=true;
  el("model-manual").hidden=true;
  el("model-status").textContent=message;
}
async function discoverModels(){
  const version=++modelDiscoveryVersion;
  ++contextDefaultsVersion;
  const runtime=el("runtime").value;
  const status=el("model-status");
  const options=el("model-select");
  const existing=options.hidden?el("model").value:options.value;
  options.replaceChildren();
  useManualModel(" Connecting to server...");
  if(!["ollama","llama.cpp","colibri"].includes(runtime)){
    useManualModel(" This runtime does not provide model discovery; enter the model or local path.");
    await contextDefaults();
    return;
  }
  status.textContent=" Connecting to server...";
  try{
    const result=await api("/api/models?runtime="+encodeURIComponent(runtime)+
      "&endpoint="+encodeURIComponent(el("endpoint").value));
    if(version!==modelDiscoveryVersion)return;
    for(const model of result.models){
      const option=document.createElement("option");
      option.value=model;option.textContent=model;options.append(option);
    }
    if(result.models.length){
      options.value=result.models.includes(existing)?existing:result.models[0];
      el("model").value=options.value;
      el("model").hidden=true;
      options.hidden=false;
      el("model-manual").hidden=false;
      el("model-manual").textContent="Enter model manually";
      status.textContent=" "+result.models.length+" available model(s).";
    }else{
      useManualModel(" Server is running but reports no available models; enter one manually.");
    }
  }catch(error){
    if(version===modelDiscoveryVersion)useManualModel(" Model list unavailable: "+error.message);
  }
  if(version===modelDiscoveryVersion)await contextDefaults();
}
el("model-select").onchange=()=>{
  el("model").value=el("model-select").value;action(contextDefaults);
};
el("model").addEventListener("change",()=>action(contextDefaults));
el("hardware").onclick=()=>action(hardware);el("refresh").onclick=()=>action(runs);
el("basic").onclick=()=>action(async()=>{await basic();await contextDefaults()});
el("discover-models").onclick=()=>action(discoverModels);
el("model-manual").onclick=()=>{
  const button=el("model-manual");
  if(el("model").hidden){
    el("model").value=el("model-select").value;
    useManualModel(" Enter a model identifier manually.");
    button.hidden=false;button.textContent="Choose from server list";
  }else{
    const options=el("model-select");
    const entered=el("model").value;
    options.value=[...options.options].some(option=>option.value===entered)?
      entered:options.options[0].value;
    el("model").value=options.value;
    el("model").hidden=true;el("model-select").hidden=false;
    button.textContent="Enter model manually";
    action(contextDefaults);
  }
};
el("embedding").onclick=()=>{
  ++contextDefaultsVersion;
  el("cases").value=JSON.stringify([{id:"retrieval",task:"embedding",input:"Find a cat",
    documents:[{id:"cat",input:"A cat on a mat"},{id:"boat",input:"A boat at sea"}],
    evaluation:{metric:"retrieval",relevant_ids:["cat"],k:1}}],null,2);
  el("grid").value="{}";el("parameters").value="{}";
  action(contextDefaults);
};
el("runtime").onchange=()=>{
  const runtime=el("runtime").value;el("harness").value=runtime+"-v1";
  el("endpoint").value=discoveryInputs[runtime]?el(discoveryInputs[runtime]).value:"";
  el("endpoint").disabled=!["ollama","llama.cpp","colibri"].includes(runtime);
  el("grid").value=runtime==="ollama"?'{"context_size":[2048,4096]}':"{}";
  discoverModels();
};
el("endpoint").addEventListener("change",()=>{
  const id=discoveryInputs[el("runtime").value];
  if(id)el(id).value=el("endpoint").value;
  action(async()=>{await detectRuntimes();await discoverModels()});
});
el("compare").onclick=()=>action(async()=>{
  const ids=[...el("runs").querySelectorAll("input:checked")].map(c=>c.value);
  show(await api("/api/compare?"+ids.map(id=>"id="+encodeURIComponent(id)).join("&")));
});
el("start").onclick=()=>action(async()=>{
  const model=el("model-select").hidden?el("model").value:el("model-select").value;
  const config={model,harness:el("harness").value,cwd:el("cwd").value,
    repetitions:Number(el("repetitions").value),timeout_seconds:Number(el("timeout").value),
    seed:Number(el("seed").value),parameters:JSON.parse(el("parameters").value),
    parameter_grid:JSON.parse(el("grid").value),cases:JSON.parse(el("cases").value)};
  const runtime=el("runtime").value;
  const job=await api("/api/jobs",{config,runtime,
    endpoint:el("endpoint").value,
    command:runtime==="custom"?JSON.parse(el("command").value):undefined});
  el("start").disabled=true;
  el("benchmark-progress").hidden=false;
  el("trial-progress").value=0;
  el("progress-label").textContent="0% — preparing benchmark";
  try{
    while(true){
      const state=await api("/api/job?id="+encodeURIComponent(job.id));
      el("job").textContent=JSON.stringify(state,null,2);
      const completed=state.completed_trials;
      const total=state.total_trials;
      const percent=total?Math.floor(completed/total*100):0;
      el("trial-progress").value=percent;
      el("progress-label").textContent=percent+"% — "+completed+" / "+total+
        " trials finished ("+state.status+")";
      await runs();
      if(state.status!=="running"){
        if(state.error)throw Error(state.error);
        if(state.run_id)show(await api("/api/run?id="+state.run_id));
        break;
      }
      await new Promise(resolve=>setTimeout(resolve,1500));
    }
  }finally{el("start").disabled=false}
});
el("machine").closest("details").addEventListener("toggle",event=>{
  if(event.target.open)action(hardware);
});
action(async()=>{await basic();await runs()});
action(async()=>{await detectRuntimes();await discoverModels()});
</script></body></html>"""


class BenchmarkServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, db, base_dir):
        super().__init__(address, Handler)
        self.db = str(Path(db).resolve())
        self.base_dir = Path(base_dir).resolve()
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.lock = threading.Lock()
        with ht.connect(self.db):
            pass

    def start_job(self, body):
        """Start a new benchmark job with the given request body.

        Args:
            body (dict[str, Any]): Job configuration and runtime information.

        Returns:
            dict[str, str]: A dictionary containing the job ID.

        Raises:
            ValueError: If the request body is invalid or if a benchmark is already running.
        """
        if not isinstance(body, dict) or not isinstance(body.get("config"), dict):
            raise ValueError("Request requires a config object")
        runtime = body.get("runtime")
        config = dict(body["config"])
        if runtime == "custom":
            config["command"] = body.get("command")
        elif runtime in ("ollama", "llama.cpp", "airllm", "colibri"):
            import runtime_adapters

            config["command"] = [
                sys.executable,
                str(Path(runtime_adapters.__file__).resolve()),
                "--runtime",
                runtime,
            ]
            if runtime in ("ollama", "llama.cpp", "colibri"):
                from runtime_adapters import validate_endpoint

                endpoint = body.get("endpoint")
                if not isinstance(endpoint, str):
                    raise ValueError("Server endpoint must be a URL string")
                validate_endpoint(endpoint)
                config["command"].extend(["--endpoint", endpoint])
        else:
            raise ValueError("Unknown runtime")
        ht.validate_config(config)
        cwd = (self.base_dir / config.get("cwd", ".")).resolve()
        if not cwd.is_dir():
            raise ValueError(f"Working directory does not exist: {cwd}")
        config["cwd"] = str(cwd)
        # Media paths in the browser are relative to the selected working directory.
        ht.prepare_cases(config["cases"], cwd)
        with self.lock:
            if any(job["status"] == "running" for job in self.jobs.values()):
                raise ValueError("A benchmark is already running; avoid competing measurements")
            job_id = str(uuid.uuid4())
            self.jobs[job_id] = {
                "id": job_id,
                "status": "running",
                "completed_trials": 0,
                "total_trials": len(ht.parameter_sets(config))
                * len(config["cases"])
                * config.get("repetitions", 3),
            }
        worker = threading.Thread(target=self.run_job, args=(job_id, config, cwd), daemon=True)
        worker.start()
        return {"id": job_id}

    def run_job(self, job_id, config, base_dir):
        """Execute a benchmark job with the given configuration and working directory.

        Args:
            job_id (str): The unique identifier of the job.
            config (dict[str, Any]): The job configuration.
            base_dir (Path): The base directory for the job's working environment.

        Update the job status on completion or failure.
        """

        def update_progress(run_id, completed, total):
            with self.lock:
                self.jobs[job_id].update(
                    {
                        "run_id": run_id,
                        "completed_trials": completed,
                        "total_trials": total,
                    }
                )

        try:
            with ht.connect(self.db) as connection:
                run_id = ht.run_benchmark(connection, config, base_dir, progress=update_progress)
                status = ht.get_run(connection, run_id)["status"]
            outcome = {"status": status, "run_id": run_id}
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print(f"Web benchmark {job_id} failed: {exc}", file=sys.stderr)
            outcome = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        with self.lock:
            self.jobs[job_id].update(outcome)


class Handler(BaseHTTPRequestHandler):
    """HTTP request handler for the web-based benchmark interface.

    Attributes:
        server (BenchmarkServer): The benchmark server instance associated with this handler.
    """

    server: BenchmarkServer

    def send_json(self, value, status=200):
        payload = json.dumps(value, allow_nan=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def trusted_host(self):
        """Check if the request comes from a trusted loopback host.

        Returns:
            bool: Whether the Host header matches a trusted loopback address.
        """
        return self.headers.get("Host") in (
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        )

    def authorised(self):
        """Check if the request is authorised with a valid local session token.

        Returns:
            bool: True if the request is from a trusted host and the token matches, False otherwise.
        """
        supplied = self.headers.get("X-Harness-Token", "")
        return self.trusted_host() and secrets.compare_digest(supplied, self.server.token)

    def do_GET(self):
        """Handle HTTP GET requests for the web-based benchmark interface."""
        if not self.trusted_host():
            self.send_json({"error": "Only loopback Host headers are allowed"}, 403)
            return
        path = urlsplit(self.path)
        if path.path == "/":
            payload = HTML.replace("__TOKEN__", self.server.token).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header(
                "Content-Security-Policy",
                "default-src 'self'; script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'",
            )
            self.end_headers()
            self.wfile.write(payload)
            return
        if not self.authorised():
            self.send_json({"error": "Missing or invalid local session token"}, 403)
            return
        query = parse_qs(path.query)
        try:
            if path.path == "/api/machine":
                value = ht.environment_info()
            elif path.path == "/api/suite":
                value = basic_suite()
            elif path.path == "/api/models":
                runtime = query.get("runtime", [""])[0]
                endpoint = query.get("endpoint", [None])[0]
                if runtime not in ("ollama", "llama.cpp", "colibri"):
                    raise ValueError(
                        "Model discovery is available for Ollama, llama.cpp, and Colibri"
                    )
                from runtime_adapters import list_models

                value = {"runtime": runtime, "models": list_models(runtime, endpoint)}
            elif path.path == "/api/runtimes":
                from runtime_adapters import DEFAULT_ENDPOINTS, runtime_availability

                runtimes = [*DEFAULT_ENDPOINTS, "custom"]
                with ThreadPoolExecutor(max_workers=len(runtimes)) as executor:
                    checks = {
                        runtime: executor.submit(
                            runtime_availability, runtime, query.get(runtime, [None])[0]
                        )
                        for runtime in runtimes
                    }
                    value = {runtime: check.result() for runtime, check in checks.items()}
            elif path.path == "/api/context-defaults":
                from runtime_adapters import context_defaults

                model = query.get("model", [""])[0]
                runtime = query.get("runtime", [""])[0]
                if runtime == "airllm" and model:
                    model = str(
                        (
                            self.server.base_dir
                            / query.get("cwd", ["."])[0]
                            / Path(model).expanduser()
                        ).resolve()
                    )
                value = context_defaults(
                    runtime,
                    model,
                    query.get("endpoint", [None])[0],
                )
            elif path.path == "/api/job":
                with self.server.lock:
                    value = dict(self.server.jobs.get(query.get("id", [""])[0], {}))
                if not value:
                    raise ValueError("Unknown job")
            else:
                with ht.connect(self.server.db) as connection:
                    run_id = query.get("id", [""])[0]
                    if path.path == "/api/runs":
                        value = [
                            dict(row)
                            for row in connection.execute(
                                "SELECT id,created_at,status,scope FROM runs "
                                "ORDER BY created_at DESC"
                            )
                        ]
                    elif path.path == "/api/run":
                        value = ht.summarise(connection, run_id)
                        value["config"] = json.loads(ht.get_run(connection, run_id)["config"])
                        value["trials"] = [
                            {
                                **dict(row),
                                "parameters": json.loads(row["parameters"]),
                                "response": (
                                    json.loads(row["response"]) if row["response"] else None
                                ),
                            }
                            for row in connection.execute(
                                "SELECT * FROM trials WHERE run_id = ?", (run_id,)
                            )
                        ]
                    elif path.path == "/api/recommend":
                        value = ht.recommend(
                            connection,
                            run_id,
                            query.get("objective", ["quality"])[0],
                            float(query.get("min_quality", ["0"])[0]),
                        )
                    elif path.path == "/api/compare":
                        value = ht.compare(connection, query.get("id", []))
                    else:
                        self.send_json({"error": "Unknown endpoint"}, 404)
                        return
            self.send_json(value)
        except RuntimeError as exc:
            self.send_json({"error": str(exc)}, 502)
        except (ValueError, OSError, sqlite3.Error) as exc:
            self.send_json({"error": str(exc)}, 400)

    def do_POST(self):
        """Handle HTTP POST requests for the web-based benchmark interface."""
        if not self.authorised():
            self.send_json({"error": "Missing or invalid local session token"}, 403)
            return
        origin = self.headers.get("Origin")
        allowed = (
            f"http://127.0.0.1:{self.server.server_port}",
            f"http://localhost:{self.server.server_port}",
        )
        if origin is not None and origin not in allowed:
            self.send_json({"error": "Cross-origin requests are forbidden"}, 403)
            return
        if self.path != "/api/jobs":
            self.send_json({"error": "Unknown endpoint"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 2 * 1024 * 1024:
                raise ValueError("Request must be between 1 byte and 2 MiB")
            if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
                raise ValueError("Content-Type must be application/json")
            body = json.loads(self.rfile.read(length), parse_constant=ht.reject_constant)
            self.send_json(self.server.start_job(body), 202)
        except (ValueError, OSError) as exc:
            self.send_json({"error": str(exc)}, 400)


def serve(db, port=8765):
    """Start the web-based benchmark server.

    Args:
        db (str): The path to the SQLite3 database file.
        port (int): The port number to bind the server to (default: 8765).

    Raises:
        ValueError: If the port number is out of the valid range (0-65535).
        OSError: If the server fails to start due to an OS-level error.
    """
    if not 0 <= port <= 65535:
        raise ValueError("Port must be between 0 and 65535")
    server = BenchmarkServer(("127.0.0.1", port), db, Path.cwd())
    print(f"Web UI: http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main():
    """Entry point for the web-based benchmark server command-line interface.

    Returns:
        int: Exit code (0 success, 130 interrupt, 1 error).
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default="results.sqlite3")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    try:
        serve(args.db, args.port)
    except KeyboardInterrupt:
        return 130
    except (ValueError, OSError, sqlite3.Error) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
