/* Python runs in this worker; JavaScript only loads files and transports results. */
importScripts('https://cdn.jsdelivr.net/pyodide/v0.27.7/full/pyodide.js');
const root = new URL('../', self.location.href);
let runtime;
async function fetchFile(path) {
  const response = await fetch(new URL(path, root));
  if (!response.ok) throw new Error(`Cannot load ${path}: HTTP ${response.status}`);
  return new Uint8Array(await response.arrayBuffer());
}
function write(path, bytes) {
  runtime.FS.mkdirTree('/project/' + path.split('/').slice(0, -1).join('/'));
  runtime.FS.writeFile('/project/' + path, bytes);
}
async function boot() {
  self.postMessage({type:'status', message:'Loading Python and scientific packages…'});
  runtime = await loadPyodide({indexURL:'https://cdn.jsdelivr.net/pyodide/v0.27.7/full/',
    stdout: text => self.postMessage({type:'log', text}),
    stderr: text => self.postMessage({type:'log', text})});
  await runtime.loadPackage(['numpy','scipy','pyproj','micropip']);
  // Upstream pymoo Python sources, unchanged; its optional native extensions are not required.
  runtime.unpackArchive(await fetchFile('web/vendor/pymoo-0.6.1.3.zip'), 'zip', {extractDir:'/project'});
  await runtime.runPythonAsync("import micropip\nawait micropip.install(['autograd==1.7.0', 'cma==3.2.2', 'dill==0.3.9', 'Deprecated==1.2.18'])");
  const files = await (await fetch(new URL('web/python-files.json', root))).json();
  await Promise.all([...files,'config/config.json','config/profile_locations.json','web/runner.py'].map(async path => write(path, await fetchFile(path))));
  runtime.FS.chdir('/project');
  await runtime.runPythonAsync("import sys\nsys.path.insert(0, '/project')\nfrom pymoo.config import Config\nConfig.warnings['not_compiled'] = False\nexec(open('web/runner.py').read())");
}
let ready;
self.onmessage = async ({data}) => {
  try {
    ready ??= boot();
    await ready;
    if (!/^scenarios\/[a-z][a-z0-9_]*\/(?:cities-\d+\/)?$/.test(data.scenarioPath)) throw new Error('Invalid prepared scenario path');
    runtime.unpackArchive(await fetchFile('web/' + data.scenarioPath + 'graph.zip'), 'zip', {extractDir:'/project/scenario'});
    runtime.globals.set('browser_config_json', JSON.stringify(data.config));
    await runtime.runPythonAsync('run_browser(browser_config_json)');
  } catch (error) {
    self.postMessage({type:'error', message:error.message});
  }
};
