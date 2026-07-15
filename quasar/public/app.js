'use strict';

/* Quasar control room.
 *
 * The console renders what the server computed. It does not compute anything
 * safety-relevant itself -- if it did, it would be a second, drifting
 * implementation of the venue's logic, and the two would disagree on the night
 * it mattered. Every route, every level of service, every queue metric on this
 * page came from the deterministic plane over the wire.
 */

const state = {
  venueId: null,
  venue: null,
  venues: [],
  density: {},
  cordoned: new Set(),
  routes: {},
  plan: null,
  brief: null,
  audit: [],
};

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};

const mode = () => $('mode').value;
const liveKey = () => $('live-key').value.trim();

async function api(path, body) {
  const opts = {
    method: body ? 'POST' : 'GET',
    headers: { 'content-type': 'application/json' },
  };
  if (liveKey()) opts.headers['x-quasar-key'] = liveKey();
  // Every call is scoped to a venue. There is no ambient "the venue" any more.
  if (body) opts.body = JSON.stringify({ venue: state.venueId, ...body });
  const qs = !body && state.venueId ? `?venue=${encodeURIComponent(state.venueId)}` : '';

  const res = await fetch(`/api/${path}${qs}`, opts);
  const data = await res.json().catch(() => ({ error: 'bad_response' }));
  if (!res.ok) {
    const err = new Error(data.detail || data.error || `HTTP ${res.status}`);
    err.code = data.error;
    err.status = res.status;
    throw err;
  }
  return data;
}

/* ── level of service ──────────────────────────────────────────────
 * Mirrors quasar.crowd.level_of_service purely to COLOUR the map. The letter
 * shown in every table comes from the server; this only decides a stroke.     */
function los(d) {
  if (d < 0.308) return 'A';
  if (d < 0.431) return 'B';
  if (d < 0.719) return 'C';
  if (d < 1.075) return 'D';
  if (d < 2.174) return 'E';
  return 'F';
}

/* ── map ─────────────────────────────────────────────────────────── */

function drawMap() {
  const { nodes, edges } = state.venue;
  const byId = Object.fromEntries(nodes.map((n) => [n.id, n]));
  const gEdges = $('edges');
  const gNodes = $('nodes');
  const gLabels = $('labels');
  gEdges.innerHTML = gNodes.innerHTML = gLabels.innerHTML = '';

  for (const e of edges) {
    const u = byId[e.u], v = byId[e.v];
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', u.x); line.setAttribute('y1', u.y);
    line.setAttribute('x2', v.x); line.setAttribute('y2', v.y);
    line.setAttribute('stroke-width', Math.max(1.4, Math.min(e.width_m * 0.5, 4)));
    line.dataset.id = e.id;

    const d = state.density[e.id] ?? 0;
    let cls = 'edge ';
    if (state.cordoned.has(e.id)) cls += 'cordoned';
    else if (e.staff_only) cls += 'staff';
    else cls += 'los-' + los(d);
    line.setAttribute('class', cls);

    const t = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    t.textContent = `${e.id} · ${e.length_m} m × ${e.width_m} m · ${d.toFixed(2)} ped/m² · LOS ${los(d)}`
      + (e.staff_only ? ' · staff only' : '') + (e.step_free ? '' : ' · stepped');
    line.appendChild(t);
    gEdges.appendChild(line);
  }

  for (const n of nodes) {
    const c = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
    c.setAttribute('cx', n.x); c.setAttribute('cy', n.y);
    let kind = 'other';
    if (n.tags.includes('gate')) kind = 'gate';
    else if (n.tags.includes('medical')) kind = 'medical';
    else if (n.tags.includes('seating')) kind = 'seating';
    else if (n.tags.includes('concourse')) kind = 'concourse';
    else if (n.tags.includes('service')) kind = 'service';
    const casualty = n.id === state.venue.casualty_node;
    c.setAttribute('r', casualty ? 4 : kind === 'gate' || kind === 'seating' ? 3.2 : 2.2);
    c.setAttribute('class', 'node ' + (casualty ? 'casualty' : kind));
    const t = document.createElementNS('http://www.w3.org/2000/svg', 'title');
    t.textContent = `${n.id} — ${n.name}`;
    c.appendChild(t);
    gNodes.appendChild(c);

    if (kind === 'gate' || kind === 'seating' || kind === 'medical') {
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('x', n.x);
      label.setAttribute('y', -(n.y) - 6); // labels sit outside the flipped group
      label.setAttribute('class', 'map-label');
      label.textContent = n.id;
      gLabels.appendChild(label);
    }
  }
  drawRoutes();
}

function drawRoutes() {
  const byId = Object.fromEntries(state.venue.nodes.map((n) => [n.id, n]));
  const g = $('routes');
  g.innerHTML = '';
  for (const [key, route] of Object.entries(state.routes)) {
    const pts = route.nodes.map((n) => `${byId[n].x},${byId[n].y}`).join(' ');
    const p = document.createElementNS('http://www.w3.org/2000/svg', 'polyline');
    p.setAttribute('points', pts);
    p.setAttribute('class', 'route ' + (key.startsWith('medic') ? 'medic' : 'fan'));
    g.appendChild(p);
  }
}

/* ── deterministic plane ─────────────────────────────────────────── */

function renderState(s) {
  state.density = s.density;

  const hot = $('hotspots').querySelector('tbody');
  hot.innerHTML = '';
  for (const h of s.hotspots) {
    const tr = el('tr');
    tr.append(
      el('td', null, h.edge_id),
      el('td', 'num', h.density.toFixed(2)),
      Object.assign(el('td', 'los los-' + h.los, h.los)),
      el('td', null, h.trend),
    );
    hot.appendChild(tr);
  }

  const gates = $('gates').querySelector('tbody');
  gates.innerHTML = '';
  for (const [gid, g] of Object.entries(s.gates)) {
    const tr = el('tr');
    tr.append(
      el('td', null, gid),
      el('td', 'num', g.utilisation.toFixed(2)),
      el('td', 'num', `${g.open_lanes}/${g.installed_lanes}`),
      el('td', g.breaches ? 'breach' : 'fine',
        g.breaches ? `breach — needs ${g.lanes_needed}` : 'within trigger'),
    );
    gates.appendChild(tr);
  }

  $('floor').innerHTML =
    `<b>Severity floor: ${s.severity_floor}</b> — ${s.severity_floor_reason}`;
  $('incident-text').textContent = `“${s.incident.text}”`;
  drawMap();
}

/* ── agents ──────────────────────────────────────────────────────── */

function renderAgent(r, title, bodyFn) {
  const box = el('div', 'agent');
  const head = el('header');
  head.append(el('h3', null, title), el('span', 'badge ' + r.source, r.source.replace(/_/g, ' ')));
  box.appendChild(head);

  const body = el('div', 'body');
  bodyFn(body, r.payload);
  box.appendChild(body);

  if (r.fallback_reason) {
    box.appendChild(el('div', 'reason', '↩ ' + r.fallback_reason));
  }

  const conf = `self-reported ${r.self_reported_confidence.toFixed(2)} · `
    + `corroborated ${r.corroboration_score.toFixed(2)} · `
    + `effective ${r.effective_confidence.toFixed(2)} (floor 0.85)`;
  box.appendChild(el('div', 'meta', `${r.plane} · ${conf}`));
  if (r.corroboration_notes.length) {
    box.appendChild(el('div', 'meta', '⚠ ' + r.corroboration_notes.join('; ')));
  }
  return box;
}

function citations(parent, list) {
  const wrap = el('div', 'cites');
  for (const c of list) wrap.appendChild(el('span', 'cite', `${c.doc_id}#${c.section}`));
  parent.appendChild(wrap);
}

async function runIncident() {
  const btn = $('run');
  btn.disabled = true;
  const agents = $('agents');
  agents.innerHTML = '';
  $('agents-card').hidden = false;
  $('plan-card').hidden = true;
  $('exec-card').hidden = true;
  state.routes = {}; state.cordoned.clear(); drawMap();

  const spinner = el('p', 'spin', mode() === 'edge' ? 'Running real inference on the edge model… (10–30s per agent)' : 'Running…');
  agents.appendChild(spinner);

  try {
    const crowd = await api('agent', { agent: 'crowd', mode: mode(), audit: state.audit });
    state.audit = crowd.audit;
    spinner.remove();
    agents.appendChild(renderAgent(crowd.result, 'CrowdIntelligenceAgent', (b, p) => {
      b.appendChild(el('p', null, p.summary));
    }));

    const inc = await api('agent', { agent: 'incident', mode: mode(), audit: state.audit });
    state.audit = inc.audit;
    state.brief = inc.result.payload;
    agents.appendChild(renderAgent(inc.result, 'IncidentResponseAgent', (b, p) => {
      b.appendChild(el('p', null, `${p.severity} · ${p.category} · ${p.affected_zones.join(', ')}`));
      b.appendChild(el('p', null, p.situation));
      citations(b, p.citations);
    }));

    const plan = await api('agent', {
      agent: 'plan', mode: mode(), brief: state.brief, audit: state.audit,
    });
    state.audit = plan.audit;
    state.plan = plan.result.payload;
    agents.appendChild(renderAgent(plan.result, 'PlannerAgent', (b, p) => {
      b.appendChild(el('p', null, p.rationale));
    }));

    renderPlan(state.plan);
  } catch (e) {
    spinner.remove();
    agents.appendChild(el('div', 'reason', e.message));
  } finally {
    btn.disabled = false;
    renderAudit();
  }
}

function renderPlan(plan) {
  const ol = $('plan-actions');
  ol.innerHTML = '';
  for (const a of plan.actions) {
    const li = el('li');
    const code = el('code', null, a.type);
    li.append(code, document.createTextNode(' ' + JSON.stringify(a.params) + ' '));
    li.appendChild(el('span', 'sop', a.sop_ref));
    ol.appendChild(li);
  }
  $('barrier-result').innerHTML = '';
  $('plan-card').hidden = false;
}

/* ── the barrier ─────────────────────────────────────────────────── */

async function tryActuate(approver, plan) {
  const out = $('barrier-result');
  out.innerHTML = '';
  const payload = {
    plan: plan || state.plan,
    approver,
    mode: mode(),
    audit: state.audit,
  };
  try {
    const data = await api('actuate', payload);
    state.audit = data.audit;
    out.appendChild(el('div', 'result ok',
      `Actuated. Signed by ${data.execution.approved_by}.`));
    renderExecution(data.execution);
  } catch (e) {
    // Being refused is the system working, so it renders as a result, not a crash.
    const map = {
      approval_required: 'Blocked by the human-in-the-loop barrier',
      not_authorised: 'Refused — insufficient authority',
      policy_violation: 'Refused by policy',
      schema_violation: 'Rejected at the schema barrier',
      rejected: 'Rejected by the corroborator',
    };
    out.appendChild(el('div', 'result blocked',
      `${map[e.code] || 'Refused'} — ${e.message}`));
  } finally {
    renderAudit();
  }
}

function tamperPlan() {
  // Exactly what a hostile client would do: edit the JSON in flight. Ask for more
  // lanes than the gate has, and reverse the announce-then-cordon ordering that
  // SOP-MED-03#3 requires. The server recomputes both against the venue.
  const plan = structuredClone(state.plan);
  const lanes = plan.actions.find((a) => a.type === 'OPEN_LANES');
  if (lanes) lanes.params.lanes = 16;
  const b = plan.actions.findIndex((a) => a.type === 'BROADCAST');
  const c = plan.actions.findIndex((a) => a.type === 'CORDON_EDGE');
  if (b > -1 && c > -1) [plan.actions[b], plan.actions[c]] = [plan.actions[c], plan.actions[b]];

  $('barrier-result').innerHTML = '';
  $('barrier-result').appendChild(el('div', 'result blocked',
    'Sending a doctored plan: 16 lanes at a 12-lane gate, and the corridor cordoned '
    + 'before the diversion is announced. The browser is not a trusted component…'));
  tryActuate('commander', plan);
}

/* ── execution ───────────────────────────────────────────────────── */

function renderExecution(x) {
  const out = $('exec');
  out.innerHTML = '';
  $('exec-card').hidden = false;

  state.cordoned = new Set(x.cordoned);
  state.routes = x.routes;
  drawMap();

  for (const [key, r] of Object.entries(x.routes)) {
    const m = el('div', 'metric');
    m.append(
      el('span', null, `${key} — ${r.nodes.join(' → ')}`),
      el('b', null, `${r.distance_m} m · ${Math.round(r.eta_s)} s`),
    );
    out.appendChild(m);
  }

  for (const [gid, after] of Object.entries(x.gate_after)) {
    const before = x.gate_before[gid];
    const m = el('div', 'metric');
    const b = el('b');
    b.append(
      document.createTextNode(before.toFixed(2)),
      el('span', 'arrow', ' → '),
      document.createTextNode(after.utilisation.toFixed(2)),
    );
    m.append(el('span', null, `Gate ${gid} utilisation (${after.open_lanes} lanes)`), b);
    out.appendChild(m);
  }

  for (const d of x.dispatches) {
    out.appendChild(el('h3', null, d.template_id));
    for (const a of d.announcements) {
      const box = el('div', 'announce');
      box.appendChild(el('div', 'lang', a.language));
      box.appendChild(el('div', null, a.text));
      out.appendChild(box);
    }
    for (const lang of d.refused_languages) {
      const box = el('div', 'announce refused');
      box.appendChild(el('div', 'lang', lang + ' — refused'));
      box.appendChild(el('div', null,
        'No human-validated translation exists for this template. The system does not '
        + 'machine-translate a safety-critical announcement, and does not stay silent: '
        + `pictogram ${d.pictogram} is raised and a steward is dispatched (SOP-COMMS-07#3).`));
      out.appendChild(box);
    }
  }

  for (const w of x.warnings) out.appendChild(el('div', 'warn', w));
  for (const e of x.escalations) out.appendChild(el('div', 'warn', 'Escalated — ' + e));
}

/* ── fan ─────────────────────────────────────────────────────────── */

async function askConcierge(utterance) {
  const out = $('fan-result');
  out.innerHTML = '<p class="spin">Asking…</p>';
  try {
    const data = await api('concierge', {
      utterance,
      language: $('fan-lang').value,
      at_node: $('fan-node').value,
      seat: $('fan-seat').value,
      accessible: $('fan-accessible').checked,
      mode: mode(),
      cordoned: [...state.cordoned],
    });
    out.innerHTML = '';
    const r = data.result;
    out.appendChild(renderAgent(r, 'ConciergeAgent', (b, p) => {
      b.appendChild(el('p', null, p.reply_text));
      b.appendChild(el('div', 'meta',
        `intent: ${p.intent} · tier: ${p.safety_tier} · destination: ${p.destination_tag ?? '—'}`));
      if (p.safety_tier === 'safety_critical') {
        b.appendChild(el('div', 'reason',
          'Classified safety-critical. The model no longer speaks: the controlled '
          + 'catalogue answers and a steward is dispatched.'));
      }
    }));

    if (data.route) {
      state.routes = { ...state.routes, ['fan:' + data.route.destination]: data.route };
      drawRoutes();
      const m = el('div', 'metric');
      m.append(
        el('span', null, `${data.route.profile} route → ${data.route.destination}`),
        el('b', null, `${data.route.distance_m} m · ${Math.round(data.route.eta_s)} s · worst LOS ${data.route.worst_los}`),
      );
      out.appendChild(m);
      out.appendChild(el('div', 'meta', data.route.nodes.join(' → ')));
    } else if (data.route_error) {
      out.appendChild(el('div', 'result blocked',
        'The router refused rather than improvising: ' + data.route_error
        + ' — a step-free fan is never sent down a staircase because the alternative was long.'));
    }
  } catch (e) {
    out.innerHTML = '';
    out.appendChild(el('div', 'reason', e.message));
  }
}

/* ── pre-match ───────────────────────────────────────────────────── */

async function runStress() {
  const out = $('stress-result');
  out.innerHTML = '<p class="spin">Firing scenarios…</p>';
  try {
    const data = await api('stress', { kind: $('stress-kind').value, n: 3 });
    out.innerHTML = '';
    for (const r of data.results) {
      out.appendChild(el('h3', null, `${r.name}${r.closed_edges.length ? ' · closed: ' + r.closed_edges.join(', ') : ''}`));
      if (!r.findings.length) {
        out.appendChild(el('div', 'finding', 'No findings — the venue survives this one.'));
      }
      for (const f of r.findings) {
        const box = el('div', 'finding ' + f.severity);
        box.appendChild(el('span', 'inv', `[${f.severity}] ${f.invariant}`));
        box.appendChild(document.createTextNode(f.detail));
        out.appendChild(box);
      }
    }
  } catch (e) {
    out.innerHTML = '';
    out.appendChild(el('div', 'reason', e.message));
  }
}

/* ── audit ───────────────────────────────────────────────────────── */

function renderAudit(broken = -1) {
  const list = $('audit-list');
  list.innerHTML = '';
  for (const r of state.audit) {
    const row = el('div', 'rec' + (broken > -1 && r.seq >= broken ? ' broken' : ''));
    row.append(
      el('span', 'seq', r.seq),
      el('span', null, r.event),
      el('span', 'hash', r.hash.slice(0, 16)),
    );
    list.appendChild(row);
  }
}

async function verifyChain() {
  const out = $('verify-result');
  const data = await api('verify', { audit: state.audit });
  out.innerHTML = '';
  out.appendChild(el('div', 'result ' + (data.valid ? 'ok' : 'blocked'),
    data.valid
      ? `Chain verifies — ${data.records} records, none altered.`
      : `Chain is broken: ${data.detail}`));
}

async function tamperAudit() {
  if (state.audit.length < 2) {
    $('verify-result').innerHTML = '';
    $('verify-result').appendChild(el('div', 'result blocked',
      'Run the incident first — there is nothing to tamper with yet.'));
    return;
  }
  const i = 1;
  state.audit[i].data = { ...state.audit[i].data, tampered: 'the operator never approved this' };
  renderAudit(i);
  await verifyChain();
}

/* ── readiness ───────────────────────────────────────────────────── */

async function runReadiness() {
  const out = $('ready-result');
  out.innerHTML = '<p class="spin">Auditing the venue…</p>';
  try {
    const d = await api('readiness', {});
    out.innerHTML = '';
    out.appendChild(el('div', 'result ' + (d.ready ? 'ok' : 'blocked'),
      d.ready
        ? `${d.name} is cleared to open — no blocking findings.`
        : `${d.name} is NOT cleared to open. ${d.checks.filter((c) => c.severity === 'blocker').length} blocking finding(s).`));

    for (const c of d.checks) {
      const box = el('div', 'finding ' + (c.severity === 'blocker' ? 'critical' : c.severity));
      box.appendChild(el('span', 'inv', `[${c.severity}] ${c.id}`));
      box.appendChild(el('div', null, c.title));
      box.appendChild(el('div', 'meta', c.detail));
      if (c.severity === 'blocker' || c.severity === 'critical') {
        box.appendChild(el('div', 'meta', '→ ' + c.remedy));
      }
      out.appendChild(box);
    }
  } catch (e) {
    out.innerHTML = '';
    out.appendChild(el('div', 'reason', e.message));
  }
}

/* ── venue switching ─────────────────────────────────────────────── */

async function loadVenue(venueId) {
  state.venueId = venueId;
  state.routes = {}; state.cordoned.clear(); state.plan = null; state.brief = null;
  state.audit = [];

  const [venue, s] = await Promise.all([api('venue'), api('state')]);
  state.venue = venue;
  $('venue-name').textContent = `${venue.name} · ${venue.city} · ${venue.capacity.toLocaleString()}`;

  // The fan panel is the venue's fan: their language, their seat, their words.
  const sel = $('fan-node');
  sel.innerHTML = '';
  for (const n of venue.nodes.filter((n) => !n.tags.includes('service'))) {
    const o = el('option', null, `${n.id} — ${n.name}`);
    o.value = n.id;
    if (n.id === venue.fan.at_node) o.selected = true;
    sel.appendChild(o);
  }
  const seats = $('fan-seat');
  seats.innerHTML = '';
  for (const n of venue.nodes.filter((n) => n.tags.includes('seating'))) {
    const o = el('option', null, `${n.id} — ${n.name}`);
    o.value = n.id;
    if (n.id === venue.fan.seat) o.selected = true;
    seats.appendChild(o);
  }
  const langs = $('fan-lang');
  langs.innerHTML = '';
  for (const code of venue.languages) {
    const o = el('option', null, code);
    o.value = code;
    if (code === venue.fan.language) o.selected = true;
    langs.appendChild(o);
  }
  $('fan-text').value = venue.fan.utterance;
  $('fan-accessible').checked = venue.fan.accessible;

  ['agents-card', 'plan-card', 'exec-card'].forEach((id) => { $(id).hidden = true; });
  $('ready-result').innerHTML = '';
  renderState(s);
  renderAudit();
}

/* ── boot ────────────────────────────────────────────────────────── */

function wireTabs() {
  for (const tab of document.querySelectorAll('.tab')) {
    tab.onclick = () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
      for (const id of ['control', 'fan', 'ready', 'premat', 'audit']) {
        $('tab-' + id).hidden = id !== tab.dataset.tab;
      }
    };
  }
}

async function boot() {
  wireTabs();

  $('mode').onchange = () => {
    const live = mode() === 'live';
    $('live-key').hidden = !live;
    const notes = {
      live: 'Live mode calls Claude for real. It needs the operator key — a public URL with an open API key is a stranger’s budget. Output goes through exactly the same barrier as a recording.',
      edge: 'REAL INFERENCE on the on-venue edge model (Ollama, no API key, no internet). It is a small model and it will sometimes fail the barrier — watch which agents fall back, and why. Expect 10–30s per agent.',
      partition: 'Model plane disabled — total network partition. Every agent takes its deterministic twin. Watch what survives, and what is lost.',
      recorded: 'Recorded model output. Nothing safety-critical is faked: the router, the queueing model, the catalogue, the schema validator, the grounding check, the corroborators and the human barrier all run for real.',
    };
    $('mode-note').textContent = notes[mode()];
  };
  $('mode').onchange();

  $('run').onclick = runIncident;
  $('approve-commander').onclick = () => tryActuate('commander');
  $('approve-steward').onclick = () => tryActuate('steward');
  $('approve-none').onclick = () => tryActuate(null);
  $('tamper-plan').onclick = tamperPlan;
  $('fan-ask').onclick = () => askConcierge($('fan-text').value);
  $('fan-emergency').onclick = () => askConcierge('मदत! एक माणूस पडला आहे, he has collapsed');
  $('stress-run').onclick = runStress;
  $('verify').onclick = verifyChain;
  $('tamper-audit').onclick = tamperAudit;
  $('ready-run').onclick = runReadiness;

  const { venues, default: fallback } = await api('venues');
  state.venues = venues;
  const picker = $('venue');
  for (const v of venues) {
    const o = el('option', null, `${v.name} — ${v.city} (${v.capacity.toLocaleString()})`);
    o.value = v.id;
    picker.appendChild(o);
  }
  picker.onchange = () => loadVenue(picker.value);

  const q = new URLSearchParams(location.search);
  const chosen = venues.some((v) => v.id === q.get('venue')) ? q.get('venue') : fallback;
  picker.value = chosen;
  await loadVenue(chosen);

  // Deep links: ?run=1 walks the incident, &approve=commander signs it. Useful for
  // demoing to someone who should not have to be told which buttons to press, and
  // for driving the page from a headless browser in CI.
  if (q.get('mode')) { $('mode').value = q.get('mode'); $('mode').onchange(); }
  if (q.get('run')) {
    await runIncident();
    const approver = q.get('approve');
    if (approver && state.plan) await tryActuate(approver);
  }
}

boot().catch((e) => {
  document.body.prepend(el('p', 'note', 'Failed to reach the venue: ' + e.message));
});
