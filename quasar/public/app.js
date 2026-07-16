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
  view: '2d',      // '2d' | '3d'
  yaw: 0.62,       // 3D rotation, radians
  stepFreeOnly: false,
  lastAmenity: null,  // the attendee's last lookup, so changing language re-runs it
  lastAsk: null,      // the attendee's last concierge question, for the same reason
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

function showGroup(g, show) {
  if (show) g.removeAttribute('hidden');
  else g.setAttribute('hidden', '');
}

function kindOf(n) {
  if (n.tags.includes('gate')) return 'gate';
  if (n.tags.includes('medical')) return 'medical';
  if (n.tags.includes('seating')) return 'seating';
  if (n.tags.includes('concourse')) return 'concourse';
  if (n.tags.includes('service')) return 'service';
  return 'other';
}

// Dispatch between the flat plan and the isometric 3D view. The 2D plan is the
// authoritative operational view; the 3D view exists for accessibility — you can
// see, at a glance, which way up to a stand is a ramp and which is a staircase.
function renderMap() {
  const threeD = state.view === '3d';
  // NB: #flip and #iso are SVG <g> elements. SVGElement does not reflect the
  // `.hidden` IDL property to the attribute the way HTMLElement does, so setting
  // `.hidden` silently does nothing. Toggle the attribute directly.
  showGroup($('flip'), !threeD);
  showGroup($('iso'), threeD);
  $('view-2d').classList.toggle('active', !threeD);
  $('view-3d').classList.toggle('active', threeD);
  $('view-hint').textContent = threeD
    ? 'Drag to rotate · green = step-free ramp, red = stair'
    : 'Plan view · switch to 3D to see levels and step-free routes';
  if (threeD) render3D();
  else drawMap();
}

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

/* ── isometric 3D ────────────────────────────────────────────────────
 * A hand-rolled axonometric projection of the venue graph — no WebGL, no
 * dependencies, so it renders under the same content-security policy as
 * everything else. Levels are stacked (service below, concourse, bowl on top),
 * and the vertical connectors are colour-coded: a wheelchair user can see which
 * ramps get them up and which staircases do not. That is the accessibility point.
 */

const SVGNS = 'http://www.w3.org/2000/svg';
const LEVEL_LIFT = 34;   // screen units a level rises per storey
const ISO_TILT = 0.52;   // vertical squash of the ground plane

function project(n) {
  // Rotate about the vertical axis by yaw, then squash y for the ground plane and
  // lift by level. Classic axonometric: no perspective, so parallel lines stay
  // parallel and the plan stays readable.
  const c = Math.cos(state.yaw), s = Math.sin(state.yaw);
  const rx = n.x * c - n.y * s;
  const ry = n.x * s + n.y * c;
  return { x: rx, y: ry * ISO_TILT - (n.level || 0) * LEVEL_LIFT, depth: ry, level: n.level || 0 };
}

function convexHull(pts) {
  if (pts.length < 3) return pts;
  const p = [...pts].sort((a, b) => a.x - b.x || a.y - b.y);
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [], upper = [];
  for (const q of p) { while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], q) <= 0) lower.pop(); lower.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], q) <= 0) upper.pop(); upper.push(q); }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

function render3D() {
  const g = $('iso');
  g.innerHTML = '';
  const { nodes, edges } = state.venue;
  const P = Object.fromEntries(nodes.map((n) => [n.id, project(n)]));

  // Fit the projection to the viewBox.
  const xs = Object.values(P).map((p) => p.x), ys = Object.values(P).map((p) => p.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs), minY = Math.min(...ys), maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY) || 1;
  const scale = 300 / span;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  const tx = (p) => (p.x - cx) * scale;
  const ty = (p) => (p.y - cy) * scale;

  const add = (tag, attrs, cls) => {
    const el = document.createElementNS(SVGNS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    if (cls) el.setAttribute('class', cls);
    g.appendChild(el);
    return el;
  };

  // 1. Level plates, drawn bottom-up so higher decks sit on top.
  const levels = [...new Set(nodes.map((n) => n.level || 0))].sort((a, b) => a - b);
  for (const lvl of levels) {
    const hull = convexHull(nodes.filter((n) => (n.level || 0) === lvl).map((n) => ({ x: tx(P[n.id]), y: ty(P[n.id]) })));
    if (hull.length >= 3) {
      add('polygon', { points: hull.map((p) => `${p.x},${p.y}`).join(' ') }, `plate L${lvl}`);
    }
  }

  const stepFree = state.stepFreeOnly;

  // 2. Edges, painter-sorted by mean depth so nearer ones draw last.
  const sorted = [...edges].sort((a, b) => (P[a.u].depth + P[a.v].depth) - (P[b.u].depth + P[b.v].depth));
  for (const e of sorted) {
    const u = P[e.u], v = P[e.v];
    let cls = 'iso-edge ';
    if (state.cordoned.has(e.id)) cls += 'cordoned';
    else if (e.kind === 'ramp') cls += 'ramp';
    else if (e.kind === 'stair') cls += 'stair';
    else if (e.staff_only) cls += 'service';
    else cls += 'los-' + los(state.density[e.id] ?? 0);
    // In "step-free only" mode, dim everything a wheelchair user cannot take.
    if (stepFree && !e.step_free) cls += ' dim';
    const line = add('line', { x1: tx(u), y1: ty(u), x2: tx(v), y2: ty(v) }, cls);
    const t = document.createElementNS(SVGNS, 'title');
    t.textContent = `${e.id} · ${e.kind} · ${e.width_m} m${e.step_free ? ' · step-free' : ' · STEPPED'}`;
    line.appendChild(t);
  }

  // 3. Routes (medic / fan), in 3D.
  for (const [key, route] of Object.entries(state.routes)) {
    const pts = route.nodes.map((n) => `${tx(P[n])},${ty(P[n])}`).join(' ');
    add('polyline', { points: pts }, 'iso-route ' + (key.startsWith('medic') ? 'medic' : 'fan'));
  }

  // 4. Nodes with a short pillar to their deck, back-to-front.
  const nodeOrder = [...nodes].sort((a, b) => P[a.id].depth - P[b.id].depth);
  for (const n of nodeOrder) {
    const p = P[n.id];
    const kind = kindOf(n);
    const casualty = n.id === state.venue.casualty_node;
    // pillar down to the ground plane (level 0 projection of the same x,y)
    const base = ty({ x: p.x, y: p.y + (n.level || 0) * LEVEL_LIFT / ISO_TILT, level: 0 });
    if ((n.level || 0) > 0) add('line', { x1: tx(p), y1: ty(p), x2: tx(p), y2: base }, 'iso-pillar');
    const dot = add('circle', {
      cx: tx(p), cy: ty(p),
      r: casualty ? 4 : kind === 'gate' || kind === 'seating' ? 3.2 : 2.2,
    }, 'iso-node node ' + (casualty ? 'casualty' : kind));
    const t = document.createElementNS(SVGNS, 'title');
    t.textContent = `${n.id} — ${n.name} (level ${n.level})`;
    dot.appendChild(t);
    if (kind === 'gate' || kind === 'seating' || kind === 'medical') {
      add('text', { x: tx(p), y: ty(p) - 6 }, 'iso-label').textContent = n.id;
    }
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
  renderMap();
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
  state.routes = {}; state.cordoned.clear(); renderMap();

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
  renderMap();

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
  state.lastAsk = utterance;
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
      renderMap();
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

function rerunConcierge() {
  if (!state.lastAsk) return;
  const cheap = mode() === 'recorded' || mode() === 'partition';
  if (cheap) { askConcierge(state.lastAsk); return; }
  // Live/edge: don't spend an inference call on a language change the fan can
  // confirm. Leave the existing reply and drop a one-line prompt above it.
  const out = $('fan-result');
  const prior = out.querySelector('.lang-hint');
  if (prior) prior.remove();
  if (!out.childElementCount) return;   // nothing asked yet on screen
  out.prepend(el('div', 'lang-hint',
    `Language set to ${$('fan-lang').value}. Press Ask to answer again in it.`));
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

/* ── attendee companion ──────────────────────────────────────────── */

async function loadAmenities() {
  const grid = $('amenity-grid');
  grid.innerHTML = '';
  $('wayfind-result').innerHTML = '';
  const data = await api('amenities');
  const byGroup = {};
  for (const a of data.amenities) (byGroup[a.group] ||= []).push(a);
  for (const g of data.groups) {
    const items = byGroup[g.key] || [];
    if (!items.length) continue;
    const sec = el('div', 'amenity-group');
    sec.appendChild(el('h4', null, g.label));
    const row = el('div', 'amenity-grid');
    for (const a of items) {
      const b = el('button', 'amenity-btn');
      b.type = 'button';
      b.disabled = !a.available;
      if (!a.available) b.title = 'Not mapped at this venue';
      b.appendChild(el('span', 'ico', a.icon));
      b.appendChild(el('span', null, a.label));
      b.onclick = () => wayfindTo(a.key);
      row.appendChild(b);
    }
    sec.appendChild(row);
    grid.appendChild(sec);
  }
}

async function wayfindTo(key) {
  state.lastAmenity = key;
  const out = $('wayfind-result');
  out.innerHTML = '<p class="spin">Finding you the best way…</p>';
  try {
    const d = await api('wayfind', {
      from_node: $('fan-node').value,
      amenity: key,
      accessible: $('fan-accessible').checked,
      calm: $('fan-calm').checked,
      language: $('fan-lang').value,
      seat: $('fan-seat').value,
      cordoned: [...state.cordoned],
    });
    out.innerHTML = '';
    if (d.route) {
      // Draw it on the shared map (visible on the left in every tab).
      state.routes = { ...state.routes, ['fan:' + d.route.destination]: d.route };
      renderMap();
    }
    const card = el('div', 'wayfind-card' + (d.route ? '' : ' miss'));
    const big = el('div', 'big');
    big.append(el('span', 'ico', d.icon), el('span', null, d.destination ? d.destination.name : d.label));
    card.appendChild(big);
    // The message and the notes are already in the fan's language; the step list is
    // node ids, which are universal. Nothing here is composed in English on the client.
    card.appendChild(el('div', 'msg', d.message));
    if (d.destination && d.destination.info) card.appendChild(el('div', 'info', d.destination.info));
    if (d.route) card.appendChild(el('div', 'steps', d.route.nodes.join(' → ')));
    for (const note of d.notes || []) card.appendChild(el('div', 'info', note));
    out.appendChild(card);
  } catch (e) {
    out.innerHTML = '';
    out.appendChild(el('div', 'reason', e.message));
  }
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
  state.audit = []; state.lastAmenity = null; state.lastAsk = null;

  const [venue, s] = await Promise.all([api('venue'), api('state')]);
  state.venue = venue;
  // Show the tournament name where FIFA uses one, with the real name in parentheses.
  const title = venue.fifa_name && venue.fifa_name !== venue.name
    ? `${venue.fifa_name} (${venue.name})` : venue.name;
  $('venue-name').textContent = `${title} · ${venue.city} · ${venue.capacity.toLocaleString()}`;

  // The honesty stamp: a representative graph is not a surveyed floor plan, and the
  // console never lets you forget it.
  const prov = $('provenance');
  if (venue.topology === 'representative') {
    prov.hidden = false;
    prov.textContent = 'Representative topology — a model fitted to this venue’s '
      + 'public capacity and gate count, not a surveyed floor plan. Correct for scale '
      + 'and planning; not for live operations.';
  } else {
    prov.hidden = true;
  }

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
  await loadAmenities();
  renderState(s);
  renderAudit();
}

/* ── boot ────────────────────────────────────────────────────────── */

/* ── theme ───────────────────────────────────────────────────────── */

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  const btn = $('theme-toggle');
  if (btn) btn.textContent = theme === 'dark' ? '☀️' : '🌙';
  try { localStorage.setItem('quasar-theme', theme); } catch (_) {}
}

function wireTheme() {
  // Light by default — the console should feel welcoming, not like a terminal.
  // ?theme= wins (deep links, demos), then a remembered choice, then light.
  const q = new URLSearchParams(location.search).get('theme');
  let theme = q === 'dark' || q === 'light' ? q : null;
  if (!theme) { try { theme = localStorage.getItem('quasar-theme'); } catch (_) {} }
  applyTheme(theme || 'light');
  $('theme-toggle').onclick = () => {
    const next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
    applyTheme(next);
  };
}

function wireTabs() {
  for (const tab of document.querySelectorAll('.tab')) {
    tab.onclick = () => {
      document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
      for (const id of ['guide', 'control', 'fan', 'ready', 'premat', 'audit']) {
        $('tab-' + id).hidden = id !== tab.dataset.tab;
      }
    };
  }
}

async function renderGuide() {
  // The guide is the problem statement mapped to the console. Each item ends in a
  // deep link into the feature that proves it, so nothing here is a claim you have
  // to take on trust — the button walks you to where you can watch it hold.
  const guide = await api('guide');
  $('guide-challenge').textContent = guide.challenge;
  $('guide-thesis').textContent = guide.thesis;

  const host = $('guide-sections');
  host.replaceChildren();
  for (const section of guide.sections) {
    const group = el('section', 'guide-group');
    group.appendChild(el('h3', 'guide-heading', section.heading));
    for (const it of section.items) {
      const card = el('article', 'card');
      const h = el('h2');
      h.appendChild(el('span', 'gi', it.icon));
      h.appendChild(document.createTextNode(' ' + it.title));
      card.appendChild(h);
      card.appendChild(el('p', 'sub', it.summary));
      card.appendChild(el('p', null, it.how));

      const mods = el('div', 'modules');
      for (const m of it.modules) mods.appendChild(el('code', 'chip', m));
      card.appendChild(mods);

      const go = el('button', 'primary show-me', it.cta + ' →');
      go.onclick = () => goTo(it.where);
      card.appendChild(go);

      group.appendChild(card);
    }
    host.appendChild(group);
  }
}

function goTo(where) {
  // Reuse the boot-time deep-link handling rather than re-implementing tab
  // switching, venue loading and incident driving here: set the query and reload.
  const url = new URL(location.href);
  url.search = new URLSearchParams(where).toString();
  location.assign(url.toString());
}

function wireMapView() {
  $('view-2d').onclick = () => { state.view = '2d'; renderMap(); };
  $('view-3d').onclick = () => { state.view = '3d'; renderMap(); };
  $('step-free-only').onchange = (e) => { state.stepFreeOnly = e.target.checked; renderMap(); };

  // Drag the 3D view to rotate about the vertical axis.
  const svg = $('map');
  let dragging = false, lastX = 0;
  svg.addEventListener('pointerdown', (e) => {
    if (state.view !== '3d') return;
    dragging = true; lastX = e.clientX; svg.setPointerCapture(e.pointerId);
  });
  svg.addEventListener('pointermove', (e) => {
    if (!dragging) return;
    state.yaw += (e.clientX - lastX) * 0.01;
    lastX = e.clientX;
    render3D();
  });
  const stop = (e) => { dragging = false; try { svg.releasePointerCapture(e.pointerId); } catch (_) {} };
  svg.addEventListener('pointerup', stop);
  svg.addEventListener('pointercancel', stop);
}

async function boot() {
  wireTheme();
  wireTabs();
  wireMapView();
  await renderGuide();

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

  // Changing the language (or any routing constraint) re-runs the last amenity
  // lookup so the result switches immediately, rather than waiting for the next
  // tap. Wired once here; the controls persist across venue loads.
  const rerunWayfind = () => { if (state.lastAmenity) wayfindTo(state.lastAmenity); };
  for (const id of ['fan-lang', 'fan-node', 'fan-seat', 'fan-accessible', 'fan-calm']) {
    $(id).addEventListener('change', rerunWayfind);
  }
  // The wayfinder is deterministic and free to re-run; the concierge is a model
  // path, so it re-answers automatically only on a language change and only when
  // inference costs nothing (recorded transcripts or the deterministic partition).
  // In live/edge, a re-answer is a real call, so we prompt rather than spend it.
  $('fan-lang').addEventListener('change', rerunConcierge);
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
  if (q.get('view') === '3d') { state.view = '3d'; }
  picker.value = chosen;
  await loadVenue(chosen);

  const tab = q.get('tab');
  if (tab) document.querySelector(`.tab[data-tab="${tab}"]`)?.click();
  const find = q.get('find');
  if (find) await wayfindTo(find);

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
