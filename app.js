import * as I from './core/impact.js';

const $ = (s) => document.querySelector(s);
const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;
const pts = (v, d = 1) => (v * 100).toFixed(d);
const fmt = (n, d = 0) => n.toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });

// ------------------------------------------------------------ compute
const world = I.buildWorld();
const model = I.fitImpact(world.hist);              // all Jan-Sep history scores today's list
const means = I.featureMeans(world.hist);
const today = world.today.map((p) => ({ p, ...I.score(model, p), why: I.reasons(model, p, means, 3) }));
const cvRun = I.createCrossFit(world, 5);           // honest evaluation, filled in asynchronously
let cv = null, cmp = null, curveData = null, calib = null, oracleTruth = null;
const calledShare = world.hist.reduce((s, p) => s + p.t, 0) / world.hist.length;
const state = { view: 'worklist', market: 'all', rank: 'impact', cap: null, frac: I.CONFIG.capacityFrac };

const ICON = `<svg class="ico" viewBox="0 0 64 64" aria-hidden="true"><rect x="9" y="5" width="40" height="52" rx="4" fill="#fff" stroke="#000" stroke-width="3"/><rect x="9" y="5" width="40" height="11" rx="3" fill="#37a38f" stroke="#000" stroke-width="3"/><path d="M15 34h7l4-9 6 17 5-12 3 4h7" fill="none" stroke="#000" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/><rect x="15" y="46" width="14" height="4" fill="#37a38f"/><circle cx="50" cy="51" r="10" fill="#6a4c91" stroke="#fff" stroke-width="2.5"/><path d="M45.5 51.5l3 3 6-6.5" fill="none" stroke="#fff" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

function listFor() {
  const rows = today.filter((r) => state.market === 'all' || r.p.market === Number(state.market));
  const cap = state.cap ?? Math.round(rows.length * I.CONFIG.capacityFrac);
  const byImpact = [...rows].sort((a, b) => b.uplift - a.uplift);
  const byRisk = [...rows].sort((a, b) => b.risk - a.risk);
  const setI = new Set(byImpact.slice(0, cap)), setR = new Set(byRisk.slice(0, cap));
  const shown = state.rank === 'impact' ? byImpact : byRisk;
  const sum = (set) => [...set].reduce((s, r) => s + r.uplift, 0);
  return { rows, cap, shown, setI, setR, avoidI: sum(setI), avoidR: sum(setR), overlap: [...setI].filter((r) => setR.has(r)).length };
}

// ------------------------------------------------------------ views
function header(q, sub) {
  return `<div class="titlebar">${ICON}<h1>Post-discharge follow-up</h1></div><div class="dots"></div>`;
}
function steps(L) {
  const s = (k, v, n, cls, href) => `<a class="step ${cls}" href="${href}"><div><div class="s-k">${k}</div><div class="s-v">${v}</div><div class="s-n">${n}</div></div><span class="ring"></span></a>`;
  return `<nav class="steps">${s('Step 1', 'Discharges', `${L.rows.length} in last 48 h`, 'done', '#worklist')}${s('Step 2', 'Rank by impact', 'calls that change outcomes', state.view === 'worklist' ? 'done active' : 'done', '#worklist')}${s('Step 3', 'Call list', `${L.cap} calls today`, 'done', '#worklist')}${s('Step 4', 'Validate', cv ? 'Jan–Sep 2026 results' : 'running…', state.view === 'impact' || state.view === 'calibration' ? 'active' : '', '#impact')}</nav>`;
}
function worklist() {
  const L = listFor();
  const caps = [Math.max(1, Math.round(L.rows.length * 0.1)), Math.round(L.rows.length * 0.2), Math.round(L.rows.length * 0.3)];
  const capTiles = [...new Set(caps)].map((c) => `<button class="tile ${c === L.cap ? 'on' : ''}" data-cap="${c}">${c} calls<small>${Math.round((100 * c) / L.rows.length)}% of discharges</small></button>`).join('');
  const maxU = Math.max(...today.map((r) => r.hi), 0.2);
  const body = L.shown.map((r, i) => {
    const inI = L.setI.has(r), inR = L.setR.has(r);
    const called = state.rank === 'impact' ? inI : inR;
    const tag = inI && inR ? '<span class="tag both">Both</span>' : inI ? '<span class="tag impact">Impact only</span>' : inR ? '<span class="tag risk">Risk only</span>' : '';
    const lo = Math.max(0, r.lo), hi = Math.max(0, r.hi);
    const chips = r.why.map((w) => `<span class="chip ${w.c < 0 ? 'neg' : ''}">${w.c >= 0 ? '+' : '−'} ${w.label}</span>`).join('');
    return `<tr class="${called ? 'call' : ''} ${i === L.cap - 1 ? 'cut' : ''}">
      <td><span class="check ${called ? 'on' : ''}"></span></td><td class="rank">${i + 1}</td>
      <td class="pt"><b>${r.p.id}</b><span>${r.p.age} ${r.p.female ? 'F' : 'M'} · ${I.MARKETS[r.p.market].name}</span></td>
      <td class="pt"><b>${r.p.dx}</b><span>${r.p.dispo} · ${r.p.hoursSince} h ago · ${r.p.meds} meds</span></td>
      <td class="num">${pct(r.risk)}</td>
      <td><div style="display:flex;align-items:center;gap:10px"><div class="ubar" title="90% interval ${pts(r.lo)} to ${pts(r.hi)} pts"><span class="f" style="width:${(Math.max(0, r.uplift) / maxU) * 100}%"></span><span class="ci" style="left:${(lo / maxU) * 100}%;width:${((hi - lo) / maxU) * 100}%"></span></div><span class="uval">${r.uplift >= 0 ? '' : '−'}${pts(Math.abs(r.uplift))}</span></div></td>
      <td><div class="chips">${chips}</div></td><td>${tag}</td></tr>`;
  }).join('');
  const moved = [...L.setI].filter((r) => !L.setR.has(r)).length;
  return `${header()}<div class="layout">${steps(L)}<section>
    <h2 class="q">Who should we call in the next 48 hours?</h2>
    <p class="sub">${L.rows.length} Medicare patients discharged ${state.market === 'all' ? 'across partner hospitals' : `at ${I.MARKETS[state.market].partner}`}. Rank by the readmissions a call can prevent, not by who is sickest.</p>
    <div class="tiles"><span class="label">Rank by</span><button class="tile ${state.rank === 'impact' ? 'on' : ''}" data-rank="impact">Readmissions avoided<small>impact of a call</small></button><button class="tile ${state.rank === 'risk' ? 'on' : ''}" data-rank="risk">Readmission risk<small>current practice</small></button>
      <span class="label" style="margin-left:18px">Capacity</span>${capTiles}</div>
    <div class="sumrow">
      <div class="card lav"><div class="pad"><h3>Today's ${L.cap} calls</h3><p class="muted">Expected readmissions avoided</p>
        <div class="bigrow"><div class="big teal">${fmt(L.avoidI, 2)}</div><p>ranked by impact<br>vs <b style="color:var(--purple)">${fmt(L.avoidR, 2)}</b> ranked by risk</p></div>
        <p class="fine">Model estimate: the sum of each called patient's predicted risk reduction.</p></div></div>
      <div class="card"><div class="pad"><div class="kv" style="margin-top:0"><span>Same patients on both lists</span><span class="v">${L.overlap} of ${L.cap}</span><span>Added by impact ranking</span><span class="v t">${moved}</span><span>Expected gain from re-ranking</span><span class="v p">${L.avoidR > 0 ? '+' + Math.round((L.avoidI / L.avoidR - 1) * 100) + '%' : '—'}</span></div>
        ${cmp ? `<p class="fine">Out of sample, Jan–Sep 2026, ${Math.round(state.frac * 100)}% capacity: <b style="color:var(--teal-ink)">${fmt(cmp.avoidedImpact, 1)}</b> vs <b style="color:var(--purple)">${fmt(cmp.avoidedRisk, 1)}</b> readmissions avoided per 1,000. <a href="#impact">Validation</a></p>` : `<p class="loading">Validating on ${fmt(world.hist.length)} discharges… fold ${cvRun.done} of 5</p>`}</div></div>
      <div class="card"><div class="pad"><h3 style="font-size:17px">Why risk ranking misses</h3><p style="font-size:14.5px">The sickest patients (high HCC, frequent admissions, SNF discharges) often readmit whether or not someone calls. A call moves outcomes most for patients going home alone, on many medications, without a recent PCP visit.</p>
        <button class="btn purple" style="margin-top:12px;width:100%;font-size:15px;padding:9px" id="export2">Send ${L.cap} calls to care team</button></div></div>
    </div>
    <div class="card tablecard" style="margin-top:20px"><table class="wl"><thead><tr><th></th><th>#</th><th>Patient</th><th>Discharge</th><th class="num">30-day risk</th><th>Readmission risk avoided by a call</th><th>Why</th><th>List</th></tr></thead><tbody>${body}</tbody></table></div>
    </div></section></div>`;
}

function curveSVG() {
  const W = 760, H = 330, L = 56, R = 20, T = 20, B = 46;
  const xs = curveData.map((d) => d.frac), ymax = Math.ceil(Math.max(...curveData.map((d) => Math.max(d.impact, d.risk))) / 10) * 10;
  const X = (f) => L + (f / 1) * (W - L - R), Y = (v) => T + (1 - Math.max(0, v) / ymax) * (H - T - B);
  let g = '';
  for (let v = 0; v <= ymax; v += 10) g += `<line class="gl" x1="${L}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="tk" x="${L - 10}" y="${Y(v) + 4}" text-anchor="end">${v}</text>`;
  [0, 0.2, 0.4, 0.6, 0.8, 1].forEach((f) => { g += `<text class="tk" x="${X(f)}" y="${H - B + 22}" text-anchor="middle">${Math.round(f * 100)}%</text>`; });
  g += `<text class="tk" x="${(L + W - R) / 2}" y="${H - 6}" text-anchor="middle">Share of discharges the team can call</text><text class="tk" transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">Readmissions avoided per 1,000</text>`;
  const path = (k) => 'M' + [[0, 0], ...curveData.map((d) => [d.frac, d[k]])].map(([f, v]) => `${X(f)},${Y(v)}`).join('L');
  const m = curveData.find((d) => Math.abs(d.frac - state.frac) < 1e-9) || curveData[3];
  const mark = `<line x1="${X(m.frac)}" x2="${X(m.frac)}" y1="${T}" y2="${H - B}" stroke="#9eabb5" stroke-dasharray="3 4"/><circle cx="${X(m.frac)}" cy="${Y(m.impact)}" r="6" fill="#37a38f" stroke="#fff" stroke-width="2"/><circle cx="${X(m.frac)}" cy="${Y(m.risk)}" r="6" fill="#6a4c91" stroke="#fff" stroke-width="2"/>
    <text class="lbl-i" x="${X(m.frac) + 10}" y="${Y(m.impact) - 8}">${fmt(m.impact, 1)}</text><text class="lbl-r" x="${X(m.frac) + 10}" y="${Y(m.risk) + 18}">${fmt(m.risk, 1)}</text>`;
  const tpath = (k) => 'M' + [[0, 0], ...curveData.map((d) => [d.frac, d['true_' + k]])].map(([f, v]) => `${X(f)},${Y(v)}`).join('L');
  return `<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-label="Capacity curve">${g}<path d="${tpath('risk')}" fill="none" stroke="#6a4c91" stroke-opacity=".35" stroke-width="1.5"/><path d="${tpath('impact')}" fill="none" stroke="#37a38f" stroke-opacity=".45" stroke-width="1.5"/><path class="ln-r" d="${path('risk')}"/><path class="ln-i" d="${path('impact')}"/>${mark}</svg>`;
}
function impact() {
  const L = listFor();
  if (!cmp) return `${header()}<div class="layout">${steps(L)}<section><h2 class="q">Does ranking by impact avoid more readmissions?</h2><p class="loading">Cross-fitting on ${fmt(world.hist.length)} discharges (Jan–Sep 2026)… fold ${cvRun.done} of 5</p><div class="progress"><span style="width:${(cvRun.done / 5) * 100}%"></span></div></section></div>`;
  const c = cmp, o = oracleTruth;
  return `${header()}<div class="layout">${steps(L)}<section>
    <h2 class="q">Does ranking by impact avoid more readmissions?</h2>
    <p class="sub">${fmt(world.hist.length)} discharges, Jan–Sep 2026. Each month block is scored by models trained on the other months; outcomes under each call list are estimated with a doubly robust (AIPW) estimator that corrects for who care managers chose to call.</p>
    <div class="slider">Capacity <input id="frac" type="range" min="0.05" max="0.6" step="0.05" value="${state.frac}"><span>${Math.round(state.frac * 100)}% of discharges called</span></div>
    <div class="stats3">
      <div class="card stat"><div class="k">Impact ranking</div><div class="v t">${fmt(c.avoidedImpact, 1)}</div><div class="s">avoided per 1,000 · 90% CI ${fmt(c.ci.impact[0], 1)} to ${fmt(c.ci.impact[1], 1)}</div></div>
      <div class="card stat"><div class="k">Risk ranking</div><div class="v p">${fmt(c.avoidedRisk, 1)}</div><div class="s">avoided per 1,000 · 90% CI ${fmt(c.ci.risk[0], 1)} to ${fmt(c.ci.risk[1], 1)}</div></div>
      <div class="card stat"><div class="k">Extra from re-ranking</div><div class="v">${c.extra >= 0 ? '+' : ''}${fmt(c.extra, 1)}</div><div class="s">per 1,000, same calls · 90% CI ${fmt(c.ci.extra[0], 1)} to ${fmt(c.ci.extra[1], 1)}</div></div>
      <div class="card stat"><div class="k">Current practice</div><div class="v" style="color:var(--slate)">${fmt(1000 * (c.rate.none - c.rate.current), 1)}</div><div class="s">avoided per 1,000 while calling ${pct(calledShare)} of discharges</div></div>
    </div>
    <div class="card" style="margin-top:22px"><div class="pad"><h3>Readmissions avoided by call capacity</h3><div class="leg"><span><i></i>Rank by impact</span><span><i class="r"></i>Rank by risk</span><span><i class="thin"></i>True values (synthetic)</span></div>${curveSVG()}
    <p class="fine">Per 1,000 discharges, relative to calling no one. Thick lines are AIPW estimates from observed outcomes; thin lines are the true values. Above about 40% capacity the two lists converge and the estimates are within noise of each other; at 100% both call everyone.</p></div></div>
    <div class="card lav" style="margin-top:22px"><div class="pad"><h3>Check against the truth</h3><p>Because this data is synthetic, the true effect of every call is known. At ${Math.round(state.frac * 100)}% capacity the true numbers are <b style="color:var(--teal-ink)">${fmt(o.impact, 1)}</b> avoided per 1,000 with impact ranking and <b style="color:var(--purple)">${fmt(o.risk, 1)}</b> with risk ranking (<b>${o.impact >= o.risk ? '+' : ''}${Math.round((o.impact / o.risk - 1) * 100)}%</b>)${(o.impact - o.risk) / 1 >= c.ci.extra[0] && (o.impact - o.risk) <= c.ci.extra[1] ? ', and the true difference falls inside the estimator\'s interval' : ''}. Current practice, which called ${pct(calledShare)} of discharges, truly avoided ${fmt(o.current, 1)}: impact ranking ${o.impact >= o.current - 0.5 ? 'matches it' : 'reaches ' + Math.round((100 * o.impact) / o.current) + '% of it'} with ${Math.round(state.frac * 100)}% of discharges called.</p></div></div>
  </section></div>`;
}
function calibration() {
  const L = listFor();
  if (!calib) return `${header()}<div class="layout">${steps(L)}<section><h2 class="q">Are the impact estimates right?</h2><p class="loading">Cross-fitting… fold ${cvRun.done} of 5</p><div class="progress"><span style="width:${(cvRun.done / 5) * 100}%"></span></div></section></div>`;
  const W = 760, H = 320, Lm = 56, R = 20, T = 20, B = 50;
  const vmax = Math.ceil(Math.max(...calib.map((g) => Math.max(g.hi, g.pred, g.truth))) * 100 / 2) * 2 / 100;
  const vmin = Math.min(0, Math.floor(Math.min(...calib.map((g) => Math.min(g.lo, g.pred))) * 100 / 2) * 2 / 100);
  const Y = (v) => T + (1 - (v - vmin) / (vmax - vmin)) * (H - T - B);
  const bw = (W - Lm - R) / calib.length;
  let g = '';
  for (let v = vmin; v <= vmax + 1e-9; v += 0.02) g += `<line class="gl" x1="${Lm}" x2="${W - R}" y1="${Y(v)}" y2="${Y(v)}"/><text class="tk" x="${Lm - 10}" y="${Y(v) + 4}" text-anchor="end">${Math.round(v * 100)}</text>`;
  calib.forEach((c, i) => {
    const x0 = Lm + i * bw + bw * 0.18, w = bw * 0.28;
    g += `<rect x="${x0}" y="${Math.min(Y(c.pred), Y(0))}" width="${w}" height="${Math.abs(Y(c.pred) - Y(0))}" fill="#9eabb5"/>`;
    const xo = x0 + w + bw * 0.1 + w / 2;
    g += `<line x1="${xo}" x2="${xo}" y1="${Y(c.hi)}" y2="${Y(c.lo)}" stroke="#37a38f" stroke-width="2.5"/><circle cx="${xo}" cy="${Y(c.obs)}" r="6" fill="#37a38f" stroke="#fff" stroke-width="2"/>`;
    g += `<line x1="${x0 - 4}" x2="${xo + w / 2 + 4}" y1="${Y(c.truth)}" y2="${Y(c.truth)}" stroke="#6a4c91" stroke-width="2.5" stroke-dasharray="5 3"/>`;
    g += `<text class="tk" x="${Lm + i * bw + bw / 2}" y="${H - B + 20}" text-anchor="middle">${['Lowest', '2nd', '3rd', '4th', 'Highest'][i]} fifth</text><text class="tk" x="${Lm + i * bw + bw / 2}" y="${H - B + 36}" text-anchor="middle">risk ${pct(c.risk)}</text>`;
  });
  return `${header()}<div class="layout">${steps(L)}<section>
    <h2 class="q">Are the impact estimates right?</h2>
    <p class="sub">Patients grouped into fifths by predicted impact. Grey: predicted risk reduction. Teal: the reduction estimated from what actually happened (AIPW, 90% interval). Dashed purple: the true reduction, known only because the data is synthetic.</p>
    <div class="card" style="margin-top:22px"><div class="pad"><div class="leg"><span><i class="t"></i>Predicted</span><span><i class="o"></i>Observed (AIPW)</span><span><i class="r"></i>True (synthetic)</span></div>
    <svg class="chart" viewBox="0 0 ${W} ${H}">${g}<text class="tk" transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">Readmission risk avoided (pts)</text></svg>
    <p class="fine">Note the risk row under each fifth: average 30-day risk barely moves across fifths, while the benefit of a call rises about ${Math.round(calib[4].truth / Math.max(calib[0].truth, 0.001))}× from the lowest to the highest fifth. Risk and impact are different rankings.</p></div></div>
  </section></div>`;
}
function method() {
  const L = listFor();
  return `${header()}<div class="layout">${steps(L)}<section class="prose">
    <h2 class="q">How impact ranking works</h2>
    <h3>1. Learn from past calls, corrected for who got called</h3>
    <p>Care managers did not call at random: in the history they called ${pct(world.hist.reduce((s, p) => s + p.t, 0) / world.hist.length)} of discharges, favoring sicker patients with a PCP on record. A propensity model estimates each patient's chance of being called, and two outcome models estimate 30-day readmission with and without a call. All three are cross-fitted (each patient is predicted by models that did not see them).</p>
    <h3>2. Estimate the benefit of a call for each patient</h3>
    <p>The doubly robust learner turns each past discharge into a pseudo-outcome, <code>m0 − m1 + (1−T)(Y−m0)/(1−e) − T(Y−m1)/e</code>, whose average is the true benefit even if one of the models is wrong. A ridge regression on patient features (with discharge-home interactions) predicts that benefit for new patients, with a sandwich-variance 90% interval shown as the whisker on each row.</p>
    <h3>3. Fill the day's capacity</h3>
    <p>With capacity for k calls, call the k patients with the largest predicted benefit. "Why" chips show the features that move a patient's benefit most relative to the average patient.</p>
    <h3>4. Validate before trusting it</h3>
    <p>Each block of months is scored by models trained on the other months. The readmission rate under any call list is estimated with AIPW and a 200-sample bootstrap. Because this demo is synthetic, the page also shows the true numbers so you can see the estimator is honest.</p>
    <h3>Limits</h3>
    <p>Needs past variation in who was called and overlap (every kind of patient sometimes called and sometimes not). Unmeasured reasons for calling (for example a nurse's judgment) would bias the history. In production this should start with a randomized slice of the worklist to confirm the effect.</p>
  </section></div>`;
}

// ------------------------------------------------------------ render + events
function render() {
  document.querySelectorAll('#nav a').forEach((a) => a.classList.toggle('active', a.dataset.view === state.view));
  $('#app').innerHTML = { worklist, impact, calibration, method }[state.view]();
  document.querySelectorAll('[data-rank]').forEach((b) => b.addEventListener('click', () => { state.rank = b.dataset.rank; render(); }));
  document.querySelectorAll('[data-cap]').forEach((b) => b.addEventListener('click', () => { state.cap = Number(b.dataset.cap); render(); }));
  const f = $('#frac'); if (f) f.addEventListener('input', () => { state.frac = Number(f.value); recompute(); render(); });
  ['#export2'].forEach((id) => { const b = $(id); if (b) b.addEventListener('click', exportCsv); });
}
function exportCsv() {
  const L = listFor();
  const lines = ['rank,patient,age,sex,dx,disposition,risk,avoided_pts,ci_low,ci_high'];
  L.shown.slice(0, L.cap).forEach((r, i) => lines.push([i + 1, r.p.id, r.p.age, r.p.female ? 'F' : 'M', r.p.dx, r.p.dispo, r.risk.toFixed(3), (r.uplift * 100).toFixed(2), (r.lo * 100).toFixed(2), (r.hi * 100).toFixed(2)].join(',')));
  const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([lines.join('\n')], { type: 'text/csv' })); a.download = 'pulse_call_list.csv'; a.click();
}
$('#export').addEventListener('click', exportCsv);
$('#market').addEventListener('change', (e) => { state.market = e.target.value; state.cap = null; render(); });
function fromHash() { const v = location.hash.replace('#', ''); state.view = ['worklist', 'impact', 'calibration', 'method'].includes(v) ? v : 'worklist'; }
window.addEventListener('hashchange', () => { fromHash(); render(); window.scrollTo(0, 0); });

function recompute() {
  if (!cv) return;
  cmp = I.compareCV(cv, state.frac, 200);
  const hist = cv.rows.map((r) => r.p);
  const n0 = I.oracle(hist, new Array(hist.length).fill(0));
  oracleTruth = { impact: 1000 * (n0 - I.oracle(hist, cmp.callI)), risk: 1000 * (n0 - I.oracle(hist, cmp.callR)), current: 1000 * (n0 - I.oracle(hist, hist.map((p) => p.t))) };
}
function stepCV() {
  cvRun.step();
  if (cvRun.done < cvRun.K) { render(); setTimeout(stepCV, 0); return; }
  cv = cvRun.result();
  recompute();
  curveData = I.curveCV(cv, [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1.0]);
  { const hist = cv.rows.map((r) => r.p), n0 = I.oracle(hist, new Array(hist.length).fill(0));
    curveData.forEach((d) => { const c = I.compareCV(cv, d.frac, 1); d.true_impact = 1000 * (n0 - I.oracle(hist, c.callI)); d.true_risk = 1000 * (n0 - I.oracle(hist, c.callR)); }); }
  calib = I.calibrationCV(cv).map((g, i, arr) => g);
  // true mean benefit per fifth (synthetic world only)
  const order = cv.rows.map((r, i) => [r.uplift, i]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  calib.forEach((g, k) => { const ids = order.slice(Math.floor((k * order.length) / 5), Math.floor(((k + 1) * order.length) / 5)); g.truth = ids.reduce((s, i) => s + I.trueUplift(cv.rows[i].p), 0) / ids.length; });
  render();
}
fromHash();
render();
setTimeout(stepCV, 50);
