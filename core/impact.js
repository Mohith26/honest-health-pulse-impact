// PULSE impact ranking: rank post-discharge follow-up calls by expected
// readmissions avoided (uplift), not by readmission risk.
// Doubly robust learner (Kennedy 2023) with cross-fitted nuisance models, AIPW
// policy evaluation under a call-capacity limit, bootstrap intervals, and a
// sandwich interval for each patient's estimated impact.
// Pure ES module, shared by the page and `node --test`.

// ------------------------------------------------------------ rng
export function mulberry32(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
export function normal(r) { let u = 0; while (u === 0) u = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r()); }
function poisson(r, lam) { const L = Math.exp(-lam); let k = 0, p = 1; do { k++; p *= r(); } while (p > L); return k - 1; }
export const sigmoid = (z) => 1 / (1 + Math.exp(-z));
const pick = (r, items, w) => { let u = r() * w.reduce((a, b) => a + b, 0); for (let i = 0; i < items.length; i++) { u -= w[i]; if (u <= 0) return items[i]; } return items[items.length - 1]; };

// ------------------------------------------------------------ world
export const DX = ['CHF', 'COPD', 'Pneumonia', 'Sepsis', 'AMI', 'Hip/knee', 'Other medical'];
export const DISPO = ['Home', 'Home health', 'SNF'];
export const MARKETS = [
  { id: 'west', name: 'West', partner: 'Cascade Valley Health', share: 0.34 },
  { id: 'midwest', name: 'Midwest', partner: 'Lakeshore Health Partners', share: 0.38 },
  { id: 'east', name: 'East', partner: 'Mohawk Regional Health', share: 0.28 },
];
export const CONFIG = { seed: 20261110, histN: 30000, trainFrac: 0.6, todayN: 64, capacityToday: 13, capacityFrac: 0.2 };

const DX_RISK = { CHF: 0.55, COPD: 0.45, Pneumonia: 0.3, Sepsis: 0.6, AMI: 0.25, 'Hip/knee': -0.85, 'Other medical': 0 };
export function drawPatient(r) {
  const market = pick(r, [0, 1, 2], MARKETS.map((m) => m.share));
  const age = Math.min(97, Math.max(65, Math.round(77 + 7.5 * normal(r))));
  const female = r() < 0.55 ? 1 : 0;
  const dx = pick(r, DX, [0.16, 0.12, 0.13, 0.1, 0.07, 0.12, 0.3]);
  const hcc = Math.min(6, Math.exp(0.15 + 0.55 * normal(r)) + (dx === 'CHF' || dx === 'Sepsis' ? 0.4 : 0));
  const prior = poisson(r, 0.35 + 0.35 * Math.max(0, hcc - 1));
  const ed = poisson(r, 0.5 + 0.2 * prior);
  const meds = Math.max(1, Math.round(7 + 2.2 * hcc + 2.5 * normal(r) + (dx === 'CHF' ? 2 : 0)));
  const alone = r() < (age > 82 ? 0.42 : 0.28) ? 1 : 0;
  const pcp = r() < 0.72 - 0.08 * (prior > 1) ? 1 : 0; // PCP visit in last 6 months
  const dispo = pick(r, DISPO, dx === 'Hip/knee' ? [0.45, 0.35, 0.2] : age > 85 ? [0.45, 0.3, 0.25] : [0.62, 0.24, 0.14]);
  const los = Math.max(1, Math.round((dx === 'Sepsis' ? 6 : 3.8) + 1.4 * normal(r) + 0.6 * hcc));
  const dual = r() < 0.18 ? 1 : 0;
  return { market, age, female, dx, hcc: +hcc.toFixed(2), prior, ed, meds, alone, pcp, dispo, los, dual };
}
// True baseline readmission log-odds (no call) and the true effect of a
// 48-hour follow-up call (log-odds, <= 0). The effect is largest for patients
// who are at home, alone, on many medications, without a recent PCP visit,
// and with CHF/COPD; it fades for SNF discharges, elective joints, and the
// very sickest (who readmit regardless).
export function trueBaseLogit(p) {
  return -2.7 + 0.6 * (p.hcc - 1.4) + DX_RISK[p.dx] + 0.36 * p.prior + 0.14 * p.ed + 0.035 * (p.meds - 9) + 0.2 * p.dual + (p.dispo === 'SNF' ? 0.3 : 0) + 0.012 * (p.age - 77);
}
export function trueEffectLogit(p) {
  let e = 0.2 + 0.6 * p.alone + (p.meds >= 12 ? 0.55 : p.meds >= 9 ? 0.25 : 0) + 0.65 * (1 - p.pcp) + (p.dx === 'CHF' || p.dx === 'COPD' ? 0.45 : 0);
  if (p.dx === 'Hip/knee') e *= 0.15;
  if (p.dx === 'Sepsis') e *= 0.45;
  e *= p.dispo === 'Home' ? 1 : p.dispo === 'Home health' ? 0.5 : 0.1;
  if (p.hcc > 3) e *= 0.15; else if (p.hcc > 2.4) e *= 0.45;
  if (p.prior >= 3) e *= 0.35; else if (p.prior === 2) e *= 0.75;
  return -e;
}
export const trueRisk = (p, t) => sigmoid(trueBaseLogit(p) + (t ? trueEffectLogit(p) : 0));
export const trueUplift = (p) => trueRisk(p, 0) - trueRisk(p, 1); // readmission probability avoided by a call
// Historical practice: care managers called sicker, more reachable patients.
export function histPropensity(p) {
  return sigmoid(-0.9 + 0.55 * (p.hcc - 1.4) + 0.35 * Math.min(p.prior, 3) + 0.55 * p.pcp + (p.dispo === 'SNF' ? -0.6 : 0) + [0.15, -0.1, 0][p.market]);
}

export function buildWorld(opts = {}) {
  const seed = opts.seed ?? CONFIG.seed, N = opts.histN ?? CONFIG.histN;
  const r = mulberry32(seed);
  const hist = [];
  const start = Date.UTC(2026, 0, 1), span = 273 * 864e5; // Jan 1 - Sep 30, 2026
  for (let i = 0; i < N; i++) {
    const p = drawPatient(r);
    p.t = r() < histPropensity(p) ? 1 : 0;
    p.y = r() < trueRisk(p, p.t) ? 1 : 0;
    p.date = start + Math.floor(r() * span);
    hist.push(p);
  }
  hist.sort((a, b) => a.date - b.date);
  hist.forEach((p, i) => { p.id = i; });
  const rT = mulberry32(seed + 7);
  const today = [];
  for (let i = 0; i < (opts.todayN ?? CONFIG.todayN); i++) {
    const p = drawPatient(rT);
    p.id = `P-${String(40211 + i * 37).padStart(6, '0')}`;
    p.hoursSince = Math.floor(rT() * 46) + 1;
    today.push(p);
  }
  return { seed, hist, today };
}

// ------------------------------------------------------------ features
export const FEATURES = ['age', 'female', 'hcc', 'prior', 'ed', 'meds', 'meds10', 'alone', 'noPcp', 'dual', 'los',
  'dx_CHF', 'dx_COPD', 'dx_Pneumonia', 'dx_Sepsis', 'dx_AMI', 'dx_Hip/knee', 'dispo_Home health', 'dispo_SNF', 'hccHigh', 'mkt_midwest', 'mkt_east'];
export const FEATURE_LABEL = {
  age: 'Age', female: 'Female', hcc: 'HCC risk score', prior: 'Admissions, last 12 mo', ed: 'ED visits, last 6 mo', meds: 'Discharge medications', meds10: '10+ discharge medications',
  alone: 'Lives alone', noPcp: 'No PCP visit in 6 mo', dual: 'Dual eligible', los: 'Length of stay', 'dx_CHF': 'Heart failure', 'dx_COPD': 'COPD', 'dx_Pneumonia': 'Pneumonia', 'dx_Sepsis': 'Sepsis',
  'dx_AMI': 'AMI', 'dx_Hip/knee': 'Elective hip/knee', 'dispo_Home health': 'Discharged with home health', 'dispo_SNF': 'Discharged to SNF', hccHigh: 'HCC above 2.4', 'mkt_midwest': 'Midwest market', 'mkt_east': 'East market',
};
export function featurize(p) {
  return [
    (p.age - 77) / 8, p.female, p.hcc - 1.4, Math.min(p.prior, 5), Math.min(p.ed, 6), (p.meds - 9) / 3, p.meds >= 10 ? 1 : 0,
    p.alone, 1 - p.pcp, p.dual, (p.los - 4) / 2,
    p.dx === 'CHF' ? 1 : 0, p.dx === 'COPD' ? 1 : 0, p.dx === 'Pneumonia' ? 1 : 0, p.dx === 'Sepsis' ? 1 : 0, p.dx === 'AMI' ? 1 : 0, p.dx === 'Hip/knee' ? 1 : 0,
    p.dispo === 'Home health' ? 1 : 0, p.dispo === 'SNF' ? 1 : 0, p.hcc > 2.4 ? 1 : 0, p.market === 1 ? 1 : 0, p.market === 2 ? 1 : 0,
  ];
}

// ------------------------------------------------------------ linear algebra
function cholSolve(A, b) {
  const n = A.length, L = A.map((row) => row.slice());
  for (let j = 0; j < n; j++) {
    let s = L[j][j]; for (let k = 0; k < j; k++) s -= L[j][k] * L[j][k];
    L[j][j] = Math.sqrt(Math.max(s, 1e-12));
    for (let i = j + 1; i < n; i++) { let t = L[i][j]; for (let k = 0; k < j; k++) t -= L[i][k] * L[j][k]; L[i][j] = t / L[j][j]; }
  }
  const y = new Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i][k] * y[k]; y[i] = s / L[i][i]; }
  const x = new Array(n); for (let i = n - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k]; x[i] = s / L[i][i]; }
  return x;
}
export function inverse(A) { const n = A.length; return Array.from({ length: n }, (_, j) => cholSolve(A, Array.from({ length: n }, (_, i) => (i === j ? 1 : 0)))).map((col, j, cols) => cols.map((c) => c[j])); }
const addIntercept = (x) => [1, ...x];

// L2-regularized logistic regression by Newton/IRLS. Returns coefficients
// (intercept first).
export function fitLogistic(X, y, lambda = 1, iters = 25) {
  const n = X.length, d = X[0].length + 1;
  let b = new Array(d).fill(0);
  for (let it = 0; it < iters; it++) {
    const H = Array.from({ length: d }, () => new Array(d).fill(0)); const g = new Array(d).fill(0);
    for (let i = 0; i < n; i++) {
      const xi = addIntercept(X[i]); let z = 0; for (let j = 0; j < d; j++) z += b[j] * xi[j];
      const p = sigmoid(z), w = p * (1 - p);
      for (let j = 0; j < d; j++) { g[j] += (y[i] - p) * xi[j]; const wx = w * xi[j]; for (let k = j; k < d; k++) H[j][k] += wx * xi[k]; }
    }
    for (let j = 0; j < d; j++) { for (let k = 0; k < j; k++) H[j][k] = H[k][j]; if (j > 0) { H[j][j] += lambda; g[j] -= lambda * b[j]; } else H[j][j] += 1e-8; }
    const step = cholSolve(H, g);
    let mx = 0; for (let j = 0; j < d; j++) { b[j] += step[j]; mx = Math.max(mx, Math.abs(step[j])); }
    if (mx < 1e-7) break;
  }
  return b;
}
export const predictLogistic = (b, x) => { let z = b[0]; for (let j = 0; j < x.length; j++) z += b[j + 1] * x[j]; return sigmoid(z); };

// Ridge regression with a sandwich (HC0) covariance, for the CATE stage.
export function fitRidge(X, y, lambda = 5) {
  const n = X.length, d = X[0].length + 1;
  const A = Array.from({ length: d }, () => new Array(d).fill(0)), c = new Array(d).fill(0);
  const Xa = X.map(addIntercept);
  for (let i = 0; i < n; i++) { const xi = Xa[i]; for (let j = 0; j < d; j++) { c[j] += xi[j] * y[i]; for (let k = j; k < d; k++) A[j][k] += xi[j] * xi[k]; } }
  for (let j = 0; j < d; j++) { for (let k = 0; k < j; k++) A[j][k] = A[k][j]; if (j > 0) A[j][j] += lambda; }
  const b = cholSolve(A, c);
  const Ainv = inverse(A);
  const M = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < n; i++) { const xi = Xa[i]; let r = y[i]; for (let j = 0; j < d; j++) r -= b[j] * xi[j]; const r2 = r * r; for (let j = 0; j < d; j++) for (let k = j; k < d; k++) M[j][k] += r2 * xi[j] * xi[k]; }
  for (let j = 0; j < d; j++) for (let k = 0; k < j; k++) M[j][k] = M[k][j];
  const tmp = Ainv.map((row) => M[0].map((_, k) => row.reduce((s, v, j) => s + v * M[j][k], 0)));
  const V = tmp.map((row) => Ainv[0].map((_, k) => row.reduce((s, v, j) => s + v * Ainv[j][k], 0)));
  return { b, V };
}
export function ridgePredict(m, x) { const xa = addIntercept(x); let s = 0; for (let j = 0; j < xa.length; j++) s += m.b[j] * xa[j]; return s; }
export function ridgeSE(m, x) { const xa = addIntercept(x); let s = 0; for (let j = 0; j < xa.length; j++) for (let k = 0; k < xa.length; k++) s += xa[j] * m.V[j][k] * xa[k]; return Math.sqrt(Math.max(s, 0)); }

// ------------------------------------------------------------ DR-learner
const PCLIP = [0.05, 0.95];
// Cross-fitted nuisances on the training months, then the CATE stage. Returns
// the uplift model (risk avoided by a call, on the probability scale) plus
// out-of-fold nuisance predictions.
export function fitImpact(train, opts = {}) {
  const K = opts.folds ?? 2;
  const X = train.map(featurize);
  const n = train.length;
  const fold = train.map((_, i) => i % K);
  const e = new Array(n), m0 = new Array(n), m1 = new Array(n);
  for (let k = 0; k < K; k++) {
    const tr = [], te = [];
    for (let i = 0; i < n; i++) (fold[i] === k ? te : tr).push(i);
    const bE = fitLogistic(tr.map((i) => X[i]), tr.map((i) => train[i].t), 1);
    const c0 = tr.filter((i) => !train[i].t), c1 = tr.filter((i) => train[i].t);
    const b0 = fitLogistic(c0.map((i) => X[i]), c0.map((i) => train[i].y), 1);
    const b1 = fitLogistic(c1.map((i) => X[i]), c1.map((i) => train[i].y), 1);
    for (const i of te) { e[i] = Math.min(PCLIP[1], Math.max(PCLIP[0], predictLogistic(bE, X[i]))); m0[i] = predictLogistic(b0, X[i]); m1[i] = predictLogistic(b1, X[i]); }
  }
  // pseudo-outcome for uplift u = P(readmit | no call) - P(readmit | call)
  const phi = train.map((p, i) => (m0[i] - m1[i]) + ((1 - p.t) * (p.y - m0[i])) / (1 - e[i]) - (p.t * (p.y - m1[i])) / e[i]);
  const cate = fitRidge(train.map(cateFeaturize), phi, opts.ridge ?? 20);
  // full-data nuisance models used to score new patients and for risk ranking
  const bE = fitLogistic(X, train.map((p) => p.t), 1);
  const c0 = train.filter((p) => !p.t), c1 = train.filter((p) => p.t);
  const b0 = fitLogistic(c0.map(featurize), c0.map((p) => p.y), 1);
  const b1 = fitLogistic(c1.map(featurize), c1.map((p) => p.y), 1);
  return { cate, bE, b0, b1, phiMean: phi.reduce((a, b) => a + b, 0) / n };
}
export function score(model, p, z = 1.6449) {
  const x = featurize(p), xc = cateFeaturize(p);
  const u = ridgePredict(model.cate, xc), se = ridgeSE(model.cate, xc);
  return { uplift: u, lo: u - z * se, hi: u + z * se, se, risk: predictLogistic(model.b0, x), riskIfCalled: predictLogistic(model.b1, x) };
}
// Why this patient: largest contributions to estimated impact relative to the
// average patient (coefficient x deviation from the training mean).
export function reasons(model, p, means, top = 3) {
  const x = cateFeaturize(p);
  return CATE_FEATURES.map((f, j) => ({ f, label: FEATURE_LABEL[f], c: model.cate.b[j + 1] * (x[j] - means[j]) }))
    .filter((r) => !r.f.startsWith('mkt_')).sort((a, b) => Math.abs(b.c) - Math.abs(a.c)).slice(0, top);
}
// The uplift stage adds care-management interactions: what a call can change
// depends on whether the patient went home and on their social and medication risk.
export const CATE_FEATURES = [...FEATURES, 'home', 'home_alone', 'home_noPcp', 'home_meds10', 'home_chfCopd', 'home_hccHigh', 'prior3'];
Object.assign(FEATURE_LABEL, { home: 'Discharged home', home_alone: 'Home and lives alone', home_noPcp: 'Home, no recent PCP visit', home_meds10: 'Home with 10+ medications', home_chfCopd: 'Home with CHF or COPD', home_hccHigh: 'Home with HCC above 2.4', prior3: '3+ admissions last year' });
export function cateFeaturize(p) {
  const x = featurize(p), home = p.dispo === 'Home' ? 1 : 0;
  return [...x, home, home * p.alone, home * (1 - p.pcp), home * (p.meds >= 10 ? 1 : 0), home * (p.dx === 'CHF' || p.dx === 'COPD' ? 1 : 0), home * (p.hcc > 2.4 ? 1 : 0), p.prior >= 3 ? 1 : 0];
}
export function featureMeans(rows) { const X = rows.map(cateFeaturize); return X[0].map((_, j) => X.reduce((s, x) => s + x[j], 0) / X.length); }

// ------------------------------------------------------------ evaluation
// Doubly robust (AIPW) estimate of the readmission rate if a policy calls the
// patients flagged in `call`. Nuisances are fit on the training months only.
export function aipwTerms(model, test) {
  return test.map((p) => {
    const x = featurize(p);
    const e = Math.min(PCLIP[1], Math.max(PCLIP[0], predictLogistic(model.bE, x)));
    const m0 = predictLogistic(model.b0, x), m1 = predictLogistic(model.b1, x);
    return { y1: m1 + (p.t * (p.y - m1)) / e, y0: m0 + ((1 - p.t) * (p.y - m0)) / (1 - e) };
  });
}
export function policyValue(terms, call, idx = null) {
  const ids = idx ?? terms.map((_, i) => i);
  let s = 0; for (const i of ids) s += call[i] ? terms[i].y1 : terms[i].y0;
  return s / ids.length;
}
export function topK(scores, frac) {
  const n = scores.length, k = Math.round(frac * n);
  const order = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
  const call = new Array(n).fill(0); for (let j = 0; j < k; j++) call[order[j][1]] = 1; return call;
}
// Readmissions avoided per 1,000 discharges vs calling no one, for impact- and
// risk-ranked policies at a capacity fraction, with a paired bootstrap.
export function compare(model, test, frac, B = 200, seed = 99) {
  const terms = aipwTerms(model, test);
  const sc = test.map((p) => score(model, p));
  const callI = topK(sc.map((s) => s.uplift), frac), callR = topK(sc.map((s) => s.risk), frac);
  const none = new Array(test.length).fill(0);
  const v = (call, idx) => policyValue(terms, call, idx);
  const est = { none: v(none), impact: v(callI), risk: v(callR), current: test.reduce((s, p, i) => s + (p.t ? terms[i].y1 : terms[i].y0), 0) / test.length };
  const r = mulberry32(seed); const diffs = [], avI = [], avR = [];
  for (let b = 0; b < B; b++) {
    const idx = Array.from({ length: test.length }, () => Math.floor(r() * test.length));
    const n0 = v(none, idx), vi = v(callI, idx), vr = v(callR, idx);
    avI.push(1000 * (n0 - vi)); avR.push(1000 * (n0 - vr)); diffs.push(1000 * (vr - vi));
  }
  const q = (a, p) => { const s = [...a].sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.max(0, Math.floor(p * (s.length - 1))))]; };
  return {
    frac, calls: callI.reduce((a, b) => a + b, 0), n: test.length,
    avoidedImpact: 1000 * (est.none - est.impact), avoidedRisk: 1000 * (est.none - est.risk),
    extra: 1000 * (est.risk - est.impact),
    ci: { impact: [q(avI, 0.05), q(avI, 0.95)], risk: [q(avR, 0.05), q(avR, 0.95)], extra: [q(diffs, 0.05), q(diffs, 0.95)] },
    rate: est, callI, callR, scores: sc,
  };
}
// Oracle values from the known world (tests and README only, never the page's estimators).
export function oracle(test, call) { return test.reduce((s, p, i) => s + trueRisk(p, call[i]), 0) / test.length; }

// Calibration of the uplift model on the test months: patients grouped by
// predicted impact; observed effect per group from AIPW terms.
export function calibration(model, test, groups = 5) {
  const terms = aipwTerms(model, test);
  const sc = test.map((p) => score(model, p).uplift);
  const order = sc.map((s, i) => [s, i]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  const out = [];
  for (let g = 0; g < groups; g++) {
    const ids = order.slice(Math.floor((g * order.length) / groups), Math.floor(((g + 1) * order.length) / groups));
    const pred = ids.reduce((s, i) => s + sc[i], 0) / ids.length;
    const obsArr = ids.map((i) => terms[i].y0 - terms[i].y1);
    const obs = obsArr.reduce((a, b) => a + b, 0) / ids.length;
    const sd = Math.sqrt(obsArr.reduce((s, v) => s + (v - obs) ** 2, 0) / (ids.length - 1));
    out.push({ group: g + 1, n: ids.length, pred, obs, lo: obs - 1.6449 * sd / Math.sqrt(ids.length), hi: obs + 1.6449 * sd / Math.sqrt(ids.length), riskMean: ids.reduce((s, i) => s + score(model, test[i]).risk, 0) / ids.length });
  }
  return out;
}
// Capacity curve: avoided per 1,000 for both rankings at several capacities.
export function curve(model, test, fracs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]) {
  const terms = aipwTerms(model, test);
  const sc = test.map((p) => score(model, p));
  const none = policyValue(terms, new Array(test.length).fill(0));
  return fracs.map((f) => ({ frac: f, impact: 1000 * (none - policyValue(terms, topK(sc.map((s) => s.uplift), f))), risk: 1000 * (none - policyValue(terms, topK(sc.map((s) => s.risk), f))) }));
}

// Cross-fitted evaluation over all history: each patient is scored and
// evaluated by models fit on the other folds (blocked by month order), so every
// discharge counts once in the policy value without being in its own training set.
export function createCrossFit(world, K = 5) {
  const hist = world.hist, n = hist.length;
  const rows = new Array(n), models = [];
  let k = 0;
  return {
    get done() { return k; }, K,
    step() {
      if (k >= K) return k;
      const lo = Math.floor((k * n) / K), hi = Math.floor(((k + 1) * n) / K);
      const train = hist.slice(0, lo).concat(hist.slice(hi)), test = hist.slice(lo, hi);
      const m = fitImpact(train); models.push(m);
      const terms = aipwTerms(m, test);
      const kk = k;
      test.forEach((p, j) => { const sc = score(m, p); rows[lo + j] = { p, uplift: sc.uplift, risk: sc.risk, y1: terms[j].y1, y0: terms[j].y0, fold: kk }; });
      return ++k;
    },
    result() { return { rows, models, K }; },
  };
}
export function crossFit(world, K = 5) { const c = createCrossFit(world, K); while (c.done < K) c.step(); return c.result(); }
export function compareCV(cv, frac, B = 200, seed = 99) {
  const rows = cv.rows, n = rows.length;
  // capacity applies within each fold (each fold is a block of months)
  const callI = new Array(n).fill(0), callR = new Array(n).fill(0);
  for (let k = 0; k < cv.K; k++) {
    const ids = rows.map((r, i) => (r.fold === k ? i : -1)).filter((i) => i >= 0);
    const ti = topK(ids.map((i) => rows[i].uplift), frac), tr = topK(ids.map((i) => rows[i].risk), frac);
    ids.forEach((i, j) => { callI[i] = ti[j]; callR[i] = tr[j]; });
  }
  const terms = rows.map((r) => ({ y1: r.y1, y0: r.y0 }));
  const none = new Array(n).fill(0);
  const v = (call, idx) => policyValue(terms, call, idx);
  const est = { none: v(none), impact: v(callI), risk: v(callR), current: rows.reduce((s, r) => s + (r.p.t ? r.y1 : r.y0), 0) / n };
  const r = mulberry32(seed); const avI = [], avR = [], dif = [], rel = [];
  for (let b = 0; b < B; b++) {
    const idx = Array.from({ length: n }, () => Math.floor(r() * n));
    const n0 = v(none, idx), vi = v(callI, idx), vr = v(callR, idx);
    avI.push(1000 * (n0 - vi)); avR.push(1000 * (n0 - vr)); dif.push(1000 * (vr - vi)); rel.push((n0 - vi) / (n0 - vr) - 1);
  }
  const q = (a, p) => { const s2 = [...a].sort((x, y) => x - y); return s2[Math.min(s2.length - 1, Math.max(0, Math.floor(p * (s2.length - 1))))]; };
  return {
    frac, n, calls: callI.reduce((a, b) => a + b, 0),
    avoidedImpact: 1000 * (est.none - est.impact), avoidedRisk: 1000 * (est.none - est.risk), extra: 1000 * (est.risk - est.impact),
    relGain: (est.none - est.impact) / (est.none - est.risk) - 1,
    ci: { impact: [q(avI, 0.05), q(avI, 0.95)], risk: [q(avR, 0.05), q(avR, 0.95)], extra: [q(dif, 0.05), q(dif, 0.95)], rel: [q(rel, 0.05), q(rel, 0.95)] },
    rate: est, callI, callR,
  };
}
export function curveCV(cv, fracs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]) {
  return fracs.map((f) => { const c = compareCV(cv, f, 1); return { frac: f, impact: c.avoidedImpact, risk: c.avoidedRisk }; });
}
export function calibrationCV(cv, groups = 5) {
  const rows = cv.rows;
  const order = rows.map((r, i) => [r.uplift, i]).sort((a, b) => a[0] - b[0]).map((x) => x[1]);
  return Array.from({ length: groups }, (_, g) => {
    const ids = order.slice(Math.floor((g * order.length) / groups), Math.floor(((g + 1) * order.length) / groups));
    const pred = ids.reduce((s, i) => s + rows[i].uplift, 0) / ids.length;
    const o = ids.map((i) => rows[i].y0 - rows[i].y1);
    const obs = o.reduce((a, b) => a + b, 0) / ids.length;
    const sd = Math.sqrt(o.reduce((s, v) => s + (v - obs) ** 2, 0) / (ids.length - 1));
    const risk = ids.reduce((s, i) => s + rows[i].risk, 0) / ids.length;
    return { group: g + 1, n: ids.length, pred, obs, lo: obs - 1.6449 * sd / Math.sqrt(ids.length), hi: obs + 1.6449 * sd / Math.sqrt(ids.length), risk };
  });
}
export function splitWorld(world, frac = CONFIG.trainFrac) {
  const k = Math.floor(world.hist.length * frac);
  return { train: world.hist.slice(0, k), test: world.hist.slice(k) };
}
