import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as I from '../core/impact.js';

const W = I.buildWorld();
const cv = I.crossFit(W, 5);
const hist = cv.rows.map((r) => r.p);
const none = new Array(hist.length).fill(0);
const n0 = I.oracle(hist, none);

test('seeded world is deterministic and plausible', () => {
  const W2 = I.buildWorld();
  assert.deepEqual(W2.hist[123], W.hist[123]);
  assert.equal(W.hist.length, 30000);
  const rate = W.hist.reduce((s, p) => s + p.y, 0) / W.hist.length;
  assert.ok(rate > 0.09 && rate < 0.16, `readmission rate ${rate}`);
  const called = W.hist.reduce((s, p) => s + p.t, 0) / W.hist.length;
  assert.ok(called > 0.3 && called < 0.5);
  assert.ok(W.hist.every((p) => I.trueUplift(p) >= 0)); // a call never raises risk in this world
});

test('logistic regression recovers known coefficients', () => {
  const r = I.mulberry32(5); const X = [], y = [];
  for (let i = 0; i < 6000; i++) { const x = [I.normal(r), I.normal(r)]; X.push(x); y.push(r() < I.sigmoid(-0.5 + 1.2 * x[0] - 0.8 * x[1]) ? 1 : 0); }
  const b = I.fitLogistic(X, y, 0.01);
  assert.ok(Math.abs(b[0] + 0.5) < 0.1 && Math.abs(b[1] - 1.2) < 0.12 && Math.abs(b[2] + 0.8) < 0.12, JSON.stringify(b));
});

test('ridge sandwich SE matches the empirical spread of the slope', () => {
  const sl = []; let se = 0;
  for (let s = 0; s < 60; s++) {
    const r = I.mulberry32(100 + s); const X = [], y = [];
    for (let i = 0; i < 400; i++) { const x = I.normal(r); X.push([x]); y.push(2 * x + (1 + Math.abs(x)) * I.normal(r)); }
    const m = I.fitRidge(X, y, 0.001); sl.push(m.b[1]); se += Math.sqrt(m.V[1][1]) / 60;
  }
  const mu = sl.reduce((a, b) => a + b, 0) / sl.length;
  const sd = Math.sqrt(sl.reduce((s, v) => s + (v - mu) ** 2, 0) / (sl.length - 1));
  assert.ok(Math.abs(mu - 2) < 0.05);
  assert.ok(se / sd > 0.75 && se / sd < 1.3, `se ${se} sd ${sd}`);
});

test('estimated uplift tracks the true uplift; risk does not', () => {
  const c = (a, b) => { const ma = a.reduce((s, x) => s + x, 0) / a.length, mb = b.reduce((s, x) => s + x, 0) / b.length; let k = 0, va = 0, vb = 0; for (let i = 0; i < a.length; i++) { k += (a[i] - ma) * (b[i] - mb); va += (a[i] - ma) ** 2; vb += (b[i] - mb) ** 2; } return k / Math.sqrt(va * vb); };
  const tu = hist.map(I.trueUplift);
  const cu = c(cv.rows.map((r) => r.uplift), tu), cr = c(cv.rows.map((r) => r.risk), tu);
  assert.ok(cu > 0.6, `uplift corr ${cu}`);
  assert.ok(cr < 0.4, `risk corr ${cr}`);
});

test('AIPW policy values are close to the truth (oracle) at 20% capacity', () => {
  const C = I.compareCV(cv, 0.2, 100);
  const ti = 1000 * (n0 - I.oracle(hist, C.callI)), tr = 1000 * (n0 - I.oracle(hist, C.callR));
  assert.ok(Math.abs(C.avoidedImpact - ti) < 4, `impact est ${C.avoidedImpact} true ${ti}`);
  assert.ok(C.ci.risk[0] <= tr && tr <= C.ci.risk[1] + 1, `risk est CI ${C.ci.risk} true ${tr}`);
  assert.ok(C.ci.extra[0] <= ti - tr && ti - tr <= C.ci.extra[1], `extra CI ${C.ci.extra} true ${ti - tr}`);
});

test('impact ranking beats risk ranking in truth at 10-30% capacity', () => {
  for (const f of [0.1, 0.2, 0.3]) {
    const C = I.compareCV(cv, f, 1);
    const ti = n0 - I.oracle(hist, C.callI), tr = n0 - I.oracle(hist, C.callR);
    assert.ok(ti > tr * 1.1, `frac ${f}: ${ti} vs ${tr}`);
  }
});

test('impact ranking at 20% matches current practice that called ~41%', () => {
  const C = I.compareCV(cv, 0.2, 1);
  const ti = n0 - I.oracle(hist, C.callI), tc = n0 - I.oracle(hist, hist.map((p) => p.t));
  assert.ok(ti > tc * 0.95, `${ti} vs ${tc}`);
});

test('AIPW is unbiased across seeds (policy value at 20%)', () => {
  const errs = [];
  for (let s = 1; s <= 6; s++) {
    const w = I.buildWorld({ seed: 500 + s, histN: 12000 });
    const c = I.crossFit(w, 3), h = c.rows.map((r) => r.p), z = I.oracle(h, new Array(h.length).fill(0));
    const C = I.compareCV(c, 0.2, 1);
    errs.push(C.avoidedImpact - 1000 * (z - I.oracle(h, C.callI)));
  }
  const m = errs.reduce((a, b) => a + b, 0) / errs.length;
  assert.ok(Math.abs(m) < 4, `mean error ${m} (${errs.map((e) => e.toFixed(1))})`);
});

test('top-k respects capacity and cross-fit covers every discharge once', () => {
  const call = I.topK([0.3, 0.1, 0.5, 0.2, 0.4], 0.4);
  assert.deepEqual(call, [0, 0, 1, 0, 1]);
  assert.equal(cv.rows.filter(Boolean).length, W.hist.length);
  assert.ok(new Set(cv.rows.map((r) => r.fold)).size === 5);
});

test('calibration: highest predicted fifth has the largest observed benefit', () => {
  const g = I.calibrationCV(cv);
  assert.equal(g.length, 5);
  assert.ok(g[4].obs > g[0].obs && g[4].obs > 0.05);
});

test("today's worklist: intervals bracket estimates and reasons are named", () => {
  const m = I.fitImpact(W.hist), means = I.featureMeans(W.hist);
  for (const p of W.today) { const s = I.score(m, p); assert.ok(s.lo <= s.uplift && s.uplift <= s.hi); assert.equal(I.reasons(m, p, means, 3).length, 3); }
});
