// Regenerates the committed data files from the seed and prints every headline
// number the page shows. The page rebuilds the same world in the browser from
// core/impact.js; these files are for inspection.
//   node scripts/generate.mjs
import { writeFileSync, mkdirSync } from 'node:fs';
import * as I from '../core/impact.js';

const W = I.buildWorld();
mkdirSync(new URL('../data/', import.meta.url), { recursive: true });
const head = 'id,discharge_date,market,age,female,dx,hcc,admits_12m,ed_6m,meds,lives_alone,pcp_visit_6m,disposition,los,dual,called_48h,readmit_30d';
const rows = W.hist.map((p) => [p.id, new Date(p.date).toISOString().slice(0, 10), I.MARKETS[p.market].id, p.age, p.female, p.dx, p.hcc, p.prior, p.ed, p.meds, p.alone, p.pcp, p.dispo, p.los, p.dual, p.t, p.y].join(','));
writeFileSync(new URL('../data/discharges_jan_sep_2026.csv', import.meta.url), [head, ...rows].join('\n') + '\n');

const cv = I.crossFit(W, 5);
const hist = cv.rows.map((r) => r.p), n0 = I.oracle(hist, new Array(hist.length).fill(0));
const at = (f) => {
  const C = I.compareCV(cv, f, 200);
  return {
    capacity: f,
    estimated: { impact: +C.avoidedImpact.toFixed(2), risk: +C.avoidedRisk.toFixed(2), extra: +C.extra.toFixed(2), current: +(1000 * (C.rate.none - C.rate.current)).toFixed(2), ci: C.ci },
    truth: { impact: +(1000 * (n0 - I.oracle(hist, C.callI))).toFixed(2), risk: +(1000 * (n0 - I.oracle(hist, C.callR))).toFixed(2), current: +(1000 * (n0 - I.oracle(hist, hist.map((p) => p.t)))).toFixed(2), best_possible: +(1000 * (n0 - I.oracle(hist, I.topK(hist.map(I.trueUplift), f)))).toFixed(2) },
  };
};
const m = I.fitImpact(W.hist);
const today = W.today.map((p) => ({ p, s: I.score(m, p) }));
const k = I.CONFIG.capacityToday;
const sumTop = (key) => [...today].sort((a, b) => b.s[key] - a.s[key]).slice(0, k).reduce((s, r) => s + r.s.uplift, 0);
const report = {
  seed: W.seed, discharges: W.hist.length, called_share: +(W.hist.reduce((s, p) => s + p.t, 0) / W.hist.length).toFixed(4), readmit_rate: +(W.hist.reduce((s, p) => s + p.y, 0) / W.hist.length).toFixed(4),
  policy: [0.1, 0.2, 0.3].map(at),
  calibration: I.calibrationCV(cv).map((g) => ({ ...g, pred: +g.pred.toFixed(4), obs: +g.obs.toFixed(4), lo: +g.lo.toFixed(4), hi: +g.hi.toFixed(4), risk: +g.risk.toFixed(4) })),
  today: { patients: today.length, calls: k, expected_avoided_by_impact: +sumTop('uplift').toFixed(3), expected_avoided_by_risk: +sumTop('risk').toFixed(3) },
};
writeFileSync(new URL('../data/report.json', import.meta.url), JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify(report, null, 2));
