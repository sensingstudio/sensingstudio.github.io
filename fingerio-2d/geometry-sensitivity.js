require('/Applications/XAMPP/xamppfiles/htdocs/sensingstudio/fingerio/fingerio-dsp.js');
const D = globalThis.FingerDSP, C = D.SOUND_C, fs = 48000;
const m1 = 0.010, m2 = 0.145;

// How far does the solved position move if one path length is off by
// exactly one sample of delay? This is the geometry's own error gain.
function pos(d1, d2) { return D.solve2D(m1, d1, m2, d2); }
const bin = C / fs;                       // 7.15 mm of path per sample
console.log('one sample of delay = ' + (bin * 1000).toFixed(2) + ' mm of path\n');
console.log('point (cm)        d/dd1 (mm per sample)   d/dd2   worst-case');
for (const [x, y] of [[0.06,0.02],[0.10,0.06],[0.16,0.16],[0.08,0.12],[0.20,0.05]]) {
  const r = Math.hypot(x, y);
  const d1 = r + Math.hypot(x, y - m1);
  const d2 = r + Math.hypot(x, y - m2);
  const base = pos(d1, d2);
  const g1 = pos(d1 + bin, d2), g2 = pos(d1, d2 + bin);
  const s1 = Math.hypot(g1.x - base.x, g1.y - base.y) * 1000;
  const s2 = Math.hypot(g2.x - base.x, g2.y - base.y) * 1000;
  console.log(`(${(x*100).toFixed(0).padStart(3)}, ${(y*100).toFixed(0).padStart(3)})` +
              `           ${s1.toFixed(1).padStart(6)}          ${s2.toFixed(1).padStart(6)}   ${Math.hypot(s1,s2).toFixed(1)}`);
}

// And what a wider baseline would buy.
console.log('\nbaseline sweep at (10, 6) cm, error per sample of delay:');
for (const b of [0.02, 0.05, 0.10, 0.135, 0.20]) {
  const M1 = 0.005, M2 = 0.005 + b;
  const x = 0.10, y = 0.06, r = Math.hypot(x, y);
  const d1 = r + Math.hypot(x, y - M1), d2 = r + Math.hypot(x, y - M2);
  const base = D.solve2D(M1, d1, M2, d2);
  const g = D.solve2D(M1, d1 + bin, M2, d2);
  const s = base && g ? Math.hypot(g.x - base.x, g.y - base.y) * 1000 : NaN;
  console.log(`  mic spacing ${(b*100).toFixed(1).padStart(5)} cm  ->  ${s.toFixed(1).padStart(6)} mm per sample`);
}
