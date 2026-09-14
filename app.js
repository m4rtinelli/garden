import * as THREE from 'three/webgpu';
import {
  Fn, uniform, attribute, varying, uv, float, vec2, vec3, vec4, mix, smoothstep, clamp, floor, abs, sin, cos, pow,
  max, min, mod, normalize, cross, dot, fract, positionGeometry, normalGeometry, positionWorld, normalWorld,
  faceDirection, screenUV, screenSize, fwidth, hash, mx_noise_float, mx_fractal_noise_float, shadow, pass, sign, cameraPosition
} from 'three/tsl';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'lil-gui';

/* ------------------------------------------------------------------ */
/*  seeded randomness + CPU noise                                      */
/* ------------------------------------------------------------------ */
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function hash2(x, y, s) {
  let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(s | 0, 2246822519)) | 0;
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296;
}
function vnoise(x, y, s) {
  const xi = Math.floor(x), yi = Math.floor(y);
  const xf = x - xi, yf = y - yi;
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf);
  const a = hash2(xi, yi, s), b = hash2(xi + 1, yi, s), c = hash2(xi, yi + 1, s), d = hash2(xi + 1, yi + 1, s);
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v;
}
function fbm(x, y, s, oct = 4) {
  let amp = 0.5, sum = 0, norm = 0;
  for (let i = 0; i < oct; i++) {
    sum += amp * vnoise(x, y, s + i * 17);
    norm += amp; x = x * 2.03 + 11.7; y = y * 2.01 - 7.3; amp *= 0.5;
  }
  return sum / norm;
}

/* ------------------------------------------------------------------ */
/*  parameters                                                         */
/* ------------------------------------------------------------------ */
const P = {
  seed: 1913,
  plantCount: 420,
  fieldRadius: 9,
  spareSeeds: 80,
  densityThreshold: 0.42,
  warp: 1.4,
  barePatches: 0.55,
  clusterRadius: 0.75,
  dahliaRatio: 0.28,
  rockCount: 16,
  timeScale: 1.0,
  delay: 2.5, grow: 6, keep: 14, die: 7,
  windStrength: 0.08, windSpeed: 0.9,
  autoOrbit: false,
  cameraMode: 'walk',
  walkSpeed: 2.2,
  brushStrength: 0.35,
  outlines: true,
  paused: false,
  showHud: true,
  lookSmoothing: 0.55,
  mouseSensitivity: 1.0,
  recordSeconds: 0,
  recordBitrate: 24,
};

const FLOWER_KINDS = {
  rose:   { rings: 3, petals: 5, petalGrowth: 0.7, length: 0.2, width: 0.15, curl: 0.16, cup: 0.07, pointed: 0.0, coreSize: 0.035, spread: 0.04, openInner: 0.25, maxOpen: 1.15, headScale: 1.0 },
  dahlia: { rings: 4, petals: 8, petalGrowth: 0.5, length: 0.26, width: 0.065, curl: 0.05, cup: 0.02, pointed: 0.4, coreSize: 0.05, spread: 0.02, openInner: 0.35, maxOpen: 1.55, headScale: 1.0 },
  lily:   { rings: 2, petals: 3, petalGrowth: 1.0, length: 0.34, width: 0.1, curl: 0.3, cup: -0.03, pointed: 0.8, coreSize: 0.025, spread: 0.015, openInner: 0.7, maxOpen: 1.7, headScale: 1.0 },
  poppy:  { rings: 2, petals: 4, petalGrowth: 0.25, length: 0.24, width: 0.24, curl: 0.05, cup: 0.12, pointed: 0.0, coreSize: 0.06, spread: 0.02, openInner: 0.8, maxOpen: 1.5, headScale: 0.9 },
  daisy:  { rings: 2, petals: 12, petalGrowth: 0.5, length: 0.22, width: 0.045, curl: 0.02, cup: 0.0, pointed: 0.3, coreSize: 0.07, spread: 0.01, openInner: 0.9, maxOpen: 1.6, headScale: 0.9 },
  tulip:  { rings: 2, petals: 3, petalGrowth: 1.0, length: 0.3, width: 0.17, curl: -0.04, cup: 0.12, pointed: 0.3, coreSize: 0.02, spread: 0.02, openInner: 0.2, maxOpen: 0.6, headScale: 1.0 },
  peony:  { rings: 5, petals: 6, petalGrowth: 0.6, length: 0.18, width: 0.16, curl: 0.12, cup: 0.09, pointed: 0.0, coreSize: 0.03, spread: 0.03, openInner: 0.15, maxOpen: 1.3, headScale: 1.1 },
};
const SHAPES = {
  A: { kind: 'rose', ...FLOWER_KINDS.rose },
  B: { kind: 'dahlia', ...FLOWER_KINDS.dahlia },
};

const U = {
  // toon / woodblock
  thresholdLow: uniform(0.18), thresholdHigh: uniform(0.42), colorLevels: uniform(2),
  thresholdNoiseScale: uniform(6.0), thresholdNoiseStrength: uniform(0.22),
  shadowTint: uniform(new THREE.Color('#6a6a86')), highlightTint: uniform(new THREE.Color('#fffaf0')),
  outlineWidth: uniform(0.011), petalRim: uniform(0.08), petalTranslucency: uniform(0.7), inkColor: uniform(new THREE.Color('#262631')),
  // ink wash shadow
  washAt: uniform(0.35), washSoft: uniform(0.45), washBleed: uniform(0.35), washMottle: uniform(0.55), washScale: uniform(0.9),
  washStr: uniform(0.75), contourWobbleScale: uniform(1.6), contourWobble: uniform(0.12), contourWidth: uniform(1.6),
  contourShade: uniform(0.5), contourStr: uniform(0.85),
  washColor: uniform(new THREE.Color('#6f7286')), contourColor: uniform(new THREE.Color('#3a3a48')),
  paperColor: uniform(new THREE.Color('#e9e2d3')), groundGrain: uniform(0.06), mistNear: uniform(10.0), mistFar: uniform(26.0),
  // silk weave
  silkOn: uniform(1), threadCount: uniform(220), irregularity: uniform(0.35), sharpness: uniform(1.4),
  silkStrength: uniform(0.22), threadTone: uniform(0.08), blotchStrength: uniform(0.18), blotchScale: uniform(2.2),
  silkTint: uniform(new THREE.Color('#fbf6ea')), vignette: uniform(0.28),
  // plants
  stemRadius: uniform(0.022), radiusAttenuation: uniform(0.55), baseFlare: uniform(0.9), startScale: uniform(0.15),
  stemBury: uniform(0.12), maxOpenA: uniform(SHAPES.A.maxOpen), maxOpenB: uniform(SHAPES.B.maxOpen),
  shedStagger: uniform(0.6), shedRise: uniform(0.9), shedSpread: uniform(0.55), riseVariance: uniform(0.5),
  leafCurl: uniform(3.2),
  // palette
  roseBase: uniform(new THREE.Color('#b8455a')), roseTip: uniform(new THREE.Color('#e8a3a0')),
  dahliaBase: uniform(new THREE.Color('#d9662e')), dahliaTip: uniform(new THREE.Color('#f2c14e')),
  stemColor: uniform(new THREE.Color('#5c6b4a')), leafColor: uniform(new THREE.Color('#6f7f52')), rockColor: uniform(new THREE.Color('#8a8f96')),
  lightDir: uniform(new THREE.Vector3(0.5, 0.8, 0.3).normalize()),
};

const PRESETS = {
  'Sumi on silk': {
    paperColor: '#e9e2d3', inkColor: '#262631', washColor: '#6f7286', contourColor: '#3a3a48',
    roseBase: '#b8455a', roseTip: '#e8a3a0', dahliaBase: '#d9662e', dahliaTip: '#f2c14e',
    stemColor: '#5c6b4a', leafColor: '#6f7f52', rockColor: '#8a8f96', shadowTint: '#6a6a86', highlightTint: '#fffaf0',
    silkTint: '#fbf6ea', silkStrength: 0.22, washStr: 0.75, contourStr: 0.85, colorLevels: 2,
  },
  'Akira red': {
    paperColor: '#c9382b', inkColor: '#1b0f10', washColor: '#7a1f18', contourColor: '#1b0f10',
    roseBase: '#efe3d2', roseTip: '#fbf7ef', dahliaBase: '#ffb703', dahliaTip: '#ffe19a',
    stemColor: '#2b1a1a', leafColor: '#3b2222', rockColor: '#4a2a28', shadowTint: '#8a4a48', highlightTint: '#fff4ea',
    silkTint: '#ffffff', silkStrength: 0.16, washStr: 0.9, contourStr: 0.9, colorLevels: 2,
  },
  'Night bloom': {
    paperColor: '#14171c', inkColor: '#dfe6ea', washColor: '#2a3140', contourColor: '#9fb3c8',
    roseBase: '#5d86ad', roseTip: '#dbe7f0', dahliaBase: '#e9c46a', dahliaTip: '#f4a261',
    stemColor: '#3b4a4a', leafColor: '#4c6157', rockColor: '#3d434a', shadowTint: '#5b6478', highlightTint: '#ffffff',
    silkTint: '#e6ecf5', silkStrength: 0.3, washStr: 0.85, contourStr: 0.6, colorLevels: 3,
  },
  'Bare paper': {
    paperColor: '#f0ede6', inkColor: '#2a2a2a', washColor: '#8c8c8c', contourColor: '#3a3a3a',
    roseBase: '#d6d6d6', roseTip: '#ffffff', dahliaBase: '#bdbdbd', dahliaTip: '#f2f2f2',
    stemColor: '#6b6b6b', leafColor: '#7c7c7c', rockColor: '#9a9a9a', shadowTint: '#7a7a7a', highlightTint: '#ffffff',
    silkTint: '#ffffff', silkStrength: 0.28, washStr: 0.7, contourStr: 1.0, colorLevels: 2,
  },
  // Hokusai's Prussian-blue prints: one imported pigment doing all the work on cream kozo paper.
  'Hokusai indigo': {
    paperColor: '#efe4cc', inkColor: '#1b2740', washColor: '#3d5a8c', contourColor: '#1b2740',
    roseBase: '#2f4f86', roseTip: '#c9d7e8', dahliaBase: '#f2ead6', dahliaTip: '#ffffff',
    stemColor: '#3f5a5a', leafColor: '#4d6b62', rockColor: '#6c7d8c', shadowTint: '#5f6f8f', highlightTint: '#fff9ec',
    silkTint: '#fbf5e6', silkStrength: 0.24, washStr: 0.8, contourStr: 0.9, colorLevels: 2,
  },
  // Hiroshige's evening skies: bokashi gradients of plum, rose and dusk-grey.
  'Hiroshige dusk': {
    paperColor: '#d9b9a6', inkColor: '#3a2a3c', washColor: '#7b5f80', contourColor: '#3a2a3c',
    roseBase: '#b25d7a', roseTip: '#f0c7cf', dahliaBase: '#e08a5a', dahliaTip: '#f7d9b0',
    stemColor: '#5a4a57', leafColor: '#6f6068', rockColor: '#8a7480', shadowTint: '#7a6284', highlightTint: '#fff1e6',
    silkTint: '#fdeee4', silkStrength: 0.2, washStr: 0.85, contourStr: 0.7, colorLevels: 3,
  },
  // Sumi with a single hanko-red: monochrome ink, only the flowers carry vermilion.
  'Sumi & cinnabar': {
    paperColor: '#ece7db', inkColor: '#1e1e22', washColor: '#5a5a60', contourColor: '#1e1e22',
    roseBase: '#c8321f', roseTip: '#ef6b4a', dahliaBase: '#d94e2a', dahliaTip: '#f7a06b',
    stemColor: '#4b4b4f', leafColor: '#5e5e62', rockColor: '#8b8b8f', shadowTint: '#6a6a72', highlightTint: '#ffffff',
    silkTint: '#fbf8f0', silkStrength: 0.3, washStr: 0.8, contourStr: 1.0, colorLevels: 2,
  },
  // Rinpa folding screens: white blossoms and deep pine on gold leaf.
  'Gold screen': {
    paperColor: '#c8a55a', inkColor: '#2a2118', washColor: '#8c6a2c', contourColor: '#2a2118',
    roseBase: '#f4ecd8', roseTip: '#ffffff', dahliaBase: '#1f4a3a', dahliaTip: '#5f8f6e',
    stemColor: '#2f3d2a', leafColor: '#3d5636', rockColor: '#6b5a3a', shadowTint: '#8a7548', highlightTint: '#fff4d6',
    silkTint: '#ffe9b5', silkStrength: 0.34, washStr: 0.85, contourStr: 0.8, colorLevels: 2,
  },
  // Celadon glaze: pale green-grey porcelain with peach blossoms.
  'Celadon': {
    paperColor: '#dfe6dc', inkColor: '#2c463f', washColor: '#7ea391', contourColor: '#2c463f',
    roseBase: '#e39a86', roseTip: '#fbe0d2', dahliaBase: '#f0c48b', dahliaTip: '#fff0d0',
    stemColor: '#5f7a6a', leafColor: '#6f8f78', rockColor: '#9ab0a4', shadowTint: '#7d9a8c', highlightTint: '#ffffff',
    silkTint: '#f2f7f1', silkStrength: 0.2, washStr: 0.7, contourStr: 0.6, colorLevels: 3,
  },
  // Neo-Tokyo at night: the film's other palette — sodium lamps, pink neon, deep ink.
  'Neo-Tokyo neon': {
    paperColor: '#0f0c14', inkColor: '#ff3f7a', washColor: '#2a1b3a', contourColor: '#ff3f7a',
    roseBase: '#ff2d6f', roseTip: '#ffb3d1', dahliaBase: '#18e0e0', dahliaTip: '#c8fff8',
    stemColor: '#2b2340', leafColor: '#3a2f55', rockColor: '#26203a', shadowTint: '#5a3f7a', highlightTint: '#ffffff',
    silkTint: '#e8dcff', silkStrength: 0.35, washStr: 0.9, contourStr: 0.9, colorLevels: 3,
  },
  // Sepia woodcut: aged kozo paper, walnut ink, ochre blooms.
  'Sepia woodcut': {
    paperColor: '#e3d2b0', inkColor: '#4a3220', washColor: '#8b6a45', contourColor: '#4a3220',
    roseBase: '#b7783a', roseTip: '#e9c48b', dahliaBase: '#8a4a2a', dahliaTip: '#d9905a',
    stemColor: '#5e4a30', leafColor: '#6f5a38', rockColor: '#8f7a5a', shadowTint: '#8a6f52', highlightTint: '#fff3dc',
    silkTint: '#fff0d4', silkStrength: 0.3, washStr: 0.85, contourStr: 1.0, colorLevels: 2,
  },
  // Snow garden: near-white ground, blue-grey shadows, red camellias.
  'Snow & camellia': {
    paperColor: '#f4f3f0', inkColor: '#2d3340', washColor: '#aeb8c8', contourColor: '#4c5668',
    roseBase: '#b01f2e', roseTip: '#e85a68', dahliaBase: '#f2f0ea', dahliaTip: '#ffffff',
    stemColor: '#3f4a45', leafColor: '#4e5f55', rockColor: '#9aa3ae', shadowTint: '#8f9ab0', highlightTint: '#ffffff',
    silkTint: '#ffffff', silkStrength: 0.18, washStr: 0.75, contourStr: 0.6, colorLevels: 2,
  },
  // Aizome: indigo-dyed cloth, white resist-paste blossoms.
  'Indigo cloth': {
    paperColor: '#1f2f4f', inkColor: '#eef1f5', washColor: '#152238', contourColor: '#9fb4d6',
    roseBase: '#e8eef5', roseTip: '#ffffff', dahliaBase: '#c7d6ec', dahliaTip: '#f4f8ff',
    stemColor: '#8fa3c2', leafColor: '#a3b6d1', rockColor: '#3a4d6e', shadowTint: '#7f93b5', highlightTint: '#ffffff',
    silkTint: '#dfe8f7', silkStrength: 0.4, washStr: 0.8, contourStr: 0.5, colorLevels: 2,
  },
};

/* ------------------------------------------------------------------ */
/*  renderer / scene                                                   */
/* ------------------------------------------------------------------ */
const stage = document.getElementById('stage');
const loading = document.getElementById('loading');
const qp = new URLSearchParams(location.search);
const forceWebGL = qp.has('webgl') || !('gpu' in navigator);
const renderer = new THREE.WebGPURenderer({ antialias: true, forceWebGL });
// If the WebGPU device misbehaves (driver loss, an API mismatch), reload once on the WebGL2 backend.
function fallbackToWebGL() {
  if (qp.has('webgl')) return;
  const u = new URL(location.href); u.searchParams.set('webgl', '1'); location.replace(u.toString());
}
renderer.onDeviceLost = () => fallbackToWebGL();
window.addEventListener('error', (e) => { if (/GPU/i.test(String(e.message))) fallbackToWebGL(); });
window.addEventListener('unhandledrejection', (e) => { if (/GPU/i.test(String(e.reason && e.reason.message || e.reason))) fallbackToWebGL(); });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFShadowMap;
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = U.paperColor.value;
const camera = new THREE.PerspectiveCamera(38, window.innerWidth / window.innerHeight, 0.1, 100);
camera.rotation.order = 'YXZ';
camera.position.set(3.2, 3.0, 9.6);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enabled = false;
controls.target.set(0, 0.55, 0);
controls.enableDamping = true;
controls.maxPolarAngle = Math.PI * 0.49;
controls.minDistance = 2.5; controls.maxDistance = 22;
controls.autoRotateSpeed = 0.35;

const sun = new THREE.DirectionalLight(0xffffff, 1);
sun.position.set(5, 8, 3);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
sun.shadow.camera.left = -9; sun.shadow.camera.right = 9;
sun.shadow.camera.top = 9; sun.shadow.camera.bottom = -9;
sun.shadow.camera.near = 1; sun.shadow.camera.far = 30;
sun.shadow.bias = -0.0008;
sun.shadow.radius = 4;
scene.add(sun, sun.target);
U.lightDir.value.copy(sun.position).normalize();

/* ------------------------------------------------------------------ */
/*  TSL helpers                                                         */
/* ------------------------------------------------------------------ */
const qrot = Fn(([q, v]) => {
  const t = cross(q.xyz, v).mul(2.0);
  return v.add(t.mul(q.w)).add(cross(q.xyz, t));
});
const rodrigues = Fn(([v, axis, ang]) => {
  const c = cos(ang), s = sin(ang);
  return v.mul(c).add(cross(axis, v).mul(s)).add(axis.mul(dot(axis, v).mul(float(1.0).sub(c))));
});
const bezier = Fn(([p0, p1, p2, p3, t]) => {
  const it = float(1.0).sub(t);
  return p0.mul(it.mul(it).mul(it)).add(p1.mul(it.mul(it).mul(t).mul(3.0)))
    .add(p2.mul(it.mul(t).mul(t).mul(3.0))).add(p3.mul(t.mul(t).mul(t)));
});
const bezierTangent = Fn(([p0, p1, p2, p3, t]) => {
  const it = float(1.0).sub(t);
  return p1.sub(p0).mul(it.mul(it).mul(3.0)).add(p2.sub(p1).mul(it.mul(t).mul(6.0))).add(p3.sub(p2).mul(t.mul(t).mul(3.0)));
});

// Woodblock toon: N·L remapped between two thresholds, a world-space noise jitters the
// threshold so the boundary reads like a carved edge, then quantised into colour levels.
const toon = Fn(([albedo, nW, pW, translucency]) => {
  const d = dot(nW, U.lightDir);
  const ndl = mix(max(d, 0.0), abs(d), translucency);
  const thresholdNoise = mx_fractal_noise_float(pW.mul(U.thresholdNoiseScale), 3).mul(0.5).mul(U.thresholdNoiseStrength);
  const preShade = clamp(ndl.sub(U.thresholdLow.add(thresholdNoise)).div(U.thresholdHigh.sub(U.thresholdLow)), 0.0, 1.0);
  const levels = U.colorLevels.sub(1.0);
  const quantized = floor(preShade.mul(levels).add(0.5)).div(levels);
  const lit = mix(albedo.mul(U.shadowTint), albedo.mul(U.highlightTint), quantized);
  const mist = smoothstep(U.mistNear, U.mistFar, pW.sub(cameraPosition).length());
  return mix(lit, U.paperColor, mist);
});

const inkMist = Fn(() => mix(U.inkColor, U.paperColor, smoothstep(U.mistNear, U.mistFar, positionWorld.sub(cameraPosition).length())));

/* ------------------------------------------------------------------ */
/*  ground: ink-wash shadow                                            */
/* ------------------------------------------------------------------ */
const groundMat = new THREE.MeshBasicNodeMaterial();
groundMat.colorNode = Fn(() => {
  const shade = float(1.0).sub(shadow(sun));
  const xz = positionWorld.xz;
  const noise = mx_fractal_noise_float(xz.mul(U.washScale), 4).mul(0.5).add(0.5);
  const fill = smoothstep(U.washAt, U.washAt.add(U.washSoft), shade.add(noise.sub(0.5).mul(U.washBleed)));
  const wash = fill.mul(float(1.0).sub(noise.mul(U.washMottle)).max(0.0));
  const wobble = mx_noise_float(xz.mul(U.contourWobbleScale)).mul(U.contourWobble);
  const penWidth = fwidth(shade).mul(U.contourWidth).max(0.0001);
  const line = float(1.0).sub(smoothstep(0.0, penWidth, shade.sub(U.contourShade.add(wobble)).abs())).mul(smoothstep(0.02, 0.2, shade));
  const shColor = mix(U.washColor, U.contourColor, line);
  const grain = mx_fractal_noise_float(xz.mul(3.0).add(31.0), 3).mul(U.groundGrain);
  const paper = U.paperColor.add(grain);
  const dist = smoothstep(U.mistNear, U.mistFar, positionWorld.sub(cameraPosition).length());
  const rgb = mix(mix(paper, shColor, max(wash.mul(U.washStr), line.mul(U.contourStr))), U.paperColor, dist);
  return vec4(rgb, 1.0);
})();
const ground = new THREE.Mesh(new THREE.PlaneGeometry(70, 70), groundMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

/* ------------------------------------------------------------------ */
/*  plant geometry builders                                            */
/* ------------------------------------------------------------------ */
function makeStemTemplate(rings = 18, sides = 6) {
  const g = new THREE.InstancedBufferGeometry();
  const pos = [], uvs = [], idx = [];
  for (let r = 0; r < rings; r++) {
    for (let s = 0; s <= sides; s++) {
      pos.push(0, 0, 0); uvs.push(r / (rings - 1), s / sides);
    }
  }
  for (let r = 0; r < rings - 1; r++) for (let s = 0; s < sides; s++) {
    const a = r * (sides + 1) + s, b = a + sides + 1;
    idx.push(a, a + 1, b, b, a + 1, b + 1);
  }
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(idx);
  return g;
}

// A petal is a small grid; each vertex carries the petal's pivot (base point), the axis it
// opens around, and (petalId, openFactor, ringFrac, isCore). Islands stay separate so the
// petals can shed one by one in the vertex shader.
function makeFlower(shape) {
  const rings = [];
  const rc = Math.max(1, Math.round(shape.rings));
  for (let i = 0; i < rc; i++) {
    const f = rc > 1 ? i / (rc - 1) : 1;
    rings.push({
      n: Math.max(1, Math.round(shape.petals * (1 + i * shape.petalGrowth))),
      r: 0.02 + i * shape.spread,
      L: shape.length * (0.6 + 0.4 * f),
      W: shape.width * (0.7 + 0.3 * f),
      open: THREE.MathUtils.lerp(shape.openInner, 1, f),
      y: 0.01 - i * 0.01,
    });
  }
  const kind = shape.kind;
  const nu = 5, nv = 8;
  const pos = [], nrm = [], piv = [], ax = [], info = [], idx = [];
  let base = 0, petalId = 1;
  const rng = mulberry32(7 + shape.petals * 3 + rc);
  const ringCount = rings.length;
  rings.forEach((ring, ri) => {
    for (let k = 0; k < ring.n; k++) {
      const a = (k / ring.n) * Math.PI * 2 + ri * 0.9 + (rng() - 0.5) * 0.25;
      const ca = Math.cos(a), sa = Math.sin(a);
      const O = [ca, 0, sa], T = [-sa, 0, ca];
      const pv = [ring.r * ca, ring.y, ring.r * sa];
      const L = ring.L * (0.9 + rng() * 0.2), W = ring.W * (0.9 + rng() * 0.2);
      const curl = shape.curl, cup = shape.cup;
      for (let j = 0; j < nv; j++) {
        const v = j / (nv - 1);
        const wprof = Math.pow(Math.sin(Math.PI * (0.06 + v * 0.94)), THREE.MathUtils.lerp(0.55, 0.35, shape.pointed)) * (1 - shape.pointed * 0.65 * v * v);
        for (let i = 0; i < nu; i++) {
          const u = (i / (nu - 1)) * 2 - 1;
          const wx = u * wprof * W;
          const bul = curl * v * v - cup * u * u * v;
          pos.push(pv[0] + T[0] * wx + O[0] * bul, pv[1] + v * L, pv[2] + T[2] * wx + O[2] * bul);
          nrm.push(0, 0, 0);
          piv.push(pv[0], pv[1], pv[2], i / (nu - 1));
          ax.push(sa, 0, -ca, v);
          info.push(petalId, ring.open, ri / Math.max(1, ringCount - 1), 0);
        }
      }
      for (let j = 0; j < nv - 1; j++) for (let i = 0; i < nu - 1; i++) {
        const q = base + j * nu + i;
        idx.push(q, q + nu, q + 1, q + nu, q + nu + 1, q + 1);
      }
      base += nu * nv; petalId++;
    }
  });
  // core
  const core = new THREE.SphereGeometry(Math.max(0.005, shape.coreSize), 10, 7);
  const cp = core.attributes.position, cn = core.attributes.normal, cuv = core.attributes.uv, ci = core.index;
  for (let i = 0; i < cp.count; i++) {
    pos.push(cp.getX(i) * 1, cp.getY(i) * 0.6 + 0.03, cp.getZ(i));
    nrm.push(cn.getX(i), cn.getY(i), cn.getZ(i));
    piv.push(0, 0.03, 0, cuv.getX(i)); ax.push(0, 0, 1, 0.15); info.push(0, 0, 0, 1);
  }
  for (let i = 0; i < ci.count; i++) idx.push(base + ci.getX(i));
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  g.setAttribute('pivot', new THREE.Float32BufferAttribute(piv, 4));
  g.setAttribute('axis', new THREE.Float32BufferAttribute(ax, 4));
  g.setAttribute('info', new THREE.Float32BufferAttribute(info, 4));
  g.setIndex(idx);
  // petal normals (skip the core, already set)
  const p = g.attributes.position, n = g.attributes.normal;
  const tmp = new THREE.Vector3(), e1 = new THREE.Vector3(), e2 = new THREE.Vector3(), A = new THREE.Vector3(), B = new THREE.Vector3(), C = new THREE.Vector3();
  const petalVerts = base;
  const acc = new Float32Array(petalVerts * 3);
  for (let i = 0; i < idx.length; i += 3) {
    const ia = idx[i], ib = idx[i + 1], ic = idx[i + 2];
    if (ia >= petalVerts) continue;
    A.fromBufferAttribute(p, ia); B.fromBufferAttribute(p, ib); C.fromBufferAttribute(p, ic);
    e1.subVectors(B, A); e2.subVectors(C, A); tmp.crossVectors(e1, e2);
    for (const k of [ia, ib, ic]) { acc[k * 3] += tmp.x; acc[k * 3 + 1] += tmp.y; acc[k * 3 + 2] += tmp.z; }
  }
  for (let i = 0; i < petalVerts; i++) {
    tmp.set(acc[i * 3], acc[i * 3 + 1], acc[i * 3 + 2]).normalize();
    n.setXYZ(i, tmp.x, tmp.y, tmp.z);
  }
  return g;
}

function makeLeaf() {
  const nu = 5, nv = 9, pos = [], nrm = [], uvs = [], idx = [];
  for (let j = 0; j < nv; j++) {
    const v = j / (nv - 1);
    const w = Math.pow(Math.sin(Math.PI * (0.02 + v * 0.98)), 0.7) * 0.09;
    for (let i = 0; i < nu; i++) {
      const u = (i / (nu - 1)) * 2 - 1;
      pos.push(u * w, v * 0.32, -Math.abs(u) * 0.02 * (1 - v));
      nrm.push(0, 0, 1); uvs.push(i / (nu - 1), v);
    }
  }
  for (let j = 0; j < nv - 1; j++) for (let i = 0; i < nu - 1; i++) {
    const q = j * nu + i; idx.push(q, q + nu, q + 1, q + nu, q + nu + 1, q + 1);
  }
  const g = new THREE.InstancedBufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.Float32BufferAttribute(nrm, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
  g.setIndex(idx);
  return g;
}

// Thin sheets (petals, leaves) get their line from an in-plane expansion toward the edge
// plus a push away from the camera, so the hull reads from either side.
const sheetOutline = Fn(([pos, nrm, edgeW, width]) => {
  const toCam = normalize(cameraPosition.sub(pos));
  const s = sign(dot(nrm, toCam));
  return pos.add(edgeW.mul(width)).sub(nrm.mul(s).mul(width.mul(1.5)));
});

/* ------------------------------------------------------------------ */
/*  materials                                                          */
/* ------------------------------------------------------------------ */
const P0 = vec3(0.0, U.stemBury.negate(), 0.0);

// Stem: a template tube swept along a per-instance cubic Bézier in the vertex shader.
function stemPosition() {
  const p1 = attribute('iP1', 'vec3'), p2 = attribute('iP2', 'vec3'), p3 = attribute('iP3', 'vec3');
  const st = attribute('iState', 'vec4'); // growth, swayX, swayZ, size
  const iPos = attribute('iPos', 'vec3');
  const t = uv().x, ang = uv().y.mul(Math.PI * 2);
  const growth = st.x;
  const c = bezier(P0, p1, p2, p3, t);
  const T = normalize(bezierTangent(P0, p1, p2, p3, t));
  const N = normalize(cross(vec3(1.0, 0.0, 0.0), T));
  const B = cross(T, N);
  const taper = float(1.0).sub(float(1.0).sub(U.radiusAttenuation).mul(t)).add(U.baseFlare.mul(pow(float(1.0).sub(t), 3.0)));
  const rScale = U.startScale.add(growth.mul(float(1.0).sub(U.startScale)));
  const front = mix(1.0, 0.3, smoothstep(growth.sub(0.12), growth, t));
  const r = U.stemRadius.mul(st.w).mul(taper).mul(rScale).mul(front);
  const nLocal = N.mul(cos(ang)).add(B.mul(sin(ang)));
  const sway = vec3(st.y, 0.0, st.z).mul(t.mul(t));
  return { pos: iPos.add(c).add(nLocal.mul(r)).add(sway), nrm: nLocal, growth, t };
}
const stemMat = new THREE.MeshBasicNodeMaterial();
{
  const s = stemPosition();
  const nW = varying(s.nrm);
  stemMat.positionNode = s.pos;
  stemMat.maskNode = uv().x.lessThanEqual(varying(s.growth).add(0.0));
  stemMat.colorNode = Fn(() => {
    const albedo = U.stemColor.mul(mix(0.85, 1.0, uv().x));
    return vec4(toon(albedo, normalize(nW), positionWorld, 0.0), 1.0);
  })();
}
const stemOutlineMat = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide });
{
  const s = stemPosition();
  stemOutlineMat.positionNode = s.pos.add(s.nrm.mul(U.outlineWidth.mul(0.6)));
  stemOutlineMat.maskNode = uv().x.lessThanEqual(varying(s.growth));
  stemOutlineMat.colorNode = vec4(inkMist(), 1.0);
}

// Flower head: petals open around their pivot, then shed (shrink to pivot, lift, fan out).
function flowerPosition(maxOpen) {
  const pivot4 = attribute('pivot', 'vec4'), axis4 = attribute('axis', 'vec4'), info = attribute('info', 'vec4');
  const pivot = pivot4.xyz, axis = axis4.xyz, puv = vec2(pivot4.w, axis4.w);
  const iPS = attribute('iPos', 'vec4'), iQuat = attribute('iQuat', 'vec4');
  const iPos = iPS.xyz, iScale = iPS.w;
  const anim = attribute('iAnim', 'vec4'); // open, shed, colorMix, unused
  const petalId = info.x, openFactor = info.y, isCore = info.w;
  const ang = anim.x.mul(maxOpen).mul(openFactor);
  const p1 = pivot.add(rodrigues(positionGeometry.sub(pivot), axis, ang));
  const n1 = rodrigues(normalGeometry, axis, ang);
  // in-plane edge direction for the outline hull: across the petal, plus toward the tip
  const tipK = max(puv.y.sub(0.75), 0.0).mul(4.0);
  const edgeDir = mix(axis.negate().mul(puv.x.mul(2.0).sub(1.0)).add(vec3(0.0, tipK, 0.0)), normalGeometry, isCore);
  const e1 = rodrigues(edgeDir, axis, ang);
  const startJitter = fract(sin(petalId.mul(127.1)).mul(43758.5453));
  const heightJitter = fract(sin(petalId.mul(127.1).add(7.13)).mul(43758.5453));
  const shed = anim.y;
  const tt = clamp(shed.sub(startJitter.mul(U.shedStagger)).div(float(1.0).sub(U.shedStagger)), 0.0, 1.0);
  const ease = tt.mul(tt).mul(float(3.0).sub(tt.mul(2.0)));
  const coreEase = smoothstep(0.85, 1.0, shed);
  const easeAll = mix(ease, coreEase, isCore);
  const shrunk = pivot.add(p1.sub(pivot).mul(float(1.0).sub(easeAll)));
  const height = float(1.0).add(heightJitter.sub(0.5).mul(2.0).mul(U.riseVariance));
  const lift = U.shedRise.mul(max(height, 0.0)).mul(ease).mul(float(1.0).sub(isCore));
  const outward = normalize(vec3(pivot.x, 0.0, pivot.z).add(vec3(0.0001, 0.0, 0.0)));
  const fan = outward.mul(U.shedSpread).mul(ease).mul(float(1.0).sub(isCore));
  const local = shrunk.add(vec3(0.0, lift, 0.0)).add(fan);
  const pos = iPos.add(qrot(iQuat, local.mul(iScale)));
  const nrm = qrot(iQuat, n1);
  const edgeW = qrot(iQuat, e1).mul(iScale);
  return { pos, nrm, anim, info, edgeW, puv };
}
function makeFlowerMaterials(maxOpen, baseCol, tipCol) {
  const mat = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide });
  {
    const f = flowerPosition(maxOpen);
    const nW = varying(f.nrm);
    const ringFrac = varying(f.info.z), isCore = varying(f.info.w), colorMix = varying(f.anim.z);
    const puv = varying(f.puv);
    mat.positionNode = f.pos;
    mat.colorNode = Fn(() => {
      const along = puv.y;
      const petal = mix(baseCol, tipCol, smoothstep(0.0, 1.0, along.mul(0.8).add(ringFrac.mul(0.3)).add(colorMix.sub(0.5).mul(0.4))));
      const albedo = mix(petal, baseCol.mul(0.7), isCore);
      const n = normalize(nW).mul(faceDirection);
      const lit = toon(albedo, n, positionWorld, U.petalTranslucency);
      const edge = min(min(puv.x, float(1.0).sub(puv.x)), float(1.0).sub(along));
      const rim = float(1.0).sub(smoothstep(0.0, U.petalRim, edge)).mul(float(1.0).sub(isCore));
      return vec4(mix(lit, U.inkColor, rim.mul(0.7)), 1.0);
    })();
  }
  const outline = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide });
  {
    const f = flowerPosition(maxOpen);
    outline.positionNode = sheetOutline(f.pos, normalize(f.nrm), f.edgeW, U.outlineWidth);
    outline.colorNode = vec4(inkMist(), 1.0);
  }
  return { mat, outline };
}
const roseMats = makeFlowerMaterials(U.maxOpenA, U.roseBase, U.roseTip);
const dahliaMats = makeFlowerMaterials(U.maxOpenB, U.dahliaBase, U.dahliaTip);

// Leaf: curled tight, easing open as the stem passes its attachment point.
function leafPosition() {
  const iPS = attribute('iPos', 'vec4'), iQuat = attribute('iQuat', 'vec4');
  const iPos = iPS.xyz, iScale = iPS.w;
  const unfold = attribute('iUnfold', 'float');
  const v = uv().y;
  const curlAng = float(1.0).sub(unfold).mul(U.leafCurl).mul(v);
  const axis = vec3(1.0, 0.0, 0.0);
  const p1 = rodrigues(positionGeometry, axis, curlAng);
  const n1 = rodrigues(normalGeometry, axis, curlAng);
  const edgeDir = vec3(uv().x.mul(2.0).sub(1.0), max(v.sub(0.75), 0.0).mul(4.0), 0.0);
  const e1 = rodrigues(edgeDir, axis, curlAng);
  const pos = iPos.add(qrot(iQuat, p1.mul(iScale)));
  return { pos, nrm: qrot(iQuat, n1), edgeW: qrot(iQuat, e1).mul(iScale) };
}
const leafMat = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide });
{
  const l = leafPosition();
  const nW = varying(l.nrm);
  leafMat.positionNode = l.pos;
  leafMat.colorNode = Fn(() => {
    const n = normalize(nW).mul(faceDirection);
    const albedo = U.leafColor.mul(mix(0.8, 1.05, uv().y));
    const lit = toon(albedo, n, positionWorld, 0.5);
    const vein = float(1.0).sub(smoothstep(0.0, 0.06, abs(uv().x.sub(0.5)))).mul(0.35);
    const edge = float(1.0).sub(smoothstep(0.0, 0.1, min(min(uv().x, float(1.0).sub(uv().x)), float(1.0).sub(uv().y))));
    return vec4(mix(lit, U.inkColor, max(vein, edge.mul(0.8))), 1.0);
  })();
}
const leafOutlineMat = new THREE.MeshBasicNodeMaterial({ side: THREE.DoubleSide });
{
  const l = leafPosition();
  leafOutlineMat.positionNode = sheetOutline(l.pos, normalize(l.nrm), l.edgeW, U.outlineWidth.mul(0.8));
  leafOutlineMat.colorNode = vec4(inkMist(), 1.0);
}

// Rocks: ordinary meshes, same toon.
const rockMat = new THREE.MeshBasicNodeMaterial();
rockMat.colorNode = Fn(() => {
  const strata = mx_fractal_noise_float(positionWorld.mul(vec3(1.5, 6.0, 1.5)), 3).mul(0.12);
  return vec4(toon(U.rockColor.mul(float(1.0).add(strata)), normalize(normalWorld), positionWorld, 0.0), 1.0);
})();
const rockOutlineMat = new THREE.MeshBasicNodeMaterial({ side: THREE.BackSide });
rockOutlineMat.positionNode = positionGeometry.add(normalGeometry.mul(U.outlineWidth.mul(1.6)));
rockOutlineMat.colorNode = vec4(inkMist(), 1.0);

/* ------------------------------------------------------------------ */
/*  post: silk weave overlay                                           */
/* ------------------------------------------------------------------ */
const post = new (THREE.RenderPipeline || THREE.PostProcessing)(renderer);
const scenePass = pass(scene, camera);
const sceneColor = scenePass.getTextureNode();
post.outputNode = Fn(() => {
  const aspect = screenSize.x.div(screenSize.y);
  const suv = screenUV;
  const coord = suv.mul(vec2(aspect, 1.0)).mul(U.threadCount);
  const x = coord.x.add(hash(floor(coord.y).add(3.0)).sub(0.5).mul(U.irregularity));
  const y = coord.y.add(hash(floor(coord.x).add(7.0)).sub(0.5).mul(U.irregularity));
  const warp = pow(abs(sin(x.mul(Math.PI))), U.sharpness);
  const weft = pow(abs(sin(y.mul(Math.PI))), U.sharpness);
  const checker = mod(floor(x).add(floor(y)), 2.0);
  const weave = mix(warp, weft, checker);
  const fabric = clamp(float(1.0).sub(U.silkStrength.mul(float(1.0).sub(weave))).add(U.threadTone), 0.0, 1.0);
  const stain = mx_fractal_noise_float(suv.mul(vec2(aspect, 1.0)).mul(U.blotchScale).add(vec2(5.2, 1.7)), 4).mul(0.5).add(0.5);
  const blotch = float(1.0).sub(U.blotchStrength.mul(smoothstep(0.45, 0.95, stain)));
  const d = suv.sub(0.5).mul(vec2(aspect, 1.0)).length();
  const vig = float(1.0).sub(smoothstep(0.45, 1.1, d).mul(U.vignette));
  const overlaid = sceneColor.rgb.mul(U.silkTint).mul(fabric).mul(blotch).mul(vig);
  return vec4(mix(sceneColor.rgb, overlaid, U.silkOn), 1.0);
})();

/* ------------------------------------------------------------------ */
/*  garden generation                                                  */
/* ------------------------------------------------------------------ */
const garden = new THREE.Group();
scene.add(garden);
let G = null; // live garden state
const UP = new THREE.Vector3(0, 1, 0);
const _v = new THREE.Vector3(), _w = new THREE.Vector3(), _t = new THREE.Vector3(), _x = new THREE.Vector3(), _z = new THREE.Vector3(), _m = new THREE.Matrix4(), _q = new THREE.Quaternion(), _q2 = new THREE.Quaternion();

function bezierJS(out, p0, p1, p2, p3, t) {
  const it = 1 - t;
  return out.set(0, 0, 0).addScaledVector(p0, it * it * it).addScaledVector(p1, 3 * it * it * t).addScaledVector(p2, 3 * it * t * t).addScaledVector(p3, t * t * t);
}
function bezierTanJS(out, p0, p1, p2, p3, t) {
  const it = 1 - t;
  return out.set(0, 0, 0).addScaledVector(_w.subVectors(p1, p0), 3 * it * it).addScaledVector(_w.subVectors(p2, p1), 6 * it * t).addScaledVector(_w.subVectors(p3, p2), 3 * t * t);
}

function density(x, y, seed) {
  // warp the coordinates first so the masses come out uneven, then a low-frequency
  // "bare patches" field pulls whole regions back down toward nothing.
  const wx = x + (fbm(x * 0.22 + 3.1, y * 0.22, seed + 1) - 0.5) * P.warp * 2.0;
  const wy = y + (fbm(x * 0.22, y * 0.22 + 9.7, seed + 2) - 0.5) * P.warp * 2.0;
  let d = fbm(wx * 0.45, wy * 0.45, seed, 4);
  const bare = fbm(x * 0.18 + 40, y * 0.18 - 20, seed + 3, 3);
  d -= P.barePatches * Math.max(0, (bare - 0.5) * 2.0);
  const rr = Math.hypot(x, y) / P.fieldRadius;
  d -= Math.max(0, rr - 0.7) * 1.4;
  return d;
}

function buildGarden() {
  if (G) {
    garden.clear();
    for (const m of G.meshes) m.geometry.dispose();
  }
  const rng = mulberry32(P.seed);
  const N = P.plantCount;
  const R = P.fieldRadius;
  const H = Math.max(3, Math.ceil(N / 7));

  // hearts: local clump centres, placed where the field is dense
  const hearts = [];
  let guard = 0;
  while (hearts.length < H && guard++ < 4000) {
    const x = (rng() * 2 - 1) * R, y = (rng() * 2 - 1) * R;
    if (density(x, y, P.seed) > P.densityThreshold + 0.05 + rng() * 0.2) hearts.push({ x, y, ox: rng() * 100, oy: rng() * 100 });
  }
  if (hearts.length === 0) hearts.push({ x: 0, y: 0, ox: 0, oy: 0 });

  const plants = [];
  const place = (heartIdx) => {
    // short random hops from a heart, validated against the density field
    for (let tries = 0; tries < 10; tries++) {
      const h = hearts[heartIdx];
      const a = rng() * Math.PI * 2, d = Math.sqrt(rng()) * P.clusterRadius * (0.6 + rng() * 0.8);
      const x = h.x + Math.cos(a) * d, y = h.y + Math.sin(a) * d;
      if (density(x, y, P.seed) > P.densityThreshold) return { x, y };
      heartIdx = Math.floor(rng() * hearts.length);
    }
    return null;
  };
  guard = 0;
  while (plants.length < N && guard++ < N * 12) {
    const heartIdx = Math.floor(rng() * hearts.length);
    const p = place(heartIdx);
    if (!p) continue;
    plants.push({ x: p.x, z: p.y, heart: heartIdx });
  }
  for (let s = 0; s < P.spareSeeds; s++) plants.push({ x: 0, z: 0, heart: 0, dormant: true });
  const count = plants.length;

  // local density -> head size, opening range
  for (let i = 0; i < count; i++) {
    let c = 0;
    if (plants[i].dormant) { plants[i].crowd = 0; continue; }
    for (let j = 0; j < count; j++) if (j !== i && !plants[j].dormant && (plants[i].x - plants[j].x) ** 2 + (plants[i].z - plants[j].z) ** 2 < 0.5 * 0.5) c++;
    plants[i].crowd = c;
  }

  const roseIdx = [], dahliaIdx = [];
  for (let i = 0; i < count; i++) {
    const p = plants[i];
    p.kind = rng() < P.dahliaRatio ? 'B' : 'A';
    (p.kind === 'A' ? roseIdx : dahliaIdx).push(i);
    p.slot = (p.kind === 'A' ? roseIdx : dahliaIdx).length - 1;
    p.rng = mulberry32(Math.floor(rng() * 1e9));
    p.p0 = new THREE.Vector3(0, -U.stemBury.value, 0);
    p.p1 = new THREE.Vector3(); p.p2 = new THREE.Vector3(); p.p3 = new THREE.Vector3();
    p.leafT = []; p.leafSide = [];
    p.leafBase = 0;
    seedPlant(p, true);
  }
  let leafTotal = 0;
  for (const p of plants) { p.leafBase = leafTotal; leafTotal += p.leafT.length; }

  // --- stems
  const stemGeo = makeStemTemplate();
  const iPos = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
  const iP1 = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
  const iP2 = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
  const iP3 = new THREE.InstancedBufferAttribute(new Float32Array(count * 3), 3);
  const iState = new THREE.InstancedBufferAttribute(new Float32Array(count * 4), 4).setUsage(THREE.DynamicDrawUsage);
  stemGeo.setAttribute('iPos', iPos); stemGeo.setAttribute('iP1', iP1); stemGeo.setAttribute('iP2', iP2); stemGeo.setAttribute('iP3', iP3); stemGeo.setAttribute('iState', iState);
  stemGeo.instanceCount = count;
  const stems = new THREE.Mesh(stemGeo, stemMat);
  const stemsOutline = new THREE.Mesh(stemGeo, stemOutlineMat);
  stems.castShadow = true; stems.frustumCulled = false; stemsOutline.frustumCulled = false;

  // --- heads
  const makeHeads = (kind, list, mats) => {
    const n = Math.max(1, list.length);
    const geo = makeFlower(SHAPES[kind]);
    const pos = new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4).setUsage(THREE.DynamicDrawUsage); // xyz + scale
    const quat = new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4).setUsage(THREE.DynamicDrawUsage);
    const anim = new THREE.InstancedBufferAttribute(new Float32Array(n * 4), 4).setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('iPos', pos); geo.setAttribute('iQuat', quat); geo.setAttribute('iAnim', anim);
    geo.instanceCount = list.length;
    const mesh = new THREE.Mesh(geo, mats.mat);
    const outline = new THREE.Mesh(geo, mats.outline);
    mesh.castShadow = true; mesh.frustumCulled = false; outline.frustumCulled = false;
    return { mesh, outline, pos, quat, anim, list, kind };
  };
  const roses = makeHeads('A', roseIdx, roseMats);
  const dahlias = makeHeads('B', dahliaIdx, dahliaMats);

  // --- leaves
  const leafGeo = makeLeaf();
  const ln = Math.max(1, leafTotal);
  const lPos = new THREE.InstancedBufferAttribute(new Float32Array(ln * 4), 4).setUsage(THREE.DynamicDrawUsage); // xyz + scale
  const lQuat = new THREE.InstancedBufferAttribute(new Float32Array(ln * 4), 4).setUsage(THREE.DynamicDrawUsage);
  const lUnfold = new THREE.InstancedBufferAttribute(new Float32Array(ln), 1).setUsage(THREE.DynamicDrawUsage);
  leafGeo.setAttribute('iPos', lPos); leafGeo.setAttribute('iQuat', lQuat); leafGeo.setAttribute('iUnfold', lUnfold);
  leafGeo.instanceCount = leafTotal;
  const leaves = new THREE.Mesh(leafGeo, leafMat);
  const leavesOutline = new THREE.Mesh(leafGeo, leafOutlineMat);
  leaves.castShadow = true; leaves.frustumCulled = false; leavesOutline.frustumCulled = false;

  // --- rocks in the bare patches
  const rocks = new THREE.Group();
  let placed = 0; guard = 0;
  while (placed < P.rockCount && guard++ < 2000) {
    const x = (rng() * 2 - 1) * R * 0.95, z = (rng() * 2 - 1) * R * 0.95;
    if (density(x, z, P.seed) > P.densityThreshold - 0.08 || Math.hypot(x, z) > R) continue;
    const s = 0.18 + rng() * 0.45;
    const geo = new THREE.IcosahedronGeometry(1, 2);
    const pa = geo.attributes.position;
    const rs = rng() * 100, sq = 0.5 + rng() * 0.3, sx = 0.8 + rng() * 0.4;
    for (let i = 0; i < pa.count; i++) {
      _v.fromBufferAttribute(pa, i);
      const n = fbm(_v.x * 1.1 + rs, _v.y * 1.1 + _v.z * 0.6, P.seed + 9, 3) - 0.5;
      _v.multiplyScalar(1 + n * 0.5).multiply(_t.set(sx, sq, 1));
      pa.setXYZ(i, _v.x, _v.y, _v.z);
    }
    geo.computeVertexNormals();
    const rock = new THREE.Mesh(geo, rockMat);
    rock.position.set(x, s * 0.35, z);
    rock.scale.setScalar(s); rock.rotation.y = rng() * Math.PI * 2;
    rock.castShadow = true;
    const ro = new THREE.Mesh(geo, rockOutlineMat);
    ro.position.copy(rock.position); ro.scale.copy(rock.scale); ro.rotation.copy(rock.rotation);
    rock.userData.radius = s; rocks.add(rock, ro);
    placed++;
  }

  garden.add(stems, stemsOutline, roses.mesh, roses.outline, dahlias.mesh, dahlias.outline, leaves, leavesOutline, rocks);
  G = {
    plants, hearts, count, stems, stemsOutline, iPos, iP1, iP2, iP3, iState, roses, dahlias, leaves, leavesOutline, lPos, lQuat, lUnfold, rocks,
    meshes: [stems, roses.mesh, dahlias.mesh, leaves, ...rocks.children.filter((m, i) => i % 2 === 0)],
    time: 0,
  };
  for (let i = 0; i < count; i++) writeStatic(i);
  iPos.needsUpdate = iP1.needsUpdate = iP2.needsUpdate = iP3.needsUpdate = true;
  applyOutlineVisibility();
  document.getElementById('seedLabel').textContent = String(P.seed);
  document.getElementById('plantLabel').textContent = String(count - P.spareSeeds);
}

function seedPlant(p, first) {
  const r = p.rng;
  const sizeFactor = THREE.MathUtils.clamp(1.15 - 0.045 * p.crowd, 0.55, 1.15);
  p.size = sizeFactor * (0.85 + r() * 0.3);
  p.stemLength = (0.8 + r() * 0.8) * (0.7 + 0.3 * sizeFactor);
  const az = r() * Math.PI * 2, lean = r() * 0.45;
  p.p3.set(Math.sin(lean) * Math.cos(az), Math.cos(lean), Math.sin(lean) * Math.sin(az)).multiplyScalar(p.stemLength);
  const bendA = r() * Math.PI * 2, bendM = (r() - 0.5) * 0.5 * p.stemLength;
  _v.set(Math.cos(bendA) * bendM, 0, Math.sin(bendA) * bendM);
  p.p1.copy(p.p0).lerp(p.p3, 0.25).add(_v);
  p.p2.copy(p.p0).lerp(p.p3, 0.75).add(_v);
  p.twist = r() * Math.PI * 2;
  p.tilt = r() * 0.6;
  p.openMax = THREE.MathUtils.clamp(1.0 - 0.03 * p.crowd, 0.55, 1.0) * (0.85 + r() * 0.15);
  p.colorMix = r();
  p.phase = r() * Math.PI * 2;
  p.dDelay = P.delay * (first ? r() * 2.2 : 0.3 + r() * 1.2);
  p.dGrow = P.grow * (0.7 + r() * 0.6);
  p.dKeep = P.keep * (0.6 + r() * 0.8);
  p.dDie = P.die * (0.7 + r() * 0.6);
  p.age = first ? r() * (p.dDelay + p.dGrow + p.dKeep) * 1.1 : 0;
  const nLeaves = p.leafT.length || (2 + (r() < 0.5 ? 1 : 0));
  p.leafT.length = 0; p.leafSide.length = 0;
  for (let k = 0; k < nLeaves; k++) {
    p.leafT.push(0.3 + (k / nLeaves) * 0.4 + r() * 0.08);
    p.leafSide.push(az + Math.PI * 0.5 + k * Math.PI + (r() - 0.5) * 0.8);
  }
  p.leafScale = 0.7 + r() * 0.6;
}

function writeStatic(i) {
  const p = G.plants[i];
  G.iPos.setXYZ(i, p.x, 0, p.z);
  G.iP1.setXYZ(i, p.p1.x, p.p1.y, p.p1.z);
  G.iP2.setXYZ(i, p.p2.x, p.p2.y, p.p2.z);
  G.iP3.setXYZ(i, p.p3.x, p.p3.y, p.p3.z);
}

function respawn(i) {
  const p = G.plants[i];
  // move near a nearby heart so the clump keeps its shape
  let best = p.heart, bd = Infinity;
  for (let h = 0; h < G.hearts.length; h++) {
    const d = (G.hearts[h].x - p.x) ** 2 + (G.hearts[h].y - p.z) ** 2;
    if (d < bd) { bd = d; best = h; }
  }
  const r = p.rng;
  for (let tries = 0; tries < 8; tries++) {
    const h = G.hearts[best];
    const a = r() * Math.PI * 2, d = Math.sqrt(r()) * P.clusterRadius;
    const x = h.x + Math.cos(a) * d, z = h.y + Math.sin(a) * d;
    if (density(x, z, P.seed) > P.densityThreshold) { p.x = x; p.z = z; p.heart = best; break; }
  }
  seedPlant(p, false);
  writeStatic(i);
  G.iPos.needsUpdate = G.iP1.needsUpdate = G.iP2.needsUpdate = G.iP3.needsUpdate = true;
}

const easeOutCubic = (t) => 1 - Math.pow(1 - t, 3);
const smooth = (a, b, x) => { const t = THREE.MathUtils.clamp((x - a) / (b - a), 0, 1); return t * t * (3 - 2 * t); };
const WIND = new THREE.Vector2(1, 0.35).normalize();

function updateGarden(dt, time) {
  const { plants, count, hearts } = G;
  // hearts wander slowly
  for (const h of hearts) {
    h.x += (fbm(h.ox + time * 0.02, 0, 1, 2) - 0.5) * dt * 0.05;
    h.y += (fbm(0, h.oy + time * 0.02, 2, 2) - 0.5) * dt * 0.05;
  }
  for (let i = 0; i < count; i++) {
    const p = plants[i];
    let growth = 0, open = 0, shed = 0;
    const D = p.dDelay, Gd = p.dGrow, K = p.dKeep, X = p.dDie;
    if (p.dormant) { growth = 0; }
    else if ((p.age += dt) < D) { growth = 0; }
    else if (p.age < D + Gd) { growth = easeOutCubic((p.age - D) / Gd); }
    else if (p.age < D + Gd + K) { growth = 1; open = smooth(0, 0.35, (p.age - D - Gd) / K); }
    else if (p.age < D + Gd + K + X) {
      const u = (p.age - D - Gd - K) / X;
      growth = 1 - smooth(0.55, 1, u);
      open = 1; shed = smooth(0, 0.65, u);
    } else { respawn(i); continue; }
    open *= p.openMax;
    // wind sway, shared by the stem shader and the head placement
    const g = Math.sin(time * P.windSpeed + p.phase + p.x * 0.4) + 0.35 * Math.sin(time * P.windSpeed * 2.3 + p.phase * 1.7);
    const sway = P.windStrength * g * (0.6 + 0.4 * p.stemLength);
    // brushing past: plants near the walker lean away, with a little spring back
    let tx = 0, tz = 0;
    if (!p.dormant && P.cameraMode === 'walk') {
      const dx = p.x - walker.pos.x, dz = p.z - walker.pos.z, d2 = dx * dx + dz * dz;
      if (d2 < 1.0 && d2 > 1e-6) { const d = Math.sqrt(d2), k = (1 - d) * P.brushStrength; tx = dx / d * k; tz = dz / d * k; }
    }
    p.pushX = (p.pushX || 0) + (tx - (p.pushX || 0)) * Math.min(1, dt * 5);
    p.pushZ = (p.pushZ || 0) + (tz - (p.pushZ || 0)) * Math.min(1, dt * 5);
    const swX = WIND.x * sway + p.pushX, swZ = WIND.y * sway + p.pushZ;
    G.iState.setXYZW(i, growth, swX, swZ, p.size);
    p.growth = growth;

    // head
    const set = p.kind === 'A' ? G.roses : G.dahlias;
    const t = Math.max(growth, 0.0001);
    bezierJS(_v, p.p0, p.p1, p.p2, p.p3, t);
    bezierTanJS(_t, p.p0, p.p1, p.p2, p.p3, t).normalize();
    _t.lerp(UP, 0.35).normalize();
    _x.crossVectors(_t, _z.set(Math.cos(p.twist), 0, Math.sin(p.twist))).normalize();
    _z.crossVectors(_x, _t);
    _m.makeBasis(_x, _t, _z);
    _q.setFromRotationMatrix(_m);
    set.quat.setXYZW(p.slot, _q.x, _q.y, _q.z, _q.w);
    const budScale = 0.25 + 0.75 * smooth(0.55, 1.0, growth) * (0.8 + 0.2 * open / Math.max(p.openMax, 0.01));
    p.hx = p.x + _v.x + swX * t * t; p.hy = _v.y; p.hz = p.z + _v.z + swZ * t * t;
    set.pos.setXYZW(p.slot, p.hx, p.hy, p.hz, growth > 0.001 ? p.size * 0.62 * SHAPES[p.kind].headScale * budScale : 0);
    set.anim.setXYZW(p.slot, open, shed, p.colorMix, 0);

    // leaves
    for (let k = 0; k < p.leafT.length; k++) {
      const lt = p.leafT[k], li = p.leafBase + k;
      const unfold = smooth(lt, lt + 0.18, growth);
      const vis = growth > lt ? 1 : 0;
      bezierJS(_v, p.p0, p.p1, p.p2, p.p3, lt);
      const side = p.leafSide[k];
      _t.set(Math.cos(side), 0.9, Math.sin(side)).normalize();
      _x.crossVectors(_t, UP).normalize();
      _z.crossVectors(_x, _t);
      _m.makeBasis(_x, _t, _z);
      _q.setFromRotationMatrix(_m);
      G.lPos.setXYZW(li, p.x + _v.x + swX * lt * lt, _v.y, p.z + _v.z + swZ * lt * lt, vis * p.leafScale * p.size * (0.4 + 0.6 * unfold));
      G.lQuat.setXYZW(li, _q.x, _q.y, _q.z, _q.w);
      G.lUnfold.setX(li, unfold);
    }
  }
  G.iState.needsUpdate = true;
  for (const s of [G.roses, G.dahlias]) { s.pos.needsUpdate = s.quat.needsUpdate = s.anim.needsUpdate = true; }
  G.lPos.needsUpdate = G.lQuat.needsUpdate = G.lUnfold.needsUpdate = true;
}

// Swap a head geometry in place, keeping the live per-instance buffers so the lifecycle doesn't reset.
function rebuildHeads(kind) {
  if (!G) return;
  const set = kind === 'A' ? G.roses : G.dahlias;
  const old = set.mesh.geometry;
  const geo = makeFlower(SHAPES[kind]);
  geo.setAttribute('iPos', set.pos); geo.setAttribute('iQuat', set.quat); geo.setAttribute('iAnim', set.anim);
  geo.instanceCount = old.instanceCount;
  set.mesh.geometry = geo; set.outline.geometry = geo;
  old.dispose();
}

function applyOutlineVisibility() {
  if (!G) return;
  G.stemsOutline.visible = G.roses.outline.visible = G.dahlias.outline.visible = G.leavesOutline.visible = P.outlines;
  G.rocks.children.forEach((m, i) => { if (i % 2 === 1) m.visible = P.outlines; });
}

/* ------------------------------------------------------------------ */
/*  first-person walker + interaction                                  */
/* ------------------------------------------------------------------ */
const walker = { yaw: 0, pitch: -0.04, tYaw: 0, tPitch: -0.04, pos: new THREE.Vector3(0, 1.3, P.fieldRadius + 2.5), vel: new THREE.Vector3(), keys: Object.create(null), locked: false, dragLook: false, bob: 0, eye: 1.3, touchMove: null, touchLook: null };
const overlay = document.getElementById('enter');
const crosshair = document.getElementById('crosshair');
const toast = document.getElementById('toast');
let toastTimer = 0;
function say(msg) { toast.textContent = msg; toast.classList.add('on'); clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove('on'), 1600); }
const isTouch = window.matchMedia('(pointer: coarse)').matches;

function enterWalk() {
  if (P.cameraMode !== 'walk') return;
  walker.menu = false;
  if (!isTouch && renderer.domElement.requestPointerLock) {
    const r = renderer.domElement.requestPointerLock();
    if (r && r.catch) r.catch(() => { walker.dragLook = true; setWalkUI(true); });
  } else { walker.dragLook = true; setWalkUI(true); }
}
const menuHint = document.getElementById('menuHint');
function setWalkUI(active) {
  overlay.hidden = active || walker.menu || P.cameraMode !== 'walk';
  crosshair.hidden = !active || P.cameraMode !== 'walk';
  menuHint.hidden = !(walker.menu && !active && P.cameraMode === 'walk');
  document.body.classList.toggle('walking', active && P.cameraMode === 'walk');
}
function toggleMenu() {
  if (P.cameraMode !== 'walk') { gui.show(gui._hidden); return; }
  if (walker.locked) {
    walker.menu = true;
    document.exitPointerLock();
    gui.show(); gui.open();
  } else if (walker.dragLook) {
    gui.show(gui._hidden);
  } else {
    walker.menu = false;
    enterWalk();
  }
}
document.addEventListener('pointerlockchange', () => {
  walker.locked = document.pointerLockElement === renderer.domElement;
  setWalkUI(walker.locked);
});
document.addEventListener('pointerlockerror', () => { walker.dragLook = true; setWalkUI(true); });
renderer.domElement.addEventListener('click', () => {
  if (P.cameraMode !== 'walk') return;
  if (walker.locked || walker.dragLook) { if (!walker.dragMoved) shedLookedAt(); }
  else enterWalk();
});
overlay.addEventListener('click', enterWalk);
document.addEventListener('mousemove', (e) => {
  if (P.cameraMode !== 'walk') return;
  if (walker.locked || (walker.dragLook && walker.dragging)) {
    walker.tYaw -= e.movementX * 0.0022 * P.mouseSensitivity;
    walker.tPitch = THREE.MathUtils.clamp(walker.tPitch - e.movementY * 0.0022 * P.mouseSensitivity, -1.35, 1.35);
    if (walker.dragging && (Math.abs(e.movementX) + Math.abs(e.movementY)) > 2) walker.dragMoved = true;
  }
});
renderer.domElement.addEventListener('mousedown', () => { walker.dragging = true; walker.dragMoved = false; });
window.addEventListener('mouseup', () => { walker.dragging = false; });
window.addEventListener('blur', () => { walker.keys = Object.create(null); });

// touch: left half = walk stick, right half = look
renderer.domElement.addEventListener('touchstart', (e) => {
  if (P.cameraMode !== 'walk') return;
  if (!walker.dragLook) { walker.dragLook = true; setWalkUI(true); }
  for (const t of e.changedTouches) {
    const side = t.clientX < window.innerWidth / 2 ? 'touchMove' : 'touchLook';
    if (!walker[side]) walker[side] = { id: t.identifier, x0: t.clientX, y0: t.clientY, x: t.clientX, y: t.clientY, moved: false };
  }
  e.preventDefault();
}, { passive: false });
renderer.domElement.addEventListener('touchmove', (e) => {
  for (const t of e.changedTouches) for (const side of ['touchMove', 'touchLook']) {
    const s = walker[side];
    if (s && s.id === t.identifier) {
      if (side === 'touchLook') { walker.tYaw -= (t.clientX - s.x) * 0.005; walker.tPitch = THREE.MathUtils.clamp(walker.tPitch - (t.clientY - s.y) * 0.005, -1.35, 1.35); }
      s.x = t.clientX; s.y = t.clientY; if (Math.hypot(t.clientX - s.x0, t.clientY - s.y0) > 8) s.moved = true;
    }
  }
  e.preventDefault();
}, { passive: false });
const endTouch = (e) => {
  for (const t of e.changedTouches) for (const side of ['touchMove', 'touchLook']) {
    const s = walker[side];
    if (s && s.id === t.identifier) { if (!s.moved && side === 'touchLook') shedLookedAt(); walker[side] = null; }
  }
};
renderer.domElement.addEventListener('touchend', endTouch);
renderer.domElement.addEventListener('touchcancel', endTouch);

const _fwd = new THREE.Vector3(), _right = new THREE.Vector3(), _wish = new THREE.Vector3(), _ray = new THREE.Vector3();
function updateWalker(dt) {
  const k = walker.keys;
  let ix = (k.KeyD || k.ArrowRight ? 1 : 0) - (k.KeyA || k.ArrowLeft ? 1 : 0);
  let iz = (k.KeyW || k.ArrowUp ? 1 : 0) - (k.KeyS || k.ArrowDown ? 1 : 0);
  if (walker.touchMove) {
    const s = walker.touchMove;
    ix += THREE.MathUtils.clamp((s.x - s.x0) / 60, -1, 1);
    iz += THREE.MathUtils.clamp(-(s.y - s.y0) / 60, -1, 1);
  }
  const run = k.ShiftLeft || k.ShiftRight ? 2.0 : 1.0;
  // smooth the look: the mouse drives a target, the camera eases toward it (frame-rate independent)
  const kLook = THREE.MathUtils.lerp(60, 7, P.lookSmoothing);
  const aLook = 1 - Math.exp(-dt * kLook);
  walker.yaw += (walker.tYaw - walker.yaw) * aLook;
  walker.pitch += (walker.tPitch - walker.pitch) * aLook;
  _fwd.set(-Math.sin(walker.yaw), 0, -Math.cos(walker.yaw));
  _right.set(Math.cos(walker.yaw), 0, -Math.sin(walker.yaw));
  _wish.set(0, 0, 0).addScaledVector(_fwd, iz).addScaledVector(_right, ix);
  if (_wish.lengthSq() > 1) _wish.normalize();
  _wish.multiplyScalar(P.walkSpeed * run);
  const a = 1 - Math.exp(-dt * 8);
  walker.vel.lerp(_wish, a);
  walker.pos.addScaledVector(walker.vel, dt);
  // soft edge of the world
  const limit = P.fieldRadius + 6;
  const r = Math.hypot(walker.pos.x, walker.pos.z);
  if (r > limit) { walker.pos.x *= limit / r; walker.pos.z *= limit / r; }
  // rocks push you out
  if (G) for (const rock of G.rocks.children) {
    if (!rock.userData.radius) continue;
    const rr = rock.userData.radius * 1.15 + 0.25;
    const dx = walker.pos.x - rock.position.x, dz = walker.pos.z - rock.position.z, d = Math.hypot(dx, dz);
    if (d < rr && d > 1e-4) { walker.pos.x = rock.position.x + dx / d * rr; walker.pos.z = rock.position.z + dz / d * rr; }
  }
  const speed = walker.vel.length();
  walker.bob += dt * speed * 3.2;
  camera.position.set(walker.pos.x, walker.eye + Math.sin(walker.bob) * 0.018 * Math.min(1, speed), walker.pos.z);
  camera.rotation.set(walker.pitch, walker.yaw, 0);
}

// Looking at a flower and clicking (or pressing E) sends it into its shedding phase.
function lookedAtPlant() {
  if (!G) return null;
  camera.getWorldDirection(_ray);
  let best = null, bestScore = Infinity;
  for (const p of G.plants) {
    if (p.dormant || p.growth < 0.3 || p.hx === undefined) continue;
    const dx = p.hx - camera.position.x, dy = p.hy - camera.position.y, dz = p.hz - camera.position.z;
    const t = dx * _ray.x + dy * _ray.y + dz * _ray.z;
    if (t < 0.2 || t > 5) continue;
    const px = dx - _ray.x * t, py = dy - _ray.y * t, pz = dz - _ray.z * t;
    const perp = Math.sqrt(px * px + py * py + pz * pz);
    if (perp < 0.22 + 0.06 * t && perp + t * 0.1 < bestScore) { bestScore = perp + t * 0.1; best = p; }
  }
  return best;
}
function shedLookedAt() {
  const p = lookedAtPlant();
  if (!p) return;
  const D = p.dDelay, Gd = p.dGrow, K = p.dKeep;
  if (p.age < D + Gd + K) { p.age = D + Gd + K; say(p.kind === 'A' ? 'petals let go' : 'petals let go'); }
}
// Pressing F plants a seed where you're looking on the ground.
function plantLookedAt() {
  if (!G) return;
  camera.getWorldDirection(_ray);
  if (_ray.y > -0.02) { say('look at the ground to plant'); return; }
  const t = -camera.position.y / _ray.y;
  if (t > 7) { say('too far — step closer'); return; }
  const x = camera.position.x + _ray.x * t, z = camera.position.z + _ray.z * t;
  let p = G.plants.find((q) => q.dormant);
  if (!p) { // recycle the farthest plant
    let far = -1; for (const q of G.plants) { const d = (q.x - x) ** 2 + (q.z - z) ** 2; if (d > far) { far = d; p = q; } }
  }
  const i = G.plants.indexOf(p);
  p.dormant = false; p.x = x; p.z = z; p.crowd = 0;
  if (!p.rng) { p.rng = mulberry32(Math.floor(Math.random() * 1e9)); p.p0 = new THREE.Vector3(0, -U.stemBury.value, 0); p.p1 = new THREE.Vector3(); p.p2 = new THREE.Vector3(); p.p3 = new THREE.Vector3(); p.leafT = []; p.leafSide = []; }
  if (p.kind === undefined) { p.kind = Math.random() < P.dahliaRatio ? 'B' : 'A'; }
  seedPlant(p, false);
  p.dDelay = 0.15; p.dGrow = Math.max(2.5, p.dGrow * 0.6); p.age = 0;
  writeStatic(i);
  G.iPos.needsUpdate = G.iP1.needsUpdate = G.iP2.needsUpdate = G.iP3.needsUpdate = true;
  say('planted');
}

function setCameraMode(mode) {
  P.cameraMode = mode;
  controls.enabled = mode === 'orbit';
  if (mode === 'orbit') {
    if (document.pointerLockElement) document.exitPointerLock();
    walker.dragLook = false;
    camera.position.set(3.2, 3.0, 9.6); controls.target.set(0, 0.55, 0);
    setWalkUI(false);
  } else {
    setWalkUI(walker.locked || walker.dragLook);
  }
}

/* ------------------------------------------------------------------ */
/*  GUI                                                                */
/* ------------------------------------------------------------------ */
const gui = new GUI({ title: 'Inkwash Garden' });
const colorProxy = {};
function addColor(folder, key, label) {
  colorProxy[key] = '#' + U[key].value.getHexString();
  return folder.addColor(colorProxy, key).name(label || key).onChange((v) => U[key].value.set(v));
}
function addU(folder, key, min, max, step, label) {
  return folder.add(U[key], 'value', min, max, step).name(label || key);
}
// Generation changes rebuild the garden; coalesce to one rebuild per frame so dragging a slider stays live.
let rebuildQueued = false;
function liveRebuild() {
  if (rebuildQueued) return;
  rebuildQueued = true;
  requestAnimationFrame(() => { rebuildQueued = false; buildGarden(); });
}
const gGen = gui.addFolder('Generation');
gGen.add(P, 'seed', 1, 99999, 1).listen().onChange(liveRebuild);
gGen.add(P, 'plantCount', 40, 1500, 10).onChange(liveRebuild);
gGen.add(P, 'fieldRadius', 3, 16, 0.1).onChange(liveRebuild);
gGen.add(P, 'densityThreshold', 0.2, 0.7, 0.01).onChange(liveRebuild);
gGen.add(P, 'warp', 0, 3, 0.05).onChange(liveRebuild);
gGen.add(P, 'barePatches', 0, 1.2, 0.05).onChange(liveRebuild);
gGen.add(P, 'clusterRadius', 0.2, 2, 0.05).onChange(liveRebuild);
gGen.add(P, 'dahliaRatio', 0, 1, 0.05).name('flower B share').onChange(liveRebuild);
gGen.add(P, 'rockCount', 0, 30, 1).onChange(liveRebuild);

const gLife = gui.addFolder('Lifecycle');
gLife.add(P, 'timeScale', 0, 4, 0.05);
gLife.add(P, 'delay', 0, 10, 0.1).onChange(liveRebuild);
gLife.add(P, 'grow', 1, 20, 0.5).onChange(liveRebuild);
gLife.add(P, 'keep', 1, 40, 0.5).onChange(liveRebuild);
gLife.add(P, 'die', 1, 20, 0.5).onChange(liveRebuild);
addU(gLife, 'shedStagger', 0, 0.95, 0.01);
addU(gLife, 'shedRise', 0, 3, 0.01);
addU(gLife, 'shedSpread', 0, 2, 0.01);
addU(gLife, 'riseVariance', 0, 1, 0.01);
gLife.add(P, 'windStrength', 0, 0.3, 0.005);
gLife.add(P, 'windSpeed', 0, 3, 0.05);
gLife.close();

const gPlant = gui.addFolder('Plants');
addU(gPlant, 'stemRadius', 0.005, 0.06, 0.001);
addU(gPlant, 'radiusAttenuation', 0, 1, 0.01);
addU(gPlant, 'baseFlare', 0, 3, 0.05);
addU(gPlant, 'startScale', 0, 1, 0.01);
addU(gPlant, 'leafCurl', 0, 8, 0.1);
gPlant.close();

const gFlowers = gui.addFolder('Flowers');
const headQueued = { A: false, B: false };
function liveHeads(kind) {
  if (headQueued[kind]) return;
  headQueued[kind] = true;
  requestAnimationFrame(() => { headQueued[kind] = false; rebuildHeads(kind); });
}
function addFlowerSlot(kind, label, baseKey, tipKey, maxOpenKey) {
  const f = gFlowers.addFolder(label);
  const sh = SHAPES[kind];
  const ctrls = [];
  f.add(sh, 'kind', Object.keys(FLOWER_KINDS)).name('kind').onChange((k) => {
    Object.assign(sh, FLOWER_KINDS[k]);
    U[maxOpenKey].value = sh.maxOpen;
    ctrls.forEach((c) => c.updateDisplay());
    rebuildHeads(kind);
  });
  const s = (key, min, max, step, name) => ctrls.push(f.add(sh, key, min, max, step).name(name || key).onChange(() => liveHeads(kind)));
  s('rings', 1, 6, 1); s('petals', 1, 24, 1, 'petals (inner ring)'); s('petalGrowth', 0, 1.5, 0.05, 'petals per ring +');
  s('length', 0.05, 0.6, 0.01); s('width', 0.02, 0.4, 0.005); s('curl', -0.3, 0.5, 0.01); s('cup', -0.15, 0.25, 0.005);
  s('pointed', 0, 1, 0.01); s('coreSize', 0, 0.12, 0.005); s('spread', 0, 0.08, 0.001, 'ring spread'); s('openInner', 0, 1, 0.01, 'inner opening');
  ctrls.push(f.add(sh, 'maxOpen', 0, 2.4, 0.01).name('bloom angle').onChange((v) => { U[maxOpenKey].value = v; }));
  s('headScale', 0.3, 2.5, 0.05, 'head size');
  addColor(f, baseKey, 'base colour'); addColor(f, tipKey, 'tip colour');
  return f;
}
addFlowerSlot('A', 'Flower A', 'roseBase', 'roseTip', 'maxOpenA');
addFlowerSlot('B', 'Flower B', 'dahliaBase', 'dahliaTip', 'maxOpenB').close();

const gToon = gui.addFolder('Woodblock shading');
addU(gToon, 'thresholdLow', -0.5, 1, 0.01);
addU(gToon, 'thresholdHigh', 0, 1.5, 0.01);
addU(gToon, 'colorLevels', 2, 6, 1);
addU(gToon, 'thresholdNoiseScale', 0.5, 20, 0.1);
addU(gToon, 'thresholdNoiseStrength', 0, 1, 0.01);
addColor(gToon, 'shadowTint'); addColor(gToon, 'highlightTint');
gui.add(P, 'outlines').name('outlines').onChange(applyOutlineVisibility);
addU(gToon, 'outlineWidth', 0, 0.04, 0.001);
addU(gToon, 'petalRim', 0, 0.4, 0.01);
addU(gToon, 'petalTranslucency', 0, 1, 0.01);
addColor(gToon, 'inkColor');

const gWash = gui.addFolder('Ink-wash shadow');
addU(gWash, 'washAt', 0, 1, 0.01); addU(gWash, 'washSoft', 0.01, 1, 0.01); addU(gWash, 'washBleed', 0, 1, 0.01);
addU(gWash, 'washMottle', 0, 1, 0.01); addU(gWash, 'washScale', 0.1, 5, 0.05); addU(gWash, 'washStr', 0, 1, 0.01);
addU(gWash, 'contourWobbleScale', 0.1, 8, 0.05); addU(gWash, 'contourWobble', 0, 0.5, 0.005);
addU(gWash, 'contourWidth', 0, 6, 0.05); addU(gWash, 'contourShade', 0, 1, 0.01); addU(gWash, 'contourStr', 0, 1, 0.01);
addColor(gWash, 'washColor'); addColor(gWash, 'contourColor');
addColor(gWash, 'paperColor').onChange((v) => { U.paperColor.value.set(v); });
addU(gWash, 'groundGrain', 0, 0.3, 0.005);
gWash.close();

const gSilk = gui.addFolder('Silk weave');
addU(gSilk, 'silkOn', 0, 1, 1).name('enabled');
addU(gSilk, 'threadCount', 40, 600, 1); addU(gSilk, 'irregularity', 0, 1, 0.01); addU(gSilk, 'sharpness', 0.2, 6, 0.05);
addU(gSilk, 'silkStrength', 0, 1, 0.01, 'strength'); addU(gSilk, 'threadTone', -0.3, 0.3, 0.005);
addU(gSilk, 'blotchStrength', 0, 1, 0.01); addU(gSilk, 'blotchScale', 0.3, 8, 0.05); addU(gSilk, 'vignette', 0, 1, 0.01);
addColor(gSilk, 'silkTint', 'tint');
gSilk.close();

const gPal = gui.addFolder('Palette');
const presetState = { preset: 'Sumi on silk' };
gPal.add(presetState, 'preset', Object.keys(PRESETS)).onChange(applyPreset);
for (const k of ['stemColor', 'leafColor', 'rockColor']) addColor(gPal, k);

const gView = gui.addFolder('Display & export');
gView.add(P, 'showHud').name('show info bar').onChange(applyHud);
gView.add({ rec() { toggleRecording(); } }, 'rec').name('start / stop recording (R)');
gView.add(P, 'recordSeconds', 0, 120, 1).name('auto-stop after (s, 0 = manual)');
gView.add(P, 'recordBitrate', 4, 60, 1).name('bitrate (Mbps)');
gView.close();

const gCam = gui.addFolder('Camera');
gCam.add(P, 'cameraMode', { 'walk (WASD)': 'walk', 'orbit': 'orbit' }).name('mode').onChange(setCameraMode);
gCam.add(P, 'walkSpeed', 0.5, 6, 0.1).name('walk speed');
gCam.add(P, 'lookSmoothing', 0, 1, 0.01).name('look smoothing');
gCam.add(P, 'mouseSensitivity', 0.2, 3, 0.05).name('mouse sensitivity');
gCam.add(P, 'brushStrength', 0, 1, 0.01).name('brush push');
addU(gCam, 'mistNear', 2, 30, 0.5).name('mist near'); addU(gCam, 'mistFar', 5, 60, 0.5).name('mist far');
gCam.add(P, 'autoOrbit').name('auto orbit (orbit mode)').onChange((v) => { controls.autoRotate = v; });
gCam.add({ reset() { walker.pos.set(0, 1.05, P.fieldRadius + 2.5); walker.yaw = walker.tYaw = 0; walker.pitch = walker.tPitch = -0.04; camera.position.set(3.2, 3.0, 9.6); controls.target.set(0, 0.55, 0); } }, 'reset').name('reset view');

function applyPreset(name) {
  const pr = PRESETS[name];
  for (const [k, v] of Object.entries(pr)) {
    if (typeof v === 'string') { U[k].value.set(v); colorProxy[k] = v; }
    else U[k].value = v;
  }
  gui.controllersRecursive().forEach((c) => c.updateDisplay());
}

/* ------------------------------------------------------------------ */
/*  interaction                                                        */
/* ------------------------------------------------------------------ */
function newSeed() { P.seed = Math.floor(Math.random() * 99999) + 1; buildGarden(); }
document.getElementById('btnNew').addEventListener('click', newSeed);
/* ------------------------------------------------------------------ */
/*  HUD toggle + video export                                          */
/* ------------------------------------------------------------------ */
const btnHud = document.getElementById('btnHud');
const btnHudShow = document.getElementById('btnHudShow');
function applyHud() {
  document.body.classList.toggle('hud-off', !P.showHud);
  btnHudShow.hidden = P.showHud;
  gui.controllersRecursive().forEach((c) => { if (c.property === 'showHud') c.updateDisplay(); });
}
btnHud.addEventListener('click', () => { P.showHud = false; applyHud(); });
btnHudShow.addEventListener('click', () => { P.showHud = true; applyHud(); });

const rec = { recorder: null, chunks: [], t0: 0, timer: 0, restore: null };
const recBadge = document.getElementById('recBadge');
const recTime = document.getElementById('recTime');
async function saveFile(blob, filename) {
  let d = null;
  try { if (window.claude && window.claude.use) d = await window.claude.use('downloads'); } catch (_) { d = null; }
  if (d) {
    try { await d.save({ filename, data: blob }); say('saved ' + filename); }
    catch (err) { if (err && err.code === 'declined') say('save cancelled'); else say('could not save: ' + (err && err.message || err)); }
    return;
  }
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a'); link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  say('saved ' + filename);
}
function pickMime() {
  const c = ['video/mp4;codecs=avc1.640033', 'video/mp4;codecs=avc1.64002A', 'video/mp4;codecs=avc1', 'video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  for (const m of c) if (window.MediaRecorder && MediaRecorder.isTypeSupported(m)) return m;
  return null;
}
function startRecording() {
  if (rec.recorder) return;
  const mime = pickMime();
  if (!mime) { say('recording is not supported in this browser'); return; }
  const canvas = renderer.domElement;
  // render at exactly 1920x1080 while recording; the CSS letterboxes the view
  rec.restore = { pr: renderer.getPixelRatio(), w: window.innerWidth, h: window.innerHeight };
  renderer.setPixelRatio(1);
  renderer.setSize(1920, 1080, false);
  camera.aspect = 16 / 9; camera.updateProjectionMatrix();
  document.body.classList.add('recording');
  const stream = canvas.captureStream(60);
  const recorder = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: P.recordBitrate * 1e6 });
  rec.chunks = [];
  recorder.ondataavailable = (e) => { if (e.data && e.data.size) rec.chunks.push(e.data); };
  recorder.onstop = () => {
    const ext = mime.startsWith('video/mp4') ? 'mp4' : 'webm';
    const blob = new Blob(rec.chunks, { type: mime.split(';')[0] });
    rec.chunks = [];
    saveFile(blob, `inkwash-garden-${P.seed}-1080p60.${ext}`);
  };
  recorder.start(500);
  rec.recorder = recorder; rec.t0 = performance.now();
  recBadge.hidden = false;
  rec.timer = setInterval(() => {
    const s = (performance.now() - rec.t0) / 1000;
    recTime.textContent = `${Math.floor(s / 60)}:${String(Math.floor(s % 60)).padStart(2, '0')}`;
    if (P.recordSeconds > 0 && s >= P.recordSeconds) stopRecording();
  }, 250);
  say(mime.startsWith('video/mp4') ? 'recording MP4 1920x1080 · R to stop' : 'recording WebM (MP4 unsupported here) · R to stop');
}
function stopRecording() {
  if (!rec.recorder) return;
  clearInterval(rec.timer);
  rec.recorder.stop();
  rec.recorder = null;
  recBadge.hidden = true;
  document.body.classList.remove('recording');
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight; camera.updateProjectionMatrix();
}
function toggleRecording() { rec.recorder ? stopRecording() : startRecording(); }
document.getElementById('btnRec').addEventListener('click', toggleRecording);

const btnPause = document.getElementById('btnPause');
btnPause.addEventListener('click', () => { P.paused = !P.paused; btnPause.textContent = P.paused ? 'Resume' : 'Pause'; });
window.addEventListener('keydown', (e) => {
  if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT')) return;
  walker.keys[e.code] = true;
  if (e.code === 'KeyN') newSeed();
  else if (e.code === 'KeyE') shedLookedAt();
  else if (e.code === 'KeyF') plantLookedAt();
  else if (e.code === 'Tab' || e.code === 'ControlLeft' || e.code === 'ControlRight') { if (!e.repeat) { e.preventDefault(); toggleMenu(); } }
  else if (e.code === 'KeyP') btnPause.click();
  else if (e.code === 'KeyR') toggleRecording();
  else if (e.code === 'KeyH') { P.showHud = !P.showHud; applyHud(); }
  if (['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) e.preventDefault();
});
window.addEventListener('keyup', (e) => { walker.keys[e.code] = false; });
window.addEventListener('resize', () => {
  if (rec.recorder) return; // keep the 1080p buffer while recording
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) { P.windStrength = 0.02; P.timeScale = 0.5; }
if (window.innerWidth < 700) { P.plantCount = 220; P.fieldRadius = 7; gui.close(); }
gui.close();
gui.domElement.addEventListener('pointerdown', () => { walker.keys = Object.create(null); });
if (qp.has('plants')) P.plantCount = parseInt(qp.get('plants'));
if (qp.has('seed')) P.seed = parseInt(qp.get('seed'));

/* ------------------------------------------------------------------ */
/*  main loop                                                          */
/* ------------------------------------------------------------------ */
let simTime = 0, lastT = performance.now();
async function start() {
  try {
    await renderer.init();
    const isGPU = renderer.backend.isWebGPUBackend === true;
    document.getElementById('backendName').textContent = isGPU ? 'WebGPU' : 'WebGL2 fallback';
    document.getElementById('backendDot').classList.toggle('gpu', isGPU);
    buildGarden();
    setCameraMode(P.cameraMode);
    renderer.setAnimationLoop(() => {
      const now = performance.now(); const dt = Math.min((now - lastT) / 1000, 0.1); lastT = now;
      if (!P.paused) {
        simTime += dt * P.timeScale;
        updateGarden(dt * P.timeScale, simTime);
      }
      if (P.cameraMode === 'walk') updateWalker(dt); else controls.update();
      // the shadow map follows the viewer so the ink wash never runs out
      sun.position.set(camera.position.x + 5, 8, camera.position.z + 3);
      sun.target.position.set(camera.position.x, 0, camera.position.z);
      post.render();
    });
    loading.classList.add('gone');
  } catch (err) {
    console.error(err);
    loading.innerHTML = '<div class="err">This page needs a browser with WebGPU or WebGL2 available. The renderer reported: ' + String(err && err.message || err) + '</div>';
  }
}
start();
window.__inkwash = { P, U, SHAPES, FLOWER_KINDS, rebuildHeads, buildGarden, walker, plantLookedAt, shedLookedAt, setWalkUI, startRecording, stopRecording };
