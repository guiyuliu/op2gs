// OP2GS interactive viewer: WebGL2 Gaussian splatting with per-Gaussian instance label and instance opacity sigma*.
// Scene files are written by script/report/make_web_scene.py (see the header comment there for the binary layout).
(() => {
  "use strict";

  const VS = `#version 300 es
precision highp float; precision highp int;
uniform highp usampler2D u_tex;
uniform mat4 projection, view;
uniform vec2 focal, viewport;
in vec2 position; in int index;
out vec4 vColor; out vec2 vPos;
void main() {
  ivec2 c0 = ivec2((uint(index) & 0x3ffu) << 1, uint(index) >> 10);
  uvec4 cen = texelFetch(u_tex, c0, 0);
  vec4 cam = view * vec4(uintBitsToFloat(cen.xyz), 1.0);
  vec4 p2 = projection * cam;
  float clip = 1.2 * p2.w;
  if (cam.z < 0.02 || p2.x < -clip || p2.x > clip || p2.y < -clip || p2.y > clip) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  uvec4 cv = texelFetch(u_tex, c0 + ivec2(1, 0), 0);
  vec2 u1 = unpackHalf2x16(cv.x), u2 = unpackHalf2x16(cv.y), u3 = unpackHalf2x16(cv.z);
  mat3 V = mat3(u1.x, u1.y, u2.x, u1.y, u2.y, u3.x, u2.x, u3.x, u3.y) * 0.25;
  mat3 J = mat3(focal.x / cam.z, 0.0, -(focal.x * cam.x) / (cam.z * cam.z),
                0.0, -focal.y / cam.z, (focal.y * cam.y) / (cam.z * cam.z),
                0.0, 0.0, 0.0);
  mat3 T = transpose(mat3(view)) * J;
  mat3 c2 = transpose(T) * V * T;
  c2[0][0] += 0.3; c2[1][1] += 0.3;
  float mid = 0.5 * (c2[0][0] + c2[1][1]);
  float rad = length(vec2(0.5 * (c2[0][0] - c2[1][1]), c2[0][1]));
  float l1 = mid + rad, l2 = mid - rad;
  if (l2 <= 0.0) { gl_Position = vec4(0.0, 0.0, 2.0, 1.0); return; }
  vec2 d = vec2(c2[0][1], l1 - c2[0][0]);
  d = dot(d, d) < 1e-12 ? (c2[0][0] >= c2[1][1] ? vec2(1.0, 0.0) : vec2(0.0, 1.0)) : normalize(d);
  vec2 ax1 = min(sqrt(l1), 1024.0) * d;
  vec2 ax2 = min(sqrt(l2), 1024.0) * vec2(d.y, -d.x);
  vColor = vec4(cv.w & 0xffu, (cv.w >> 8) & 0xffu, (cv.w >> 16) & 0xffu, (cv.w >> 24) & 0xffu) / 255.0;
  vPos = position;
  gl_Position = vec4(p2.xy / p2.w + (position.x * ax1 + position.y * ax2) * 2.0 / viewport, 0.0, 1.0);
}`;

  const FS = `#version 300 es
precision highp float;
in vec4 vColor; in vec2 vPos; out vec4 frag;
void main() {
  float r = dot(vPos, vPos);
  if (r > 9.0) discard;
  float a = min(0.99, exp(-0.5 * r) * vColor.a);
  if (a < 1.0 / 255.0) discard;
  frag = vec4(a * vColor.rgb, a);
}`;

  // depth sort in a worker: visible Gaussians only, nearest first (front-to-back blending)
  const WORKER = `
let pos = null, n = 0, vis = null;
onmessage = (e) => {
  const d = e.data;
  if (d.pos) { pos = d.pos; n = pos.length / 3; vis = null; return; }
  if (d.vis) vis = d.vis;
  if (!d.view || !pos) return;
  const v = d.view, sel = new Uint32Array(n), z = new Float32Array(n);
  let m = 0, lo = Infinity, hi = -Infinity;
  for (let i = 0; i < n; i++) {
    if (vis && !vis[i]) continue;
    const zz = v[2] * pos[3 * i] + v[6] * pos[3 * i + 1] + v[10] * pos[3 * i + 2] + v[14];
    if (zz <= 0.02) continue;
    sel[m] = i; z[m] = zz; m++;
    if (zz < lo) lo = zz; if (zz > hi) hi = zz;
  }
  const B = 65536, cnt = new Uint32Array(B), key = new Uint32Array(m);
  const s = (B - 1) / Math.max(hi - lo, 1e-6);
  for (let k = 0; k < m; k++) { key[k] = ((z[k] - lo) * s) | 0; cnt[key[k]]++; }
  for (let b = 1; b < B; b++) cnt[b] += cnt[b - 1];
  const out = new Uint32Array(m);
  for (let k = m - 1; k >= 0; k--) out[--cnt[key[k]]] = sel[k];
  postMessage({ order: out, id: d.id }, [out.buffer]);
};`;

  // ---------- small math helpers ----------
  const _f = new Float32Array(1), _i = new Int32Array(_f.buffer);
  function toHalf(x) {
    _f[0] = x; const b = _i[0], sign = (b >> 16) & 0x8000; let e = ((b >> 23) & 0xff) - 112, m = b & 0x7fffff;
    if (e <= 0) { if (e < -10) return sign; m = (m | 0x800000) >> (1 - e); return sign | ((m + 0x1000) >> 13); }
    if (e >= 31) return sign | 0x7c00;
    return sign | (e << 10) | ((m + 0x1000) >> 13);
  }
  function fromHalf(h) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * f * 5.960464477539063e-8;
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const norm = (a) => { const l = Math.hypot(a[0], a[1], a[2]) || 1; return [a[0] / l, a[1] / l, a[2] / l]; };
  function rotate(v, k, t) { // Rodrigues: rotate v around unit axis k by angle t
    const c = Math.cos(t), s = Math.sin(t), d = dot(k, v) * (1 - c);
    return [v[0] * c + (k[1] * v[2] - k[2] * v[1]) * s + k[0] * d,
            v[1] * c + (k[2] * v[0] - k[0] * v[2]) * s + k[1] * d,
            v[2] * c + (k[0] * v[1] - k[1] * v[0]) * s + k[2] * d];
  }
  function labelColor(l) { // stable, well-separated color per instance label
    const h = (l * 0.618033988749895) % 1, s = 0.55 + 0.35 * ((l * 7) % 3) / 2, v = 0.72 + 0.25 * ((l * 5) % 2);
    const i = Math.floor(h * 6), f = h * 6 - i, p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    return [[v, t, p], [q, v, p], [p, v, t], [p, q, v], [t, p, v], [v, p, q]][i % 6].map((x) => Math.round(x * 255));
  }

  // ---------- scene parsing ----------
  function parseScene(buf) {
    const hl = new Uint32Array(buf, 0, 1)[0];
    const head = JSON.parse(new TextDecoder().decode(new Uint8Array(buf, 4, hl)));
    const base = 4 + hl, n = head.n, o = head.offsets;
    const q = new Uint16Array(buf, base + o.pos, 3 * n), sh = new Uint16Array(buf, base + o.scale, 3 * n);
    const lo = head.bbox_min, ext = head.bbox_max.map((h, i) => h - lo[i]);
    const pos = new Float32Array(3 * n);
    for (let i = 0; i < 3 * n; i++) pos[i] = lo[i % 3] + (q[i] / 65535) * ext[i % 3];
    const scale = new Float32Array(3 * n);
    for (let i = 0; i < 3 * n; i++) scale[i] = fromHalf(sh[i]);
    return {
      head, n, pos, scale,
      label: new Uint16Array(buf, base + o.label, n),
      rot: new Uint8Array(buf, base + o.rot, 4 * n),
      rgba: new Uint8Array(buf, base + o.rgba, 4 * n),
      sstar: new Uint8Array(buf, base + o.sigma_star, n),
    };
  }

  // ---------- viewer ----------
  class Viewer {
    constructor(root) {
      this.root = root;
      this.canvas = root.querySelector("canvas");
      this.status = root.querySelector("[data-status]");
      this.overlay = root.querySelector("[data-overlay]");
      this.mode = "rgb"; this.tau = 0.1; this.iso = 0; this.scene = null;
      this.dirtySort = true; this.sorting = false; this.sortId = 0; this.count = 0;
      const gl = this.canvas.getContext("webgl2", { antialias: false, premultipliedAlpha: true });
      if (!gl) { this.fail("This browser does not support WebGL 2, which the viewer needs."); return; }
      this.gl = gl;
      const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
      const prog = gl.createProgram();
      gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(prog);
      if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
      gl.useProgram(prog); this.prog = prog;
      this.u = Object.fromEntries(["projection", "view", "focal", "viewport", "u_tex"].map((k) => [k, gl.getUniformLocation(prog, k)]));
      gl.disable(gl.DEPTH_TEST); gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.ONE_MINUS_DST_ALPHA, gl.ONE, gl.ONE_MINUS_DST_ALPHA, gl.ONE);
      const quad = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-3, -3, 3, -3, 3, 3, -3, 3]), gl.STATIC_DRAW);
      const aPos = gl.getAttribLocation(prog, "position"); gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
      this.idxBuf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, this.idxBuf);
      const aIdx = gl.getAttribLocation(prog, "index"); gl.enableVertexAttribArray(aIdx);
      gl.vertexAttribIPointer(aIdx, 1, gl.INT, false, 0, 0); gl.vertexAttribDivisor(aIdx, 1);
      this.tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, this.tex);
      for (const p of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, p, gl.NEAREST);
      gl.uniform1i(this.u.u_tex, 0);
      this.worker = new Worker(URL.createObjectURL(new Blob([WORKER], { type: "text/javascript" })));
      this.worker.onmessage = (e) => this.onSorted(e.data);
      this.bindInput();
      new ResizeObserver(() => { this.resize(); this.dirtySort = true; }).observe(this.canvas);
      const loop = () => { this.frame(); requestAnimationFrame(loop); };
      requestAnimationFrame(loop);
    }

    fail(msg) { this.overlay.hidden = false; this.overlay.textContent = msg; }

    async load(url) {
      this.overlay.hidden = false; this.overlay.textContent = "Loading scene…"; this.scene = null; this.count = 0;
      try {
        const res = await fetch(url);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const reader = res.body.getReader(), parts = []; let got = 0;
        for (;;) {
          const { done, value } = await reader.read(); if (done) break;
          parts.push(value); got += value.length;
          this.overlay.textContent = `Loading scene… ${(got / 1e6).toFixed(1)} MB`;
        }
        const buf = new Uint8Array(got); let off = 0; for (const p of parts) { buf.set(p, off); off += p.length; }
        this.overlay.textContent = "Preparing scene…";
        await new Promise((r) => requestAnimationFrame(() => setTimeout(r, 0)));
        this.setScene(parseScene(buf.buffer));
        this.overlay.hidden = true;
      } catch (err) {
        this.fail(`The scene could not be loaded (${err.message}). Reload the page to try again.`);
      }
    }

    setScene(s) {
      const gl = this.gl, n = s.n, W = 2048, H = Math.ceil((2 * n) / W);
      this.scene = s; this.iso = 0; this.camIdx = 0;
      const td = new Uint32Array(W * H * 4), tf = new Float32Array(td.buffer);
      const P = s.pos, S = s.scale, Q = s.rot;
      for (let i = 0; i < n; i++) {
        const o = 8 * i;
        tf[o] = P[3 * i]; tf[o + 1] = P[3 * i + 1]; tf[o + 2] = P[3 * i + 2];
        const w = (Q[4 * i] - 128) / 128, x = (Q[4 * i + 1] - 128) / 128, y = (Q[4 * i + 2] - 128) / 128, z = (Q[4 * i + 3] - 128) / 128;
        const s0 = S[3 * i], s1 = S[3 * i + 1], s2 = S[3 * i + 2];
        // M = R(q) diag(s), stored column-wise as in the 3DGS reference; Sigma = M M^T
        const m0 = (1 - 2 * (y * y + z * z)) * s0, m1 = 2 * (x * y + w * z) * s0, m2 = 2 * (x * z - w * y) * s0;
        const m3 = 2 * (x * y - w * z) * s1, m4 = (1 - 2 * (x * x + z * z)) * s1, m5 = 2 * (y * z + w * x) * s1;
        const m6 = 2 * (x * z + w * y) * s2, m7 = 2 * (y * z - w * x) * s2, m8 = (1 - 2 * (x * x + y * y)) * s2;
        td[o + 4] = toHalf(4 * (m0 * m0 + m3 * m3 + m6 * m6)) | (toHalf(4 * (m0 * m1 + m3 * m4 + m6 * m7)) << 16);
        td[o + 5] = toHalf(4 * (m0 * m2 + m3 * m5 + m6 * m8)) | (toHalf(4 * (m1 * m1 + m4 * m4 + m7 * m7)) << 16);
        td[o + 6] = toHalf(4 * (m1 * m2 + m4 * m5 + m7 * m8)) | (toHalf(4 * (m2 * m2 + m5 * m5 + m8 * m8)) << 16);
      }
      this.td = td; this.texW = W; this.texH = H;
      this.palette = new Uint32Array(s.head.n_labels + 1);
      for (let l = 1; l <= s.head.n_labels; l++) { const c = labelColor(l); this.palette[l] = c[0] | (c[1] << 8) | (c[2] << 16); }
      this.palette[0] = 140 | (140 << 8) | (140 << 16);
      this.worker.postMessage({ pos: s.pos });
      this.setCamera(0);
      this.refresh();
    }

    // colors and visibility follow the mode, the sigma* threshold and the isolated object
    refresh() {
      const s = this.scene; if (!s) return;
      const n = s.n, td = this.td, inst = this.mode === "inst", t = Math.round(this.tau * 255), vis = new Uint8Array(n);
      let shown = 0;
      for (let i = 0; i < n; i++) {
        const keep = (this.iso ? s.label[i] === this.iso && s.sstar[i] > t : inst ? s.sstar[i] > t : true);
        vis[i] = keep ? 1 : 0; shown += vis[i];
        const a = inst ? s.sstar[i] : s.rgba[4 * i + 3];
        const rgb = inst ? this.palette[s.label[i]] : s.rgba[4 * i] | (s.rgba[4 * i + 1] << 8) | (s.rgba[4 * i + 2] << 16);
        td[8 * i + 7] = (rgb | (a << 24)) >>> 0;
      }
      this.vis = vis;
      const gl = this.gl; gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32UI, this.texW, this.texH, 0, gl.RGBA_INTEGER, gl.UNSIGNED_INT, td);
      this.worker.postMessage({ vis: vis.slice() });
      this.dirtySort = true;
      this.updateStatus(shown);
    }

    updateStatus(shown) {
      const s = this.scene, fmt = (x) => x.toLocaleString("en-US");
      if (this.iso) this.status.innerHTML = `Object <b>#${this.iso}</b>: ${fmt(shown)} Gaussians with σ* &gt; ${this.tau.toFixed(2)}. <button type="button" data-showall>Show all</button>`;
      else this.status.textContent = `${fmt(shown)} of ${fmt(s.n)} Gaussians shown. Click an object to isolate it.`;
      const b = this.status.querySelector("[data-showall]"); if (b) b.onclick = () => this.isolate(0);
    }

    isolate(l) { this.iso = l; this.refresh(); }

    setCamera(k) {
      const h = this.scene.head, c = h.cameras[k % h.cameras.length];
      this.R = c.rotation.map((r) => r.slice()); this.cpos = c.position.slice(); this.camFy = c.fy / c.height;
      const f = this.col(2), d = Math.max(dot([h.center[0] - this.cpos[0], h.center[1] - this.cpos[1], h.center[2] - this.cpos[2]], f), 0.5);
      this.target = this.cpos.map((p, i) => p + f[i] * d);
      this.up = h.up; this.dirtySort = true;
    }
    col(j) { return [this.R[0][j], this.R[1][j], this.R[2][j]]; }
    setCol(j, v) { for (let i = 0; i < 3; i++) this.R[i][j] = v[i]; }

    viewMatrix() {
      const R = this.R, t = this.cpos;
      return new Float32Array([
        R[0][0], R[0][1], R[0][2], 0, R[1][0], R[1][1], R[1][2], 0, R[2][0], R[2][1], R[2][2], 0,
        -(t[0] * R[0][0] + t[1] * R[1][0] + t[2] * R[2][0]), -(t[0] * R[0][1] + t[1] * R[1][1] + t[2] * R[2][1]),
        -(t[0] * R[0][2] + t[1] * R[1][2] + t[2] * R[2][2]), 1]);
    }

    resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2), w = Math.round(this.canvas.clientWidth * dpr), h = Math.round(this.canvas.clientHeight * dpr);
      if (this.canvas.width !== w || this.canvas.height !== h) { this.canvas.width = w; this.canvas.height = h; }
    }

    frame() {
      const gl = this.gl; if (!gl || !this.scene) return;
      this.resize();
      const W = this.canvas.width, H = this.canvas.height, f = this.camFy * H, zn = 0.05, zf = 1000;
      this.view = this.viewMatrix(); this.fpx = f;
      if (this.dirtySort && !this.sorting) {
        this.sorting = true; this.dirtySort = false;
        this.worker.postMessage({ view: this.view, id: ++this.sortId });
      }
      gl.viewport(0, 0, W, H); gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT);
      if (!this.count) return;
      gl.uniformMatrix4fv(this.u.projection, false, new Float32Array([2 * f / W, 0, 0, 0, 0, -2 * f / H, 0, 0, 0, 0, zf / (zf - zn), 1, 0, 0, -(zf * zn) / (zf - zn), 0]));
      gl.uniformMatrix4fv(this.u.view, false, this.view);
      gl.uniform2fv(this.u.focal, [f, f]); gl.uniform2fv(this.u.viewport, [W, H]);
      gl.bindTexture(gl.TEXTURE_2D, this.tex);
      gl.drawArraysInstanced(gl.TRIANGLE_FAN, 0, 4, this.count);
    }

    onSorted(d) {
      this.sorting = false;
      const gl = this.gl; gl.bindBuffer(gl.ARRAY_BUFFER, this.idxBuf);
      gl.bufferData(gl.ARRAY_BUFFER, d.order, gl.DYNAMIC_DRAW); this.count = d.order.length;
    }

    // ---------- camera controls ----------
    orbit(dx, dy) {
      const tg = this.target, rel = this.cpos.map((p, i) => p - tg[i]);
      const rotAll = (axis, ang) => {
        const r2 = rotate(rel, axis, ang); for (let i = 0; i < 3; i++) rel[i] = r2[i];
        for (let j = 0; j < 3; j++) this.setCol(j, rotate(this.col(j), axis, ang));
      };
      rotAll(this.up, -dx * 0.005);
      const before = this.col(2);
      const saveR = this.R.map((r) => r.slice()), saveRel = rel.slice();
      rotAll(norm(this.col(0)), dy * 0.005);
      if (Math.abs(dot(this.col(2), this.up)) > 0.98) { this.R = saveR; for (let i = 0; i < 3; i++) rel[i] = saveRel[i]; }
      void before;
      this.cpos = tg.map((t, i) => t + rel[i]); this.dirtySort = true;
    }
    pan(dx, dy) {
      const dist = Math.hypot(...this.cpos.map((p, i) => p - this.target[i])), k = dist / (this.camFy * this.canvas.clientHeight);
      const x = this.col(0), y = this.col(1);
      for (let i = 0; i < 3; i++) { const m = -k * (dx * x[i] + dy * y[i]); this.cpos[i] += m; this.target[i] += m; }
      this.dirtySort = true;
    }
    zoom(factor) {
      const f = this.col(2), dist = Math.max(Math.hypot(...this.cpos.map((p, i) => p - this.target[i])) * factor, 0.05);
      this.cpos = this.target.map((t, i) => t - f[i] * dist); this.dirtySort = true;
    }

    pick(cx, cy) {
      const s = this.scene; if (!s) return;
      const v = this.view, W = this.canvas.clientWidth, H = this.canvas.clientHeight, f = this.camFy * H, r2 = 36;
      const inst = this.mode === "inst"; let best = -1, bz = Infinity;
      for (let i = 0; i < s.n; i++) {
        if (!this.vis[i]) continue;
        if ((inst ? s.sstar[i] : s.rgba[4 * i + 3]) < 76) continue;
        const x = s.pos[3 * i], y = s.pos[3 * i + 1], z = s.pos[3 * i + 2];
        const zc = v[2] * x + v[6] * y + v[10] * z + v[14]; if (zc < 0.05 || zc > bz) continue;
        const u = (f * (v[0] * x + v[4] * y + v[8] * z + v[12])) / zc + W / 2 - cx;
        const w = (f * (v[1] * x + v[5] * y + v[9] * z + v[13])) / zc + H / 2 - cy;
        if (u * u + w * w < r2) { best = i; bz = zc; }
      }
      if (best >= 0 && s.label[best] > 0) this.isolate(s.label[best]);
    }

    bindInput() {
      const c = this.canvas, ptrs = new Map(); let moved = 0, pinch = 0;
      c.addEventListener("contextmenu", (e) => e.preventDefault());
      c.addEventListener("pointerdown", (e) => { c.setPointerCapture(e.pointerId); ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY, b: e.button, sh: e.shiftKey }); moved = 0; pinch = 0; });
      c.addEventListener("pointermove", (e) => {
        const p = ptrs.get(e.pointerId); if (!p) return;
        const dx = e.clientX - p.x, dy = e.clientY - p.y; moved += Math.abs(dx) + Math.abs(dy);
        if (ptrs.size === 2) {
          const [a, b] = [...ptrs.values()], d0 = Math.hypot(a.x - b.x, a.y - b.y);
          p.x = e.clientX; p.y = e.clientY;
          const d1 = Math.hypot(a.x - b.x, a.y - b.y);
          if (pinch && d0 > 0) this.zoom(d0 / d1); pinch = 1; this.pan(dx / 2, dy / 2); return;
        }
        p.x = e.clientX; p.y = e.clientY;
        if (p.b === 2 || p.b === 1 || p.sh) this.pan(dx, dy); else this.orbit(dx, dy);
      });
      const up = (e) => {
        const p = ptrs.get(e.pointerId); ptrs.delete(e.pointerId);
        if (p && moved < 5 && ptrs.size === 0 && p.b === 0) { const r = c.getBoundingClientRect(); this.pick(e.clientX - r.left, e.clientY - r.top); }
      };
      c.addEventListener("pointerup", up); c.addEventListener("pointercancel", (e) => ptrs.delete(e.pointerId));
      c.addEventListener("wheel", (e) => { e.preventDefault(); this.zoom(Math.exp(e.deltaY * 0.0012)); }, { passive: false });
      window.addEventListener("keydown", (e) => { if (e.key === "Escape" && this.iso) this.isolate(0); });
    }
  }

  // ---------- page wiring ----------
  window.addEventListener("DOMContentLoaded", () => {
    const root = document.getElementById("viewer"); if (!root) return;
    let v;
    try { v = new Viewer(root); } catch (err) { root.querySelector("[data-overlay]").textContent = `The viewer could not start (${err.message}).`; return; }
    if (!v.gl) return;
    const sceneBtns = [...root.querySelectorAll("[data-scene]")], modeBtns = [...root.querySelectorAll("[data-mode]")];
    const press = (list, b) => list.forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    sceneBtns.forEach((b) => (b.onclick = () => { press(sceneBtns, b); v.load(b.dataset.scene); }));
    modeBtns.forEach((b) => (b.onclick = () => { press(modeBtns, b); v.mode = b.dataset.mode; v.refresh(); }));
    const tau = root.querySelector("#tau"), tauOut = root.querySelector("#tau-value");
    tau.oninput = () => { v.tau = +tau.value; tauOut.textContent = (+tau.value).toFixed(2); v.refresh(); };
    root.querySelector("[data-nextview]").onclick = () => { if (v.scene) { v.camIdx = (v.camIdx + 1) % v.scene.head.cameras.length; v.setCamera(v.camIdx); } };
    press(sceneBtns, sceneBtns[0]); press(modeBtns, modeBtns.find((b) => b.dataset.mode === v.mode));
    v.overlay.textContent = "The scene loads when you scroll here.";
    let started = false;
    const start = () => { if (!started) { started = true; v.load(sceneBtns[0].dataset.scene); } };
    if ("IntersectionObserver" in window) {
      const io = new IntersectionObserver((es) => { if (es.some((e) => e.isIntersecting)) { io.disconnect(); start(); } }, { rootMargin: "300px" });
      io.observe(root);
    } else start();
  });
})();
