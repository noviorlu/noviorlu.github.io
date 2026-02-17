/* ============================================================
   ARKNIGHTS: ENDFIELD — Dark Industrial Theme
   Hugo + Blowfish Custom JavaScript
   ============================================================ */

(function () {
  'use strict';

  // ── Utilities ─────────────────────────────────────────────
  const qs = (s, p) => (p || document).querySelector(s);
  const qsa = (s, p) => [...(p || document).querySelectorAll(s)];
  const ce = (tag, cls) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    return el;
  };

  // Throttle helper
  function throttle(fn, ms) {
    let last = 0;
    return function (...args) {
      const now = performance.now();
      if (now - last >= ms) { last = now; fn.apply(this, args); }
    };
  }

  // Detect reduced motion preference
  const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  // ── Loading Bar ───────────────────────────────────────────
  const loadBar = ce('div', 'ef-loading-bar');
  document.body.appendChild(loadBar);
  let loadProgress = 10;
  loadBar.style.width = loadProgress + '%';

  function advanceLoad(target) {
    const step = () => {
      if (loadProgress < target) {
        loadProgress += Math.random() * 8 + 2;
        if (loadProgress > target) loadProgress = target;
        loadBar.style.width = loadProgress + '%';
        if (loadProgress < target) requestAnimationFrame(step);
      }
    };
    requestAnimationFrame(step);
  }

  advanceLoad(60);

  window.addEventListener('load', () => {
    advanceLoad(100);
    setTimeout(() => {
      loadBar.style.opacity = '0';
      setTimeout(() => loadBar.remove(), 400);
    }, 500);
  });

  // ── Scan Lines + Scan Bar ─────────────────────────────────
  document.body.appendChild(ce('div', 'ef-scanlines'));
  if (!prefersReduced) {
    document.body.appendChild(ce('div', 'ef-scan-bar'));
  }

  // ── Corner Brackets ───────────────────────────────────────
  ['tl', 'tr', 'bl', 'br'].forEach(pos => {
    document.body.appendChild(ce('div', `ef-corner ef-corner--${pos}`));
  });

  // ── Status Indicator ──────────────────────────────────────
  const status = ce('div', 'ef-status');
  status.innerHTML = '<div class="ef-status-dot"></div>[ SYS ] ONLINE';
  document.body.appendChild(status);

  // Rotate status messages
  const statusMsgs = [
    '[ SYS ] ONLINE',
    '[ REC ] ACTIVE',
    '[ NET ] CONNECTED',
    '[ SCN ] MONITORING',
    '[ DAT ] STREAMING',
  ];
  let statusIdx = 0;
  setInterval(() => {
    statusIdx = (statusIdx + 1) % statusMsgs.length;
    status.innerHTML = '<div class="ef-status-dot"></div>' + statusMsgs[statusIdx];
  }, 5000);

  // ── Cursor Glow ───────────────────────────────────────────
  const glow = ce('div', 'ef-cursor-glow');
  document.body.appendChild(glow);

  let mouseX = -300, mouseY = -300;
  document.addEventListener('mousemove', throttle(e => {
    mouseX = e.clientX;
    mouseY = e.clientY;
    glow.style.left = mouseX + 'px';
    glow.style.top = mouseY + 'px';
    glow.style.opacity = '1';
  }, 16));

  document.addEventListener('mouseleave', () => {
    glow.style.opacity = '0';
  });

  // ── Flow Lines Canvas ─────────────────────────────────────
  const canvas = ce('canvas');
  canvas.id = 'ef-flow-canvas';
  document.body.appendChild(canvas);
  const ctx = canvas.getContext('2d');

  let W, H;
  function resizeCanvas() {
    W = canvas.width = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resizeCanvas();
  window.addEventListener('resize', throttle(resizeCanvas, 200));

  // Flow line class — traces circuit-like paths
  class FlowLine {
    constructor() { this.reset(); }

    reset() {
      this.segments = [];
      // Start from an edge
      const side = Math.random() * 4 | 0;
      let x, y;
      switch (side) {
        case 0: x = 0; y = Math.random() * H; break;
        case 1: x = W; y = Math.random() * H; break;
        case 2: x = Math.random() * W; y = 0; break;
        default: x = Math.random() * W; y = H; break;
      }

      this.x = x;
      this.y = y;
      this.segments.push({ x, y });

      // Generate path with right-angle turns (circuit traces)
      const numSegs = 3 + (Math.random() * 5 | 0);
      let horizontal = Math.random() > 0.5;

      for (let i = 0; i < numSegs; i++) {
        const len = 40 + Math.random() * 200;
        if (horizontal) {
          x += (Math.random() > 0.5 ? 1 : -1) * len;
        } else {
          y += (Math.random() > 0.5 ? 1 : -1) * len;
        }
        x = Math.max(-50, Math.min(W + 50, x));
        y = Math.max(-50, Math.min(H + 50, y));
        this.segments.push({ x, y });
        horizontal = !horizontal;
      }

      this.totalLen = 0;
      for (let i = 1; i < this.segments.length; i++) {
        const dx = this.segments[i].x - this.segments[i - 1].x;
        const dy = this.segments[i].y - this.segments[i - 1].y;
        this.totalLen += Math.sqrt(dx * dx + dy * dy);
      }

      this.progress = 0;
      this.speed = 0.003 + Math.random() * 0.006;
      this.trailLen = 0.08 + Math.random() * 0.15;
      this.isYellow = Math.random() < 0.3;
      this.alpha = 0.15 + Math.random() * 0.25;
      this.width = this.isYellow ? 1.5 : 1;
    }

    getPoint(t) {
      const targetDist = t * this.totalLen;
      let dist = 0;
      for (let i = 1; i < this.segments.length; i++) {
        const dx = this.segments[i].x - this.segments[i - 1].x;
        const dy = this.segments[i].y - this.segments[i - 1].y;
        const segLen = Math.sqrt(dx * dx + dy * dy);
        if (dist + segLen >= targetDist) {
          const frac = segLen > 0 ? (targetDist - dist) / segLen : 0;
          return {
            x: this.segments[i - 1].x + dx * frac,
            y: this.segments[i - 1].y + dy * frac,
          };
        }
        dist += segLen;
      }
      return this.segments[this.segments.length - 1];
    }

    update() {
      this.progress += this.speed;
      if (this.progress > 1 + this.trailLen) this.reset();
    }

    draw(ctx) {
      const start = Math.max(0, this.progress - this.trailLen);
      const end = Math.min(1, this.progress);
      if (start >= end) return;

      const steps = 30;
      const dt = (end - start) / steps;

      ctx.beginPath();
      const p0 = this.getPoint(start);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i <= steps; i++) {
        const p = this.getPoint(start + dt * i);
        ctx.lineTo(p.x, p.y);
      }

      const color = this.isYellow ? `rgba(255,209,0,${this.alpha})` : `rgba(255,255,255,${this.alpha * 0.5})`;
      ctx.strokeStyle = color;
      ctx.lineWidth = this.width;
      ctx.stroke();

      // Draw head dot
      const head = this.getPoint(end);
      ctx.beginPath();
      ctx.arc(head.x, head.y, this.isYellow ? 2 : 1.5, 0, Math.PI * 2);
      ctx.fillStyle = this.isYellow ? `rgba(255,209,0,${this.alpha + 0.2})` : `rgba(255,255,255,${this.alpha})`;
      ctx.fill();
    }
  }

  // Geometric blocks that pulse
  class GeoBlock {
    constructor() { this.reset(); }
    reset() {
      this.x = Math.random() * W;
      this.y = Math.random() * H;
      this.w = 80 + Math.random() * 250;
      this.h = 60 + Math.random() * 200;
      this.baseAlpha = 0.01 + Math.random() * 0.03;
      this.phase = Math.random() * Math.PI * 2;
      this.speed = 0.005 + Math.random() * 0.01;
      this.shade = Math.random() < 0.1 ? 'yellow' : 'gray';
    }
    update(t) {
      this.alpha = this.baseAlpha + Math.sin(t * this.speed + this.phase) * 0.01;
    }
    draw(ctx) {
      if (this.shade === 'yellow') {
        ctx.fillStyle = `rgba(255,209,0,${this.alpha * 0.5})`;
      } else {
        const v = 26 + Math.random() * 10 | 0;
        ctx.fillStyle = `rgba(${v},${v},${v},${this.alpha + 0.02})`;
      }
      ctx.fillRect(this.x, this.y, this.w, this.h);
    }
  }

  const lineCount = prefersReduced ? 0 : Math.min(15, Math.max(6, (W / 150) | 0));
  const flowLines = Array.from({ length: lineCount }, () => new FlowLine());

  const blockCount = prefersReduced ? 0 : 8;
  const geoBlocks = Array.from({ length: blockCount }, () => new GeoBlock());

  let animTime = 0;
  function animate() {
    animTime++;
    ctx.clearRect(0, 0, W, H);

    // Draw geo blocks
    for (const b of geoBlocks) {
      b.update(animTime);
      b.draw(ctx);
    }

    // Draw flow lines
    for (const line of flowLines) {
      line.update();
      line.draw(ctx);
    }

    requestAnimationFrame(animate);
  }

  if (!prefersReduced) {
    requestAnimationFrame(animate);
  }

  // ── Scroll-Triggered Reveal ───────────────────────────────
  function setupReveal() {
    // Add ef-reveal to article content children
    const targets = qsa('article > *, .article-content > *, main section > *');
    targets.forEach(el => {
      if (!el.classList.contains('ef-reveal') &&
          !el.closest('.ef-hero') &&
          el.offsetHeight > 0) {
        el.classList.add('ef-reveal');
      }
    });

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('ef-visible');
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

    qsa('.ef-reveal').forEach(el => observer.observe(el));
  }

  // ── Typing Effect for Homepage H1 ────────────────────────
  function setupTyping() {
    const hero = qs('.ef-hero-content h1, .hero h1, main > h1:first-of-type');
    if (!hero || hero.dataset.efTyped) return;
    hero.dataset.efTyped = '1';

    const text = hero.textContent;
    hero.textContent = '';
    hero.style.borderRight = '2px solid var(--ef-yellow)';

    let i = 0;
    const type = () => {
      if (i < text.length) {
        hero.textContent += text[i++];
        setTimeout(type, 30 + Math.random() * 60);
      } else {
        // Blink cursor then remove
        setTimeout(() => { hero.style.borderRight = 'none'; }, 2000);
      }
    };
    // Delay start
    setTimeout(type, 800);
  }

  // ── Particle Burst on Click ───────────────────────────────
  function setupParticles() {
    if (prefersReduced) return;

    document.addEventListener('click', e => {
      const count = 6;
      for (let i = 0; i < count; i++) {
        const p = ce('div');
        Object.assign(p.style, {
          position: 'fixed',
          left: e.clientX + 'px',
          top: e.clientY + 'px',
          width: '3px',
          height: '3px',
          background: Math.random() < 0.4 ? '#FFD100' : '#888',
          pointerEvents: 'none',
          zIndex: '9999',
          transition: 'all 0.6s ease-out',
          opacity: '1',
        });
        document.body.appendChild(p);

        const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.5;
        const dist = 20 + Math.random() * 40;

        requestAnimationFrame(() => {
          p.style.transform = `translate(${Math.cos(angle) * dist}px, ${Math.sin(angle) * dist}px)`;
          p.style.opacity = '0';
        });

        setTimeout(() => p.remove(), 700);
      }
    });
  }

  // ── Hero Section Injection (Homepage) ─────────────────────
  function setupHero() {
    // Only on homepage
    const isHome = window.location.pathname === '/' ||
                   window.location.pathname === '/index.html' ||
                   document.body.classList.contains('home');

    if (!isHome) return;

    // Check if hero already exists
    if (qs('.ef-hero')) return;

    const main = qs('main') || qs('#main') || qs('.main');
    if (!main) return;

    const hero = ce('div', 'ef-hero');
    hero.innerHTML = `
      <div class="ef-hero-bg"></div>
      <div class="ef-hero-content">
        <span class="ef-hero-label">[ Noviorlu喵 // Personal Log ]</span>
        <h1>RENDERING IS DEAD,<br>LONG LIVE AI</h1>
        <p>// 计算机图形学 & AI 技术笔记</p>
      </div>
    `;

    main.insertBefore(hero, main.firstChild);
  }

  // ── Circuit Dividers ──────────────────────────────────────
  function addCircuitDividers() {
    // Add circuit trace dividers between major sections
    qsa('article hr, main hr').forEach(hr => {
      hr.classList.add('ef-circuit-h');
    });
  }

  // ── Init ──────────────────────────────────────────────────
  function init() {
    setupHero();
    setupTyping();
    setupParticles();
    addCircuitDividers();
    advanceLoad(90);

    // Delay reveal setup slightly so DOM is settled
    requestAnimationFrame(() => {
      setupReveal();
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
