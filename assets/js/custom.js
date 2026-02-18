/* ============================================================
   ARKNIGHTS: ENDFIELD — Dark Industrial Theme v2
   Hugo + Blowfish Custom JavaScript
   ============================================================ */

(function () {
  'use strict';

  const qs = (s, p) => (p || document).querySelector(s);
  const qsa = (s, p) => [...(p || document).querySelectorAll(s)];
  const ce = (tag, cls) => {
    const el = document.createElement(tag);
    if (cls) el.className = cls;
    return el;
  };

  function throttle(fn, ms) {
    let last = 0;
    return function (...args) {
      const now = performance.now();
      if (now - last >= ms) { last = now; fn.apply(this, args); }
    };
  }

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

  const statusMsgs = [
    '[ SYS ] ONLINE',
    '[ REC ] ACTIVE',
    '[ NET ] CONNECTED',
    '[ SCN ] MONITORING',
    '[ DAT ] STREAMING',
    '[ MEM ] 0x4F6B',
    '[ CPU ] NOMINAL',
  ];
  let statusIdx = 0;
  setInterval(() => {
    statusIdx = (statusIdx + 1) % statusMsgs.length;
    status.innerHTML = '<div class="ef-status-dot"></div>' + statusMsgs[statusIdx];
  }, 4000);

  // ── Data Coordinates Overlay ──────────────────────────────
  const coords = ce('div', 'ef-coords');
  document.body.appendChild(coords);

  function updateCoords() {
    const scrollY = window.scrollY | 0;
    const scrollPct = document.body.scrollHeight > window.innerHeight
      ? ((scrollY / (document.body.scrollHeight - window.innerHeight)) * 100).toFixed(1)
      : '0.0';
    const now = new Date();
    const ts = now.toTimeString().substring(0, 8);
    coords.innerHTML = `
      POS: ${scrollY}<br>
      SCR: ${scrollPct}%<br>
      UTC: ${ts}
    `;
  }
  updateCoords();
  window.addEventListener('scroll', throttle(updateCoords, 100));
  setInterval(updateCoords, 1000);

  // ── Cursor Glow ───────────────────────────────────────────
  const glow = ce('div', 'ef-cursor-glow');
  document.body.appendChild(glow);

  document.addEventListener('mousemove', throttle(e => {
    glow.style.left = e.clientX + 'px';
    glow.style.top = e.clientY + 'px';
    glow.style.opacity = '1';
  }, 16));

  document.addEventListener('mouseleave', () => {
    glow.style.opacity = '0';
  });

  // ── Reading Progress Bar (article pages) ──────────────────
  function setupReadingProgress() {
    const article = qs('main > article');
    if (!article) return;

    const bar = ce('div', 'ef-reading-progress');
    bar.style.width = '0%';
    document.body.appendChild(bar);

    window.addEventListener('scroll', throttle(() => {
      const rect = article.getBoundingClientRect();
      const articleTop = rect.top + window.scrollY;
      const articleHeight = rect.height;
      const scrolled = window.scrollY - articleTop;
      const pct = Math.max(0, Math.min(100, (scrolled / (articleHeight - window.innerHeight)) * 100));
      bar.style.width = pct + '%';
    }, 30));
  }

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

  class FlowLine {
    constructor() { this.reset(); }

    reset() {
      this.segments = [];
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

      const head = this.getPoint(end);
      ctx.beginPath();
      ctx.arc(head.x, head.y, this.isYellow ? 2 : 1.5, 0, Math.PI * 2);
      ctx.fillStyle = this.isYellow ? `rgba(255,209,0,${this.alpha + 0.2})` : `rgba(255,255,255,${this.alpha})`;
      ctx.fill();
    }
  }

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

    for (const b of geoBlocks) {
      b.update(animTime);
      b.draw(ctx);
    }

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
    const targets = qsa('article > *, .article-content > *, main section > *');
    targets.forEach((el, i) => {
      if (!el.classList.contains('ef-reveal') &&
          !el.closest('.ef-hero') &&
          el.offsetHeight > 0) {
        el.classList.add('ef-reveal');
        // Stagger cards
        if (el.matches('a[class*="card"], article.card')) {
          el.classList.add(`ef-reveal-delay-${(i % 3) + 1}`);
        }
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
        setTimeout(() => { hero.style.borderRight = 'none'; }, 2000);
      }
    };
    setTimeout(type, 800);
  }

  // ── Particle Burst on Click ───────────────────────────────
  function setupParticles() {
    if (prefersReduced) return;

    document.addEventListener('click', e => {
      const count = 8;
      for (let i = 0; i < count; i++) {
        const p = ce('div');
        const isYellow = Math.random() < 0.4;
        const size = isYellow ? '4px' : '3px';
        Object.assign(p.style, {
          position: 'fixed',
          left: e.clientX + 'px',
          top: e.clientY + 'px',
          width: size,
          height: size,
          background: isYellow ? '#FFD100' : '#888',
          pointerEvents: 'none',
          zIndex: '9999',
          transition: 'all 0.7s cubic-bezier(0.25, 0.46, 0.45, 0.94)',
          opacity: '1',
        });
        document.body.appendChild(p);

        const angle = (Math.PI * 2 * i) / count + (Math.random() - 0.5) * 0.8;
        const dist = 25 + Math.random() * 50;

        requestAnimationFrame(() => {
          p.style.transform = `translate(${Math.cos(angle) * dist}px, ${Math.sin(angle) * dist}px) scale(0)`;
          p.style.opacity = '0';
        });

        setTimeout(() => p.remove(), 800);
      }
    });
  }

  // ── Section Numbers for Article H2s ───────────────────────
  function setupSectionNumbers() {
    const h2s = qsa('article .prose h2, article section h2');
    h2s.forEach((h2, i) => {
      if (h2.querySelector('.ef-section-num')) return;
      const num = ce('span', 'ef-section-num');
      num.textContent = `[ SEC.${String(i + 1).padStart(2, '0')} ]`;
      h2.insertBefore(num, h2.firstChild);
    });
  }

  // ── Glitch Effect on Article Title ────────────────────────
  function setupGlitch() {
    const title = qs('article header h1');
    if (!title || title.classList.contains('ef-glitch')) return;
    title.classList.add('ef-glitch');
    title.setAttribute('data-text', title.textContent);
  }

  // ── Active TOC Highlighting ───────────────────────────────
  function setupTocHighlight() {
    const toc = qs('#TableOfContents');
    if (!toc) return;

    const headings = qsa('article h2[id], article h3[id]');
    if (!headings.length) return;

    const tocLinks = qsa('a', toc);
    
    window.addEventListener('scroll', throttle(() => {
      let current = '';
      for (const h of headings) {
        if (h.getBoundingClientRect().top <= 100) {
          current = h.id;
        }
      }
      tocLinks.forEach(a => {
        const li = a.parentElement;
        if (a.getAttribute('href') === '#' + current) {
          li.classList.add('ef-toc-active-li');
        } else {
          li.classList.remove('ef-toc-active-li');
        }
      });
    }, 100));
  }

  // ── Hero Section Injection (Homepage) ─────────────────────
  function setupHero() {
    const isHome = window.location.pathname === '/' ||
                   window.location.pathname === '/index.html' ||
                   document.body.classList.contains('home');

    if (!isHome) return;
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
    qsa('article hr, main hr').forEach(hr => {
      hr.classList.add('ef-circuit-h');
    });
  }

  // ── Hover Sound Effect (subtle) ───────────────────────────
  function setupHoverFeedback() {
    if (prefersReduced) return;
    
    // Add a subtle border-glow effect on card hover
    qsa('a[class*="card"]').forEach(card => {
      card.addEventListener('mouseenter', () => {
        card.style.boxShadow = '0 0 20px rgba(255, 209, 0, 0.1), inset 0 0 20px rgba(255, 209, 0, 0.03)';
      });
      card.addEventListener('mouseleave', () => {
        card.style.boxShadow = '';
      });
    });
  }

  // ── Tag Colorization ───────────────────────────────────────
  function setupTagColors() {
    const TAG_COLORS = 10;
    // Simple hash to get consistent color per tag text
    function hashStr(s) {
      let h = 0;
      for (let i = 0; i < s.length; i++) {
        h = ((h << 5) - h + s.charCodeAt(i)) | 0;
      }
      return Math.abs(h);
    }

    const tagLinks = qsa('a[href*="/tags/"]');
    tagLinks.forEach(link => {
      // Find the inner span (Blowfish puts tag text in span.rounded-md)
      const span = link.querySelector('span.rounded-md') || link.querySelector('span span') || link;
      if (span.classList.contains('ef-tag-colored')) return;
      const text = span.textContent.trim().toLowerCase();
      if (!text) return;
      const idx = hashStr(text) % TAG_COLORS;
      span.classList.add(`ef-tag-${idx}`, 'ef-tag-colored');
    });
  }

  // ── Init ──────────────────────────────────────────────────
  function init() {
    setupHero();
    setupTyping();
    setupParticles();
    setupGlitch();
    setupSectionNumbers();
    setupTocHighlight();
    setupReadingProgress();
    setupHoverFeedback();
    setupTagColors();
    addCircuitDividers();
    advanceLoad(90);

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
