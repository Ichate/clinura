/* ============================================================
   CLINURA — SITE ENGINE
   Atmosphere, scroll progress, 3D tilt, parallax, counters,
   magnetic buttons, reveal, nav. Respects reduced motion.
   ============================================================ */
(() => {
  const reduce = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const fine = matchMedia("(pointer: fine)").matches;
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const lerp = (a, b, t) => a + (b - a) * t;

  /* ---------- atmosphere ---------- */
  const aurora = document.createElement("div");
  aurora.className = "aurora";
  aurora.setAttribute("aria-hidden", "true");
  aurora.innerHTML = "<i></i><i></i><i></i>";
  document.body.prepend(aurora);

  const glow = document.createElement("div");
  glow.className = "cursor-glow";
  glow.setAttribute("aria-hidden", "true");
  document.body.appendChild(glow);
  if (fine) document.body.classList.add("pointer-fine");

  let gx = innerWidth / 2, gy = innerHeight / 2, cx = gx, cy = gy;
  if (fine && !reduce) {
    addEventListener("pointermove", (e) => { gx = e.clientX; gy = e.clientY; }, { passive: true });
    (function follow() {
      cx = lerp(cx, gx, 0.12);
      cy = lerp(cy, gy, 0.12);
      glow.style.transform = `translate3d(${cx}px,${cy}px,0)`;
      requestAnimationFrame(follow);
    })();
  }

  /* ---------- header ---------- */
  const header = document.querySelector(".site-header");
  const onHeader = () => header && header.classList.toggle("scrolled", scrollY > 24);
  addEventListener("scroll", onHeader, { passive: true });
  onHeader();

  /* ---------- scroll progress ---------- */
  const bar = document.querySelector(".scroll-progress i");
  let targetP = 0, curP = 0;
  const measure = () => {
    const max = document.documentElement.scrollHeight - innerHeight;
    targetP = max > 0 ? clamp(scrollY / max, 0, 1) : 0;
  };
  addEventListener("scroll", measure, { passive: true });
  addEventListener("resize", measure);
  measure();
  (function tickBar() {
    curP = lerp(curP, targetP, 0.18);
    if (bar) bar.style.width = (curP * 100).toFixed(2) + "%";
    requestAnimationFrame(tickBar);
  })();

  /* ---------- mobile nav ---------- */
  const menu = document.querySelector(".menu-toggle");
  const nav = document.querySelector(".site-header nav");
  if (menu && nav) {
    menu.addEventListener("click", () => {
      const open = menu.getAttribute("aria-expanded") !== "true";
      menu.setAttribute("aria-expanded", String(open));
      nav.classList.toggle("open", open);
    });
  }

  /* ---------- 3D tilt on cards ---------- */
  if (fine && !reduce) {
    document.querySelectorAll(".feature-tile, .detail-card, .principle, .wellness-card, .limitation, .signal-step, .flow-card, .care-tool, .dataflow-card").forEach((el) => {
      el.addEventListener("pointermove", (e) => {
        const r = el.getBoundingClientRect();
        const px = (e.clientX - r.left) / r.width;
        const py = (e.clientY - r.top) / r.height;
        el.style.setProperty("--mx", (px * 100).toFixed(1) + "%");
        el.style.setProperty("--my", (py * 100).toFixed(1) + "%");
        const rx = (0.5 - py) * 7;
        const ry = (px - 0.5) * 9;
        el.style.transform = `perspective(1100px) rotateX(${rx.toFixed(2)}deg) rotateY(${ry.toFixed(2)}deg) translateY(-4px)`;
      });
      el.addEventListener("pointerleave", () => { el.style.transform = ""; });
    });
  }

  /* ---------- hero dashboard 3D ---------- */
  const showcase = document.querySelector(".monitor-showcase:not(.standalone)");
  const dash = showcase && showcase.querySelector(".dashboard");
  let dRX = 2, dRY = -5, dTY = 0, tRX = 2, tRY = -5, tTY = 0;

  if (showcase && dash && fine && !reduce) {
    showcase.addEventListener("pointermove", (e) => {
      const r = showcase.getBoundingClientRect();
      const x = (e.clientX - r.left) / r.width - 0.5;
      const y = (e.clientY - r.top) / r.height - 0.5;
      tRY = -5 + x * 12;
      tRX = 2 - y * 8;
    });
    showcase.addEventListener("pointerleave", () => { tRY = -5; tRX = 2; });
  }

  /* ---------- counters ---------- */
  const counters = [...document.querySelectorAll(".intro-stat strong")];
  if (!reduce && "IntersectionObserver" in window) {
    const obs = new IntersectionObserver((entries, o) => {
      entries.forEach((en) => {
        if (!en.isIntersecting) return;
        const el = en.target;
        const raw = el.textContent.trim();
        const m = raw.match(/^(\d+)/);
        if (!m) { o.unobserve(el); return; }
        const end = parseInt(m[1], 10);
        const suffix = raw.slice(m[1].length);
        const pad = m[1].length;
        const t0 = performance.now();
        (function step(now) {
          const p = clamp((now - t0) / 1100, 0, 1);
          const eased = 1 - Math.pow(1 - p, 3);
          el.textContent = String(Math.round(end * eased)).padStart(pad, "0") + suffix;
          if (p < 1) requestAnimationFrame(step);
        })(t0);
        o.unobserve(el);
      });
    }, { threshold: 0.6 });
    counters.forEach((c) => obs.observe(c));
  }

  /* ---------- magnetic buttons ---------- */
  if (fine && !reduce) {
    document.querySelectorAll(".button, .header-button").forEach((btn) => {
      btn.addEventListener("pointermove", (e) => {
        const r = btn.getBoundingClientRect();
        const x = (e.clientX - r.left - r.width / 2) / r.width;
        const y = (e.clientY - r.top - r.height / 2) / r.height;
        btn.style.transform = `translate(${(x * 6).toFixed(2)}px,${(y * 5 - 2).toFixed(2)}px)`;
      });
      btn.addEventListener("pointerleave", () => { btn.style.transform = ""; });
    });
  }

  /* ---------- scroll loop: parallax + dashboard ---------- */
  const parallax = [...document.querySelectorAll("[data-parallax]")];
  document.querySelectorAll(".ambient, .hero-caption, .float-note, .care-pop, .privacy-stat").forEach((el, i) => {
    if (!el.hasAttribute("data-parallax")) el.setAttribute("data-parallax", String(0.05 + i * 0.02));
  });
  const parallaxAll = [...document.querySelectorAll("[data-parallax]")];

  if (!reduce) {
    (function frame() {
      const vh = innerHeight;
      for (const el of parallaxAll) {
        const speed = parseFloat(el.getAttribute("data-parallax")) || 0.06;
        const r = el.getBoundingClientRect();
        const mid = r.top + r.height / 2 - vh / 2;
        el.style.transform = `translate3d(0,${(-mid * speed).toFixed(2)}px,0)`;
      }
      if (dash) {
        const r = showcase.getBoundingClientRect();
        const progress = clamp((vh - r.top) / (vh + r.height), 0, 1);
        tTY = (progress - 0.5) * -26;
        dRX = lerp(dRX, tRX, 0.08);
        dRY = lerp(dRY, tRY, 0.08);
        dTY = lerp(dTY, tTY, 0.08);
        dash.style.transform = `rotateY(${dRY.toFixed(2)}deg) rotateX(${dRX.toFixed(2)}deg) translateY(${dTY.toFixed(2)}px)`;
      }
      requestAnimationFrame(frame);
    })();
  }

  /* ---------- reveal ---------- */
  const reveals = [...document.querySelectorAll(".reveal")];
  if (reduce || !("IntersectionObserver" in window)) {
    reveals.forEach((el) => el.classList.add("visible"));
  } else {
    const obs = new IntersectionObserver((entries, o) => {
      entries.forEach((en) => {
        if (!en.isIntersecting) return;
        en.target.classList.add("visible");
        o.unobserve(en.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -40px 0px" });
    reveals.forEach((el) => obs.observe(el));
  }
})();
