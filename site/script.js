const progress = document.querySelector(".progress i");
const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function updateProgress() {
  const scrollable = document.documentElement.scrollHeight - window.innerHeight;
  const amount = scrollable > 0 ? (window.scrollY / scrollable) * 100 : 0;
  progress.style.width = `${Math.min(100, Math.max(0, amount))}%`;
}

window.addEventListener("scroll", updateProgress, { passive: true });
window.addEventListener("resize", updateProgress);
updateProgress();

const revealItems = document.querySelectorAll(".reveal");
if (reducedMotion || !("IntersectionObserver" in window)) {
  revealItems.forEach((item) => item.classList.add("visible"));
} else {
  const observer = new IntersectionObserver(
    (entries, currentObserver) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("visible");
          currentObserver.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.12, rootMargin: "0px 0px -30px 0px" },
  );
  revealItems.forEach((item) => observer.observe(item));
}

const hero = document.querySelector(".hero");
const orb = document.querySelector(".orb-main");
if (hero && orb && !reducedMotion && window.matchMedia("(pointer: fine)").matches) {
  hero.addEventListener("pointermove", (event) => {
    const bounds = hero.getBoundingClientRect();
    const x = (event.clientX - bounds.left) / bounds.width - 0.5;
    const y = (event.clientY - bounds.top) / bounds.height - 0.5;
    orb.style.setProperty("--tilt-x", `${y * -7}deg`);
    orb.style.setProperty("--tilt-y", `${x * 9}deg`);
    orb.style.transform = `translateY(-5px) rotateX(${y * -5}deg) rotateY(${x * 8}deg)`;
  });
  hero.addEventListener("pointerleave", () => {
    orb.style.transform = "";
  });
}
