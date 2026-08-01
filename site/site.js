/* Shared site script.
 *
 * RENAME POINT: the benchmark name lives in the BENCH constant below.
 * Every ".bench-name" element and the BibTeX block are filled from it,
 * so renaming the benchmark means editing this object plus the single
 * <title> tag at the top of each HTML file.
 */
const BENCH = {
  name: "℞-bench",      // display name (℞-bench)
  slug: "rx-bench",          // machine-friendly name (repo, BibTeX key)
  year: "2026",
  tagline: "A Benchmark for Medical AI Voice Agents",
  repoUrl: "https://github.com/sohan-shingade/rx-bench" // TODO: real repo URL
};

document.addEventListener("DOMContentLoaded", () => {
  // Inject the benchmark name everywhere it appears.
  document.querySelectorAll(".bench-name").forEach((el) => {
    el.textContent = BENCH.name;
  });

  // Point all GitHub links at the repo.
  document.querySelectorAll("a[data-repo-link]").forEach((a) => {
    a.href = BENCH.repoUrl;
  });

  // Render the BibTeX block, if present on this page.
  const bibtexEl = document.getElementById("bibtex");
  if (bibtexEl) {
    bibtexEl.textContent = [
      `@misc{${BENCH.slug.replace(/-/g, "")}${BENCH.year},`,
      `  title        = {{${BENCH.name}}: ${BENCH.tagline.replace("–", "--")}},`,
      `  author       = {{${BENCH.name} Team}},`,
      `  year         = {${BENCH.year}},`,
      `  howpublished = {\\url{${BENCH.repoUrl}}},`,
      `  note         = {Built on Sierra's \\(\\tau^2\\)-bench framework and the`,
      `                  \\(\\tau\\)-voice full-duplex stack (MIT license).}`,
      `}`
    ].join("\n");
  }

  // Copy-to-clipboard buttons (data-copy-target="<element id>").
  document.querySelectorAll("[data-copy-target]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = document.getElementById(btn.dataset.copyTarget);
      if (!target) return;
      navigator.clipboard.writeText(target.textContent).then(() => {
        const original = btn.textContent;
        btn.textContent = "Copied";
        setTimeout(() => { btn.textContent = original; }, 1500);
      });
    });
  });
});
