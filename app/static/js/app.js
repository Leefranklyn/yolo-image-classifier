(() => {
  const zone = document.getElementById("upload-zone");
  const input = document.getElementById("file-input");
  const frame = zone?.querySelector(".upload__frame");
  const preview = document.getElementById("preview");
  const previewImg = document.getElementById("preview-img");
  const previewMeta = document.getElementById("preview-meta");
  const analyze = document.getElementById("analyze-btn");
  const slot = document.getElementById("result-slot");

  if (!zone) return;

  let currentFile = null;

  const formatBytes = (n) => `${(n / 1024 / 1024).toFixed(1)}MB`;

  function setFile(file) {
    currentFile = file;
    previewImg.src = URL.createObjectURL(file);
    previewImg.onload = () => {
      previewMeta.textContent = `${file.name} — ${previewImg.naturalWidth}×${previewImg.naturalHeight} — ${formatBytes(file.size)}`;
      preview.hidden = false;
    };
    analyze.disabled = false;
  }

  frame?.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files[0]) setFile(input.files[0]);
  });
  ["dragenter", "dragover"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.add("is-dragover");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    zone.addEventListener(evt, (e) => {
      e.preventDefault();
      zone.classList.remove("is-dragover");
    })
  );
  zone.addEventListener("drop", (e) => {
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith("image/")) setFile(file);
  });

  analyze?.addEventListener("click", async () => {
    if (!currentFile) return;

    analyze.disabled = true;
    analyze.setAttribute("aria-busy", "true");
    slot.innerHTML = `<p class="mono result-empty">PROCESSING — INFERENCE IN PROGRESS<span class="cursor">_</span></p>`;

    try {
      const body = new FormData();
      body.append("file", currentFile);
      const res = await fetch("/predict", {
        method: "POST",
        headers: { Accept: "application/json" },
        body,
      });
      const { html, data } = await res.json();
      slot.innerHTML = html;

      // stagger bar growth
      slot.querySelectorAll("[data-bar]").forEach((bar, i) => {
        bar.style.setProperty("--i", String(i));
      });

      // paint scores into the class manifest
      if (data?.prob_table) {
        document.querySelectorAll("[data-class-score]").forEach((el) => {
          const match = data.prob_table.find(
            (cp) => cp.name === el.dataset.classScore
          );
          if (match) el.textContent = match.prob.toFixed(4);
        });
        const top = data.top_class;
        document.querySelectorAll(".manifest__row").forEach((row) => {
          const nameEl = row.querySelector(".manifest__name");
          row.classList.toggle("is-current", nameEl?.textContent.trim() === top);
        });
      }
    } catch {
      slot.innerHTML = `<p class="mono result-empty">INLET FAULT — REQUEST FAILED — E-01</p>`;
    } finally {
      analyze.disabled = false;
      analyze.removeAttribute("aria-busy");
    }
  });
})();
