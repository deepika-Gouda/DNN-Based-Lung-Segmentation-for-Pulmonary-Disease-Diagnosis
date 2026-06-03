document.addEventListener("DOMContentLoaded", () => {
  const dropArea = document.getElementById("dropArea");
  const fileInput = document.getElementById("fileInput");
  const preview = document.getElementById("preview");
  const form = document.getElementById("uploadForm");
  const submitBtn = document.getElementById("submitBtn");
  const spinnerWrap = document.getElementById("spinnerWrap");

  if (dropArea) dropArea.addEventListener("click", () => fileInput.click());
  if (fileInput) fileInput.addEventListener("change", () => showPreview(fileInput.files[0]));

  if (dropArea) {
    dropArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropArea.style.borderColor = "rgba(52,211,153,0.35)";
      dropArea.style.background = "rgba(52,211,153,0.04)";
    });
    dropArea.addEventListener("dragleave", (e) => {
      e.preventDefault();
      dropArea.style.borderColor = "";
      dropArea.style.background = "";
    });
    dropArea.addEventListener("drop", (e) => {
      e.preventDefault();
      if (e.dataTransfer.files && e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        showPreview(fileInput.files[0]);
      }
      dropArea.style.borderColor = "";
      dropArea.style.background = "";
    });
  }

  function showPreview(file) {
    if (!file) {
      preview.innerHTML = "";
      return;
    }
    if (!file.type.startsWith("image/")) {
      preview.innerHTML = `<p style="color:#ffb4b4">Please upload an image file.</p>`;
      return;
    }
    const reader = new FileReader();
    reader.onload = (ev) => {
      preview.innerHTML = `<img src="${ev.target.result}" alt="Preview" style="max-width:100%; border-radius:8px;">`;
    };
    reader.readAsDataURL(file);
  }

  if (form) form.addEventListener("submit", () => {
    submitBtn.disabled = true;
    if (spinnerWrap) spinnerWrap.style.display = "inline-flex";
  });

  const compareRange = document.getElementById("compareRange");
  const overlayImg = document.getElementById("overlayImg");
  if (compareRange && overlayImg) {
    function updateOverlay() {
      const pct = parseInt(compareRange.value, 10);
      overlayImg.style.width = pct + "%";
    }
    compareRange.addEventListener("input", updateOverlay);
    updateOverlay();
  }

  const thumbs = document.querySelectorAll(".result-thumbs img, .thumb");
  const baseImg = document.getElementById("baseImg");
  if (thumbs && thumbs.length && baseImg) {
    thumbs.forEach((t) => {
      t.addEventListener("click", () => {
        const src = t.getAttribute("data-src");
        const type = t.getAttribute("data-type") || "overlay";
        const overlay = document.getElementById("overlayImg");
        if (!overlay) return;
        if (type === "overlay") {
          overlay.src = src;
        } else {
          baseImg.src = src;
        }
      });
    });
  }
});
