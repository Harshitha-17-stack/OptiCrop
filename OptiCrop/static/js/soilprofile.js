// Live "soil profile" visualization.
// As the farmer types N, P, and K readings, the three bands in the
// side panel grow/shrink proportionally, giving instant visual
// feedback on the nutrient balance of the reading before it's submitted.

(function () {
  const MAX_VALUES = { n: 140, p: 145, k: 205 };

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function updateBand(inputEl, bandEl, valueEl, maxValue) {
    if (!inputEl || !bandEl) return;
    const raw = parseFloat(inputEl.value);
    const value = isNaN(raw) ? 0 : clamp(raw, 0, maxValue);
    const pct = maxValue === 0 ? 0 : (value / maxValue) * 100;
    bandEl.style.flexGrow = String(Math.max(pct, 4));
    if (valueEl) valueEl.textContent = isNaN(raw) ? "--" : raw;
  }

  document.addEventListener("DOMContentLoaded", function () {
    const nInput = document.getElementById("nitrogen");
    const pInput = document.getElementById("phosphorous");
    const kInput = document.getElementById("potassium");

    const nBand = document.getElementById("band-n");
    const pBand = document.getElementById("band-p");
    const kBand = document.getElementById("band-k");

    const nValue = document.getElementById("band-n-value");
    const pValue = document.getElementById("band-p-value");
    const kValue = document.getElementById("band-k-value");

    function refresh() {
      updateBand(nInput, nBand, nValue, MAX_VALUES.n);
      updateBand(pInput, pBand, pValue, MAX_VALUES.p);
      updateBand(kInput, kBand, kValue, MAX_VALUES.k);
    }

    [nInput, pInput, kInput].forEach(function (el) {
      if (el) el.addEventListener("input", refresh);
    });

    refresh();
  });
})();
