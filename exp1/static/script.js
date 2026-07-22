// ---------------------------------------------------------
// Eco-Sorter frontend logic
// ---------------------------------------------------------
const form = document.getElementById('upload-form');
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file-input');
const dzEmpty = document.getElementById('dz-empty');
const dzPreview = document.getElementById('dz-preview');
const previewImg = document.getElementById('preview-img');

const classifyBtn = document.getElementById('classify-btn');
const resetBtn = document.getElementById('reset-btn');
const btnLabel = classifyBtn.querySelector('.btn-label');
const spinner = classifyBtn.querySelector('.spinner');

const errorBox = document.getElementById('error-box');
const resultCard = document.getElementById('result-card');
const resultMaterial = document.getElementById('result-material');
const resultEmoji = document.getElementById('result-emoji');
const confidenceValue = document.getElementById('confidence-value');
const confidenceFill = document.getElementById('confidence-fill');
const resultBinLabel = document.getElementById('result-bin-label');

const bins = Array.from(document.querySelectorAll('.bin'));

// Emoji shown in the result card per detected class.
const CLASS_EMOJI = {
  battery: '🔋', biological: '🍎', cardboard: '📦', clothes: '👕',
  glass: '🍾', metal: '🥫', paper: '📄', plastic: '🧴',
  shoes: '👟', trash: '🗑️',
};

let selectedFile = null;

// ---------- file selection ----------
function handleFiles(files) {
  if (!files || !files.length) return;
  const file = files[0];

  if (!file.type.startsWith('image/')) {
    showError('That does not look like an image. Please choose a PNG, JPG or WEBP file.');
    return;
  }

  selectedFile = file;
  clearError();
  resetResult();

  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    dzEmpty.hidden = true;
    dzPreview.hidden = false;
  };
  reader.readAsDataURL(file);

  classifyBtn.disabled = false;
  resetBtn.hidden = false;
}

fileInput.addEventListener('change', (e) => handleFiles(e.target.files));

// ---------- drag & drop ----------
['dragenter', 'dragover'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  });
});
['dragleave', 'drop'].forEach((evt) => {
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('dragover');
  });
});
dropzone.addEventListener('drop', (e) => {
  const dt = e.dataTransfer;
  if (dt && dt.files.length) {
    fileInput.files = dt.files; // keep input in sync
    handleFiles(dt.files);
  }
});

// keyboard accessibility: Enter/Space triggers the file picker
dropzone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

// ---------- submit ----------
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  if (!selectedFile) {
    showError('Please choose an image first.');
    return;
  }

  setLoading(true);
  clearError();
  resetResult();

  const data = new FormData();
  data.append('image', selectedFile);

  try {
    const res = await fetch('/predict', { method: 'POST', body: data });
    const payload = await res.json();

    if (!res.ok) {
      throw new Error(payload.error || `Server error (${res.status}).`);
    }
    showResult(payload);
  } catch (err) {
    showError(err.message || 'Something went wrong. Please try again.');
  } finally {
    setLoading(false);
  }
});

// ---------- reset ----------
resetBtn.addEventListener('click', () => {
  selectedFile = null;
  fileInput.value = '';
  previewImg.src = '';
  dzEmpty.hidden = false;
  dzPreview.hidden = true;
  classifyBtn.disabled = true;
  resetBtn.hidden = true;
  clearError();
  resetResult();
});

// ---------- UI helpers ----------
function setLoading(isLoading) {
  if (isLoading) {
    classifyBtn.disabled = true;
    btnLabel.textContent = 'Analyzing…';
    spinner.hidden = false;
  } else {
    classifyBtn.disabled = !selectedFile;
    btnLabel.textContent = 'Classify Waste';
    spinner.hidden = true;
  }
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}
function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = '';
}

function resetResult() {
  resultCard.hidden = true;
  confidenceFill.style.width = '0%';
  bins.forEach((b) => b.classList.remove('active', 'dimmed'));
}

function showResult(payload) {
  const { predicted_class, confidence, bin_id, bin_label } = payload;

  // result card
  resultMaterial.textContent = predicted_class;
  resultEmoji.textContent = CLASS_EMOJI[predicted_class] || '🧠';
  confidenceValue.textContent = `${confidence}%`;
  resultBinLabel.textContent = bin_label;
  resultCard.hidden = false;

  // animate the confidence bar on the next frame so the transition fires
  requestAnimationFrame(() => {
    confidenceFill.style.width = `${confidence}%`;
  });

  // highlight the winning bin, dim the others
  bins.forEach((b) => {
    if (b.dataset.bin === bin_id) {
      b.classList.add('active');
      b.classList.remove('dimmed');
    } else {
      b.classList.add('dimmed');
      b.classList.remove('active');
    }
  });

  // bring the bins into view so the interaction is visible
  document.getElementById('bins').scrollIntoView({ behavior: 'smooth', block: 'center' });
}
