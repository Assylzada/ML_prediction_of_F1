// ============================================================
// F1 ML Lab - frontend
// ============================================================

const RUN_URL = "/run";   // Flask endpoint, возвращающий результат run_ml()

const $ = (id) => document.getElementById(id);
const fmt = (n) => (n == null ? "—" : Number(n).toLocaleString("en-US"));
const num = (v, d = 4) => (v == null ? "—" : Number(v).toFixed(d));
const pct = (v) => (v == null ? "—" : (Number(v) * 100).toFixed(1) + "%");
const esc = (s) =>
    String(s).replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));

// Описание признаков для таблицы Task 4
const FEATURE_INFO = {
    grid_clean:             ["Continuous", "Starting position (pit-lane start = last slot)"],
    grid_pct:               ["Continuous", "Starting position relative to field size"],
    qualifying_position:    ["Continuous", "Qualifying result"],
    grid_vs_quali:          ["Engineered", "Grid penalty / change vs qualifying"],
    grid_minus_team_avg:    ["Engineered", "Start vs teammate (car-independent pace)"],
    driver_age:             ["Engineered", "Driver age at race date"],
    drv_prev_races:         ["Engineered", "Driver experience (previous starts)"],
    drv_podium_rate:        ["Engineered", "Smoothed career podium rate (before race)"],
    drv_form_finish5:       ["Engineered", "Average finish over last 5 races"],
    drv_podium_rate10:      ["Engineered", "Podium rate over last 10 races"],
    drv_dnf_rate10:         ["Engineered", "Retirement rate over last 10 races"],
    drv_pts_share:          ["Engineered", "Season points as share of the leader"],
    drv_champ_rank_pct:     ["Engineered", "Championship rank (percentile)"],
    drv_form_rank_pct:      ["Engineered", "Driver form rank within the race"],
    drv_circuit_starts:     ["Engineered", "Previous starts at this circuit"],
    drv_circuit_avg_finish: ["Engineered", "Average finish at this circuit"],
    con_form_finish5:       ["Engineered", "Team average finish, last 5 races"],
    con_podium_rate10:      ["Engineered", "Team podiums per car, last 10 races"],
    con_pts_share:          ["Engineered", "Team season points vs leader"],
    con_season_rank_pct:    ["Engineered", "Constructor standing (percentile)"],
    con_form_rank_pct:      ["Engineered", "Team form rank within the race"],
    pit_lane_start:         ["Binary", "Started from the pit lane"],
    front_row:              ["Binary", "Started on the front row"],
    top6_start:             ["Binary", "Started in the top 6"],
    top10_start:            ["Binary", "Started in the top 10"],
    quali_missing:          ["Binary", "Qualifying data not available"],
    grid_zone:              ["Categorical", "Grid zone (One-Hot encoded)"],
};

// ------------------------------------------------------------
// Navigation
// ------------------------------------------------------------

function goto(id) {
    document.querySelectorAll(".nav-item").forEach((b) =>
        b.classList.toggle("active", b.dataset.section === id));
    document.querySelectorAll(".page").forEach((p) => p.classList.toggle("active", p.id === id));
    window.scrollTo({ top: 0 });
}

document.querySelectorAll(".nav-item").forEach((btn) => {
    btn.addEventListener("click", () => goto(btn.dataset.section));
});

// ------------------------------------------------------------
// API + background run with live progress
// ------------------------------------------------------------

async function api(url, opts) {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.error || "HTTP " + r.status);
    return body;
}

const runButton = $("runButton");
const statusEl = $("status");
let MODELS_META = [];
let SELECTED = new Set();
let pollTimer = null;
let LAST_RESULT = null;

function setProgress(pct) {
    $("progressBar").style.width = pct + "%";
    $("labProgress").style.width = pct + "%";
}

function applyStatus(st) {
    const running = st.state === "running";
    runButton.disabled = running;
    $("labRun").disabled = running;
    statusEl.classList.toggle("running", running);
    if (running) {
        statusEl.textContent = `RUNNING · ${st.percent}% · ${st.message}`;
    } else if (st.state === "error") {
        statusEl.textContent = "ERROR: " + st.message;
    } else if (st.state === "done") {
        statusEl.textContent = "DONE · " + st.elapsed + " s";
    } else {
        statusEl.textContent = st.has_result ? "READY · RESULTS LOADED" : "READY";
    }
    setProgress(st.state === "done" ? 100 : st.percent || 0);
    $("labLog").textContent = st.log && st.log.length ? st.log.join("\n") : "Idle.";
    $("labLog").scrollTop = $("labLog").scrollHeight;
}

async function poll() {
    try {
        const st = await api("/api/status");
        applyStatus(st);
        if (st.state === "running") {
            pollTimer = setTimeout(poll, 1500);
        } else {
            pollTimer = null;
            if (st.state === "done") await loadResult();
        }
    } catch (err) {
        statusEl.textContent = "ERROR: " + err.message;
        pollTimer = setTimeout(poll, 4000);
    }
}

async function startRun() {
    const avail = MODELS_META.filter((m) => m.available).map((m) => m.name);
    const chosen = [...SELECTED].filter((n) => avail.includes(n));
    if (!chosen.length) { alert("Select at least one model in Run Settings."); goto("lab"); return; }
    const all = chosen.length === avail.length;
    try {
        await api("/api/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ models: all ? null : chosen }),
        });
        if (!pollTimer) poll();
    } catch (err) {
        statusEl.textContent = "ERROR: " + err.message;
    }
}

runButton.addEventListener("click", startRun);
$("labRun").addEventListener("click", startRun);

async function loadResult() {
    try {
        const data = await api("/api/result");
        LAST_RESULT = data;
        render(data);
    } catch (err) {
        console.warn(err.message);
    }
}

// ---------- Run Settings page ----------

function renderLab() {
    const groups = {};
    MODELS_META.forEach((m) => (groups[m.family] = groups[m.family] || []).push(m));
    $("labModels").innerHTML = Object.entries(groups).map(([fam, items]) => items.map((m) => `
        <label class="check-item ${m.available ? "" : "off"}" title="${esc(m.hint || fam)}">
            <input type="checkbox" data-model="${esc(m.name)}" ${m.available ? "" : "disabled"}
                   ${SELECTED.has(m.name) ? "checked" : ""}>
            <span>${esc(m.name)}</span>
            <small>${m.available ? (m.slow ? "SLOW · " : "") + esc(fam.toUpperCase()) : esc(m.hint)}</small>
        </label>`).join("")).join("");
}

function setSelection(fn) {
    SELECTED = new Set(MODELS_META.filter((m) => m.available && fn(m)).map((m) => m.name));
    renderLab();
}
$("labAll").addEventListener("click", () => setSelection(() => true));
$("labFast").addEventListener("click", () => setSelection((m) => !m.slow));
$("labNone").addEventListener("click", () => setSelection(() => false));
$("labModels").addEventListener("change", (e) => {
    const cb = e.target.closest("input[data-model]");
    if (!cb) return;
    cb.checked ? SELECTED.add(cb.dataset.model) : SELECTED.delete(cb.dataset.model);
});

async function initApp() {
    try {
        MODELS_META = await api("/api/models");
        SELECTED = new Set(MODELS_META.filter((m) => m.available).map((m) => m.name));
        renderLab();
    } catch (err) { console.warn(err.message); }
    renderLiterature(null);
    try {
        const st = await api("/api/status");
        applyStatus(st);
        if (st.state === "running") poll();
        else if (st.has_result) await loadResult();
    } catch (err) { statusEl.textContent = "ERROR: " + err.message; }
}

// ------------------------------------------------------------
// Render
// ------------------------------------------------------------

let PLOTS = null;   // URL графиков из ответа /run: {models: {имя: url}, summary: {ключ: url}}

function render(d) {
    PLOTS = d.plots || null;
    renderOverview(d);
    renderPreparation(d);
    renderFeatures(d);
    renderModels(d);
    renderCV(d);
    renderTuning(d);
    renderComparison(d);
    renderPredictions(d);
    renderCharts(d);
    renderRunInfo(d);
    renderPCA(d);
    renderProfiles(d);
    renderMultilevel(d);
    renderExplorer(d);
    renderPredictor(d);
    renderLiterature(d);
}

function renderOverview(d) {
    $("datasetRows").textContent = fmt(d.dataset.rows);
    $("trainRows").textContent = fmt(d.dataset.train);
    $("testRows").textContent = fmt(d.dataset.test);
    $("trainYears").textContent = "seasons " + d.dataset.train_years;
    $("testYears").textContent = "seasons " + d.dataset.test_years + " (unseen)";
    $("bestModel").textContent = d.best_model;
    $("podiumCount").textContent = fmt(d.target.podium);
    $("nonPodiumCount").textContent = fmt(d.target.non_podium);

    const tuned = d.tuning.test_metrics;
    const b = d.baseline;
    const rl = d.race_level;
    $("vsBaseline").innerHTML = `
        <tr>
            <td>${esc(b.Model)}</td>
            <td>${num(b.F1)}</td><td>${num(b.ROC_AUC)}</td><td>${num(b.PR_AUC)}</td>
            <td>${pct(rl.baseline_top3_hit_rate)}</td>
        </tr>
        <tr class="best-row">
            <td>${esc(tuned.Model)}</td>
            <td>${num(tuned.F1)}</td><td>${num(tuned.ROC_AUC)}</td><td>${num(tuned.PR_AUC)}</td>
            <td>${pct(rl.model_top3_hit_rate)}</td>
        </tr>`;
}

function renderPreparation(d) {
    const total = d.dataset.train + d.dataset.test;
    const trainPct = (d.dataset.train / total) * 100;
    $("trainBar").style.width = trainPct + "%";
    $("trainBar").textContent = `TRAIN · ${trainPct.toFixed(0)}% · ${d.dataset.train_years}`;
    $("testBar").style.width = 100 - trainPct + "%";
    $("testBar").textContent = `TEST · ${(100 - trainPct).toFixed(0)}% · ${d.dataset.test_years}`;
}

function renderFeatures(d) {
    const fs = d.feature_selection;

    $("featureTable").innerHTML = fs.used_features
        .map((f) => {
            const [type, role] = FEATURE_INFO[f] || ["—", "—"];
            return `<tr><td>${esc(f)}</td><td>${esc(type)}</td><td>${esc(role)}</td></tr>`;
        })
        .join("");

    $("droppedFeatures").innerHTML = fs.dropped_for_collinearity.length
        ? fs.dropped_for_collinearity.map((f) => `<span class="chip">${esc(f)}</span>`).join("")
        : "No features removed";

    $("targetCorr").innerHTML = barList(
        Object.entries(fs.target_correlation).map(([k, v]) => [k, v]),
        1.0
    );

    const imp = d.feature_importance.map((x) => [x.Feature, x.Importance]);
    const maxImp = Math.max(...imp.map((x) => x[1]), 0.0001);
    $("featureImportance").innerHTML = barList(imp, maxImp);
}

function barList(items, max) {
    return items
        .map(([name, value]) => {
            const width = Math.max(0, Math.min(100, (value / max) * 100));
            return `
            <div class="feature-row">
                <div class="feature-head"><span>${esc(name)}</span><strong>${num(value, 4)}</strong></div>
                <div class="bar"><div class="bar-fill" style="width:${width}%"></div></div>
            </div>`;
        })
        .join("");
}

const METRIC_KEYS = [
    ["Accuracy", "Accuracy"],
    ["Precision", "Precision"],
    ["Recall", "Recall"],
    ["F1", "F1"],
    ["ROC_AUC", "ROC-AUC"],
    ["PR_AUC", "PR-AUC"],
    ["Top3_Hit", "Top-3 per race"],
];

function metricRows(m) {
    return METRIC_KEYS.filter(([k]) => m[k] != null)
        .map(([k, label]) => `
        <div class="metric-row">
            <div class="metric-head"><span>${label}</span><strong>${num(m[k])}</strong></div>
            <div class="bar"><div class="bar-fill" style="width:${Math.min(100, m[k] * 100)}%"></div></div>
        </div>`)
        .join("");
}

function modelCard(m, extra = "") {
    const thr = m.Threshold != null ? `<small class="thr">decision threshold ${m.Threshold}</small>` : "";
    const hasChart = PLOTS && PLOTS.models && PLOTS.models[m.Model];
    const chartBtn = hasChart
        ? `<button type="button" class="chart-btn" data-model="${esc(m.Model)}">VIEW CHARTS \u2193</button>`
        : "";
    return `
    <div class="model-card ${extra}">
        <div class="card-label">MODEL</div>
        <h2>${esc(m.Model)}</h2>
        ${metricRows(m)}
        ${thr}
        ${chartBtn}
    </div>`;
}

function renderModels(d) {
    const cards = [modelCard(d.baseline, "baseline")]
        .concat(d.models.map((m) => modelCard(m, m.Model.endsWith("Tuned") ? "tuned" : "")));
    $("modelsContainer").innerHTML = cards.join("");
}

function renderCV(d) {
    $("cvTable").innerHTML = d.cross_validation
        .map((r) => {
            const folds = (r.folds || []).map((f) => `<td>${num(f)}</td>`).join("");
            const best = r.Model === d.best_model ? "best-row" : "";
            return `
            <tr class="${best}">
                <td>${esc(r.Model)}</td>
                <td>${num(r["Mean F1"])}</td>
                <td>±${num(r["Std F1"])}</td>
                <td>${num(r["Mean ROC_AUC"])} ±${num(r["Std ROC_AUC"], 3)}</td>
                <td>${num(r["Mean PR_AUC"])} ±${num(r["Std PR_AUC"], 3)}</td>
                ${folds}
            </tr>`;
        })
        .join("");
}

function renderTuning(d) {
    const t = d.tuning;
    $("tuningSubtitle").textContent =
        `GridSearchCV tunes the best tunable model from cross-validation: ${d.best_model}.`;

    $("bestParameters").textContent = JSON.stringify(t.best_parameters, null, 2);
    $("tuningCv").textContent =
        `Best CV PR-AUC: ${num(t.best_cv_pr_auc)} · CV F1: ${num(t.cv_mean_f1)} ±${num(t.cv_std_f1)}`;

    const rows = METRIC_KEYS.map(([k, label]) => {
        const before = t.before_tuning[k];
        const after = t.test_metrics[k];
        const delta = after - before;
        const cls = delta > 0.0005 ? "up" : delta < -0.0005 ? "down" : "";
        const sign = delta > 0 ? "+" : "";
        return `<tr>
            <td>${label}</td><td>${num(before)}</td><td>${num(after)}</td>
            <td class="${cls}">${sign}${num(delta)}</td></tr>`;
    });
    $("beforeAfter").innerHTML = rows.join("");

    $("tunedMetrics").innerHTML = metricRows(t.test_metrics);

    const [[tn, fp], [fn, tp]] = d.confusion_matrix;
    $("confusionMatrix").innerHTML = `
        <div class="cm-axis"></div>
        <div class="cm-axis">PRED: NO PODIUM</div>
        <div class="cm-axis">PRED: PODIUM</div>
        <div class="cm-axis">ACTUAL: NO PODIUM</div>
        <div class="cm-cell good"><strong>${fmt(tn)}</strong><span>True Negative</span></div>
        <div class="cm-cell bad"><strong>${fmt(fp)}</strong><span>False Positive</span></div>
        <div class="cm-axis">ACTUAL: PODIUM</div>
        <div class="cm-cell bad"><strong>${fmt(fn)}</strong><span>False Negative</span></div>
        <div class="cm-cell good"><strong>${fmt(tp)}</strong><span>True Positive</span></div>`;
}

function renderPredictions(d) {
    const rl = d.race_level;
    $("hitModel").textContent = pct(rl.model_top3_hit_rate);
    $("hitBaseline").textContent = pct(rl.baseline_top3_hit_rate);
    const delta = (rl.model_top3_hit_rate - rl.baseline_top3_hit_rate) * 100;
    $("hitDelta").textContent = (delta >= 0 ? "+" : "") + delta.toFixed(1) + " pp";

    $("sampleContainer").innerHTML = d.sample_predictions
        .map((race) => {
            const rows = race.top_predicted
                .map((p) => {
                    const actual = p.actual_position;
                    const hit = actual != null && actual <= 3;
                    return `
                    <tr class="${hit ? "hit" : ""}">
                        <td>${esc(p.driver)}</td>
                        <td>${esc(p.team)}</td>
                        <td>${p.grid ?? "—"}</td>
                        <td>
                            <div class="bar"><div class="bar-fill" style="width:${p.podium_probability * 100}%"></div></div>
                            ${pct(p.podium_probability)}
                        </td>
                        <td>${actual == null ? "DNF" : "P" + actual}</td>
                    </tr>`;
                })
                .join("");
            return `
            <div class="panel">
                <div class="panel-label">${esc(race.race)}</div>
                <table>
                    <thead><tr><th>DRIVER</th><th>TEAM</th><th>GRID</th><th>PODIUM PROB.</th><th>ACTUAL</th></tr></thead>
                    <tbody>${rows}</tbody>
                </table>
            </div>`;
        })
        .join("");
}

// ============================================================
// MODEL COMPARISON
// ============================================================

const MODEL_INFO = {
    "Logistic Regression": {
        idea: "Fits a weighted sum of the features and passes it through a sigmoid to get a podium probability.",
        pros: "Fast, stable on small data, hard to overfit, well-calibrated probabilities, coefficients are interpretable.",
        cons: "Only linear effects, interactions must be engineered by hand.",
        params: "C (regularization strength), class_weight",
    },
    "KNN": {
        idea: "Predicts from the podium outcomes of the most similar historical driver-race rows.",
        pros: "No training step, captures local patterns.",
        cons: "Suffers in many dimensions, sensitive to feature scaling and noisy features, slow predictions.",
        params: "n_neighbors, weights, distance metric p",
    },
    "SVM (RBF)": {
        idea: "Finds a maximum-margin boundary in a kernel-transformed space; probabilities come from Platt scaling.",
        pros: "Flexible non-linear boundary.",
        cons: "Slow, probabilities are an add-on, struggles with strongly imbalanced classes.",
        params: "C, kernel, gamma (defaults used)",
    },
    "Neural Network (MLP)": {
        idea: "Two hidden layers (32 and 16 neurons) learn non-linear combinations of features.",
        pros: "Can model complex interactions.",
        cons: "Needs far more data than about 7,000 rows, so it overfits or underfits easily, results vary with random seed.",
        params: "hidden_layer_sizes, alpha (L2), early stopping",
    },
    "Random Forest": {
        idea: "Averages hundreds of decision trees trained on random subsets of rows and features.",
        pros: "Handles non-linearity and interactions, robust to outliers, little tuning needed.",
        cons: "Probabilities are less smooth, adds little when one feature (grid) dominates.",
        params: "n_estimators, max_depth, min_samples_leaf, max_features",
    },
    "Extra Trees": {
        idea: "Like Random Forest, but split thresholds are drawn at random, which further reduces variance.",
        pros: "Faster than Random Forest, smoother decision surface.",
        cons: "Slightly higher bias, same limited gain over linear models here.",
        params: "n_estimators, max_depth, min_samples_leaf",
    },
    "AdaBoost": {
        idea: "Builds weak learners one after another, each focusing on the examples the previous ones got wrong.",
        pros: "Simple boosting baseline.",
        cons: "Sensitive to noisy labels, which are common in race outcomes (crashes, safety cars).",
        params: "n_estimators, learning_rate",
    },
    "Gradient Boosting": {
        idea: "Histogram-based gradient boosting: shallow trees added sequentially to correct the previous errors (same design as LightGBM).",
        pros: "State of the art for tabular data, captures interactions, strong regularization options.",
        cons: "Needs careful regularization on small datasets, less transparent.",
        params: "learning_rate, max_depth, min_samples_leaf, max_iter, l2_regularization",
    },
    "XGBoost": {
        idea: "Regularized gradient-boosted trees with second-order optimization.",
        pros: "Very strong on tabular data, built-in regularization.",
        cons: "Many hyperparameters, no clear advantage over scikit-learn boosting on about 7,000 rows.",
        params: "max_depth, learning_rate, n_estimators",
    },
    "LightGBM": {
        idea: "Leaf-wise gradient boosting with histogram binning.",
        pros: "Fast and accurate on tabular data.",
        cons: "Leaf-wise growth overfits small datasets unless num_leaves is kept low.",
        params: "num_leaves, learning_rate, n_estimators",
    },
    "PCA + Logistic Regression": {
        idea: "Compresses the correlated features into principal components that keep 90% of the variance, then fits Logistic Regression on them (Patil et al., 2023).",
        pros: "Removes collinearity, fast, stable.",
        cons: "Components are linear mixes that ignore the target, so podium-relevant signal can be lost.",
        params: "pca n_components, C",
    },
    "EDNN-style (2 networks)": {
        idea: "One neural network reads the start and qualifying features, another reads driver and team form; a final network combines their outputs (after Fatima & Johrendt, 2023).",
        pros: "Each network specialises on one kind of information.",
        cons: "Many parameters for a small tabular dataset, so it overfits or underfits easily.",
        params: "layer sizes, alpha (fixed)",
    },
    "Random Forest (balanced)": {
        idea: "Random Forest with class weights, so rare podium rows count more during training (Yathishkumar, 2025).",
        pros: "Better recall for the minority class.",
        cons: "Probabilities are shifted upwards, so they need calibration.",
        params: "max_depth, min_samples_leaf",
    },
    "LightGBM (balanced)": {
        idea: "LightGBM with class weights for the rare podium class (Yathishkumar, 2025).",
        pros: "Strong on tabular data, handles imbalance.",
        cons: "Shifted probabilities, overfits small data without strong regularization.",
        params: "num_leaves, learning_rate, n_estimators (fixed)",
    },
    "Pairwise Ranker": {
        idea: "Learning to rank: inside every race, pairs (podium driver, other driver) teach a logistic model which driver should be ahead. The score is turned into a podium probability with a per-race softmax.",
        pros: "Compares drivers of the same race directly, so race-level conditions cancel out.",
        cons: "Linear scores, probabilities depend on the other drivers in the race.",
        params: "C",
    },
    "Multilevel Rank-Logit": {
        idea: "Plackett-Luce model of the finishing order with random effects for the driver and for the constructor, estimated as a MAP (Bayesian) model (van Kesteren & Bergkamp, 2023).",
        pros: "Separates driver skill from car quality, models the whole race at once, shrinks small-sample effects.",
        cons: "MAP instead of full MCMC, needs the whole grid, so it cannot score a single driver.",
        params: "sigma_driver, sigma_team, l2",
    },
    "LSTM (last 5 races)": {
        idea: "Reads the driver's last 5 races (finish, grid, DNF, points, podium) with an LSTM and combines it with the race features (Surendran, 2023).",
        pros: "Learns form trends from the order of results.",
        cons: "Small data, so it rarely beats feature-engineered form averages.",
        params: "hidden size 32, early stopping",
    },
    "GRU (last 5 races)": {
        idea: "Same as the LSTM model, but with a lighter GRU cell.",
        pros: "Fewer parameters, trains faster.",
        cons: "Same data limits as other recurrent models.",
        params: "hidden size 32, early stopping",
    },
    "Bi-LSTM (last 5 races)": {
        idea: "Bidirectional LSTM over the last 5 races (best architecture in Sasikumar et al., 2025).",
        pros: "Looks at the sequence in both directions.",
        cons: "Doubles the parameters, and the backward pass has little meaning for race history.",
        params: "hidden size 32, early stopping",
    },
    "TCN (last 5 races)": {
        idea: "Temporal convolutional network: causal dilated 1D convolutions over the last 5 races.",
        pros: "Fast, parallel, stable gradients.",
        cons: "A window of 5 steps gives dilations little to work with.",
        params: "3 dilated conv layers, early stopping",
    },
    "CNN-BiLSTM (last 5 races)": {
        idea: "A 1D convolution extracts local patterns, then a Bi-LSTM reads them as a sequence.",
        pros: "Combines local pattern detection with sequence memory.",
        cons: "The most complex sequence model and the easiest to overfit.",
        params: "conv + hidden size 32",
    },
    "Stacking Ensemble": {
        idea: "A logistic-regression meta-model learns how to combine the predictions of Random Forest, Gradient Boosting and Logistic Regression.",
        pros: "Combines different model families, often the most robust.",
        cons: "Slowest to train, hard to explain, and the gain disappears when the base models make similar errors.",
        params: "base estimators, meta-learner, inner cv=3",
    },
};

const p4 = (v) => (v == null ? "\u2014" : Number(v).toFixed(4));
const pp = (v) => (v == null ? "\u2014" : (v >= 0 ? "+" : "") + (v * 100).toFixed(1) + " pp");
const ci4 = (c) => (c ? `[${Number(c[0]).toFixed(3)}, ${Number(c[1]).toFixed(3)}]` : "\u2014");
const ciPp = (c) => (c ? `[${(c[0] * 100).toFixed(1)}, ${(c[1] * 100).toFixed(1)}] pp` : "\u2014");

function verdict(c) {
    if (!c) return ["\u2014", ""];
    if (c[0] > 0) return ["Significantly better", "up"];
    if (c[1] < 0) return ["Significantly worse", "down"];
    return ["Not significant", "muted"];
}

function renderComparison(d) {
    const zoo = d.model_zoo;
    const cmp = d.comparison;
    const base = d.baseline;
    const baseTop3 = d.race_level.baseline_top3_hit_rate;
    const plain = zoo.filter((z) => !z.tuned && !z.calibrated);

    // ---------- KPI ----------
    const bestTest = [...zoo].sort((a, b) => b.test.PR_AUC - a.test.PR_AUC)[0];
    const bestHit = [...zoo].sort((a, b) => b.test.Top3_Hit - a.test.Top3_Hit)[0];
    $("cmpKpis").innerHTML = `
        <div class="kpi-card"><span>MODELS COMPARED</span><strong>${cmp.n_models}</strong>
            <small>+ 1 tuned version and 1 baseline</small></div>
        <div class="kpi-card"><span>BEST BY CROSS-VALIDATION</span><strong class="model-name">${esc(zoo[0].Model)}</strong>
            <small>PR-AUC ${p4(zoo[0].cv.PR_AUC)} \u00b1${num(zoo[0].cv.PR_AUC_std, 3)}</small></div>
        <div class="kpi-card"><span>BEST ON TEST (PR-AUC)</span><strong class="model-name">${esc(bestTest.Model)}</strong>
            <small>PR-AUC ${p4(bestTest.test.PR_AUC)} vs baseline ${p4(base.PR_AUC)}</small></div>
        <div class="kpi-card"><span>BEST TOP-3 HIT RATE</span><strong class="model-name">${esc(bestHit.Model)}</strong>
            <small>${pct(bestHit.test.Top3_Hit)} vs baseline ${pct(baseTop3)}</small></div>`;

    // ---------- Findings ----------
    const sortedCv = [...plain].sort((a, b) => b.cv.PR_AUC - a.cv.PR_AUC);
    const best = sortedCv[0];
    const worst = sortedCv[sortedCv.length - 1];
    const spread = best.cv.PR_AUC - worst.cv.PR_AUC;

    const topGroup = sortedCv.filter((z) => best.cv.PR_AUC - z.cv.PR_AUC <= best.cv.PR_AUC_std);
    const topNames = topGroup.map((z) => esc(z.Model)).join(", ");

    const byFamily = {};
    plain.forEach((z) => {
        if (!byFamily[z.Family] || z.cv.PR_AUC > byFamily[z.Family].cv.PR_AUC) byFamily[z.Family] = z;
    });
    const familyText = Object.values(byFamily)
        .sort((a, b) => b.cv.PR_AUC - a.cv.PR_AUC)
        .map((z) => `${esc(z.Family)}: ${esc(z.Model)} (${p4(z.cv.PR_AUC)})`)
        .join(" \u00b7 ");

    const ens = plain.filter((z) => z.Family.includes("ensemble"));
    const bestEns = ens.sort((a, b) => b.cv.PR_AUC - a.cv.PR_AUC)[0];
    const lr = plain.find((z) => z.Model === "Logistic Regression");
    let ensText = "";
    let ensGain = false;
    if (bestEns && lr) {
        const diff = bestEns.cv.PR_AUC - lr.cv.PR_AUC;
        ensGain = diff > 0.001;
        ensText = diff > 0.001
            ? `The best ensemble (${esc(bestEns.Model)}, ${p4(bestEns.cv.PR_AUC)}) beats the simple linear model (${p4(lr.cv.PR_AUC)}) by ${p4(diff)}.`
            : `The best ensemble (${esc(bestEns.Model)}, ${p4(bestEns.cv.PR_AUC)}) does not beat the simple Logistic Regression (${p4(lr.cv.PR_AUC)}). Extra model complexity brought no gain on this dataset.`;
    }

    const sigTop3 = zoo.filter((z) => z.ci.delta_top3[0] > 0);
    const sigPr = zoo.filter((z) => z.ci.delta_pr_auc[0] > 0);
    const sigText = sigTop3.length === 0
        ? `No model beats the grid-position baseline in Top-3 hit rate with statistical confidence: every 95% interval of the difference includes zero (${cmp.n_test_races} test races).`
        : `${sigTop3.length} of ${zoo.length} models beat the baseline in Top-3 hit rate with 95% confidence: ${sigTop3.map((z) => esc(z.Model)).join(", ")}.`;
    const sigPrText = sigPr.length === 0
        ? "None is significantly better in PR-AUC either."
        : `In PR-AUC, ${sigPr.length} of ${zoo.length} models are significantly better than the baseline.`;

    const timed = plain.filter((z) => z.fit_seconds != null);
    const fastest = [...timed].sort((a, b) => a.fit_seconds - b.fit_seconds)[0];
    const slowest = [...timed].sort((a, b) => b.fit_seconds - a.fit_seconds)[0];

    const topFeat = d.feature_importance[0];

    const findings = [
        ["1", `${esc(best.Model)} ranks first`,
            `By 5-fold cross-validated PR-AUC, ${esc(best.Model)} leads with ${p4(best.cv.PR_AUC)} \u00b1${num(best.cv.PR_AUC_std, 3)}, while the weakest model, ${esc(worst.Model)}, reaches ${p4(worst.cv.PR_AUC)}. The spread between best and worst is ${p4(spread)}.`],
        ["2", "Most models are statistically tied",
            `${topGroup.length} of ${plain.length} models (${topNames}) are within one fold standard deviation (${num(best.cv.PR_AUC_std, 3)}) of the leader, so their order should not be over-interpreted.`],
        ["3", ensGain ? "Ensembles give a small gain" : "Complexity does not pay off",
            `${ensText} Best model per family: ${familyText}.`],
        ["4", "Real but modest gains over the baseline",
            `${sigText} ${sigPrText} Starting position already explains most of the podium outcome${topFeat ? ` (the most important feature is ${esc(topFeat.Feature)})` : ""}.`],
        ["5", "Cost versus value",
            fastest && slowest
                ? `${esc(fastest.Model)} trains in ${fastest.fit_seconds}s (including cross-validation), while ${esc(slowest.Model)} needs ${slowest.fit_seconds}s. Compare this cost with the PR-AUC gap on the leaderboard before choosing a model.`
                : ""],
        ["6", "Selected model",
            `${esc(d.best_model)} is the best-ranked model that can be tuned with GridSearchCV. After tuning it reaches PR-AUC ${p4(d.tuning.test_metrics.PR_AUC)} and Top-3 hit rate ${pct(d.race_level.model_top3_hit_rate)} on the test seasons, against ${pct(baseTop3)} for the baseline.`],
    ];
    $("cmpFindings").innerHTML = findings
        .map(([n, title, text]) => `
        <div class="finding">
            <div class="finding-num">${n}</div>
            <div><strong>${title}</strong><p>${text}</p></div>
        </div>`)
        .join("");

    // ---------- Leaderboard ----------
    const colBest = {};
    const cols = {
        cvPr: (z) => z.cv.PR_AUC, cvRoc: (z) => z.cv.ROC_AUC, cvF1: (z) => z.cv.F1, cvTop: (z) => z.cv.Top3_Hit,
        tPr: (z) => z.test.PR_AUC, tRoc: (z) => z.test.ROC_AUC, tF1: (z) => z.test.F1, tTop: (z) => z.test.Top3_Hit,
    };
    Object.entries(cols).forEach(([k, fn]) => {
        colBest[k] = Math.max(...zoo.map((z) => fn(z)).filter((v) => v != null));
    });
    const cell = (k, z, text) => {
        const v = cols[k](z);
        return `<td class="${v != null && v === colBest[k] ? "up" : ""}">${text}</td>`;
    };
    const lowest = (key) => Math.min(...zoo.map((z) => z.test[key]).filter((v) => v != null));
    const bestBrier = lowest("Brier"), bestLl = lowest("LogLoss");
    const lowCell = (v, best) => `<td class="${v != null && v === best ? "up" : ""}">${p4(v)}</td>`;

    const rows = zoo.map((z) => `
        <tr class="${z.rank === 1 ? "best-row" : ""}">
            <td>${z.rank}</td>
            <td>${esc(z.Model)}${z.tuned ? ' <span class="tag">TUNED</span>' : ""}${z.calibrated ? ' <span class="tag">CALIBRATED</span>' : ""}</td>
            <td class="muted">${esc(z.Family)}</td>
            ${cell("cvPr", z, `${p4(z.cv.PR_AUC)} <span class="muted">\u00b1${num(z.cv.PR_AUC_std, 3)}</span>`)}
            ${cell("cvRoc", z, `${p4(z.cv.ROC_AUC)}`)}
            ${cell("cvF1", z, `${p4(z.cv.F1)} <span class="muted">\u00b1${num(z.cv.F1_std, 3)}</span>`)}
            ${cell("cvTop", z, pct(z.cv.Top3_Hit))}
            ${cell("tPr", z, p4(z.test.PR_AUC))}
            ${cell("tRoc", z, p4(z.test.ROC_AUC))}
            ${cell("tF1", z, p4(z.test.F1))}
            ${cell("tTop", z, pct(z.test.Top3_Hit))}
            <td class="${z.test.Top3_Hit - baseTop3 > 0 ? "up" : "down"}">${pp(z.test.Top3_Hit - baseTop3)}</td>
            ${lowCell(z.test.Brier, bestBrier)}${lowCell(z.test.LogLoss, bestLl)}
            <td class="muted">${z.fit_seconds != null ? z.fit_seconds + " s" : "\u2014"}</td>
        </tr>`);
    rows.push(`
        <tr class="baseline-row">
            <td>\u2014</td><td>${esc(base.Model)}</td><td class="muted">Rule-based</td>
            <td>\u2014</td><td>\u2014</td><td>\u2014</td><td>\u2014</td>
            <td>${p4(base.PR_AUC)}</td><td>${p4(base.ROC_AUC)}</td><td>${p4(base.F1)}</td>
            <td>${pct(baseTop3)}</td><td class="muted">0.0 pp</td>
            <td class="muted">\u2014</td><td class="muted">\u2014</td><td class="muted">\u2014</td>
        </tr>`);
    $("cmpTable").innerHTML = rows.join("");

    // ---------- Bar charts ----------
    const charts = [
        ["CV PR-AUC", "Mean over 5 folds, training seasons", (z) => z.cv.PR_AUC, null, p4],
        ["TEST PR-AUC", "Unseen seasons", (z) => z.test.PR_AUC, base.PR_AUC, p4],
        ["TEST TOP-3 HIT RATE", "Podium finishers found per race", (z) => z.test.Top3_Hit, baseTop3, pct],
        ["TEST F1", "At the cross-validated threshold", (z) => z.test.F1, base.F1, p4],
    ];
    $("cmpBars").innerHTML = charts
        .map(([title, sub, fn, baseVal, fmtFn]) => {
            const items = zoo.map((z) => [z.Model + (z.tuned ? " (tuned)" : ""), fn(z), false]);
            if (baseVal != null) items.push(["Baseline", baseVal, true]);
            items.sort((a, b) => b[1] - a[1]);
            const vals = items.map((i) => i[1]);
            const mx = Math.max(...vals), mn = Math.min(...vals);
            const lo = mn - (mx - mn) * 0.6, hi = mx + (mx - mn) * 0.08;
            const bars = items
                .map(([name, v, isBase]) => {
                    const w = Math.max(2, ((v - lo) / (hi - lo)) * 100);
                    return `
                    <div class="feature-row ${isBase ? "baseline-bar" : ""}">
                        <div class="feature-head"><span>${esc(name)}</span><strong>${fmtFn(v)}</strong></div>
                        <div class="bar"><div class="bar-fill" style="width:${w}%"></div></div>
                    </div>`;
                })
                .join("");
            return `
            <div class="panel">
                <div class="panel-label">${title}</div>
                <p class="note">${sub}</p>
                ${bars}
                <p class="note">Axis starts at ${fmtFn(lo)} so small differences stay visible.</p>
            </div>`;
        })
        .join("");

    // ---------- Significance ----------
    $("cmpCi").innerHTML = zoo
        .map((z) => {
            const [v1, c1] = verdict(z.ci.delta_top3);
            const [v2, c2] = verdict(z.ci.delta_pr_auc);
            const mc = z.mcnemar_vs_baseline;
            return `
            <tr class="${z.rank === 1 ? "best-row" : ""}">
                <td>${esc(z.Model)}${z.tuned ? ' <span class="tag">TUNED</span>' : ""}</td>
                <td>${pct(z.test.Top3_Hit)} <span class="muted">${ci4(z.ci.top3)}</span></td>
                <td>${pp(z.test.Top3_Hit - baseTop3)} <span class="muted">${ciPp(z.ci.delta_top3)}</span></td>
                <td class="${c1}">${v1}</td>
                <td>${p4(z.test.PR_AUC)} <span class="muted">${ci4(z.ci.pr_auc)}</span></td>
                <td>${p4(z.test.PR_AUC - base.PR_AUC)} <span class="muted">[${num(z.ci.delta_pr_auc[0], 3)}, ${num(z.ci.delta_pr_auc[1], 3)}]</span></td>
                <td class="${c2}">${v2}</td>
                <td class="${mc && mc.p < 0.05 ? "up" : "muted"}">${mc ? "p = " + mc.p.toFixed(3) + (mc.p < 0.05 ? " (differs)" : " (not different)") : "\u2014"}</td>
            </tr>`;
        })
        .join("");
    $("cmpCiNote").textContent =
        `Bootstrap over ${cmp.n_test_races} test races, ${cmp.n_bootstrap} resamples. ` +
        `Baseline Top-3 hit rate 95% CI: ${ci4(cmp.baseline_ci.top3)}; baseline PR-AUC 95% CI: ${ci4(cmp.baseline_ci.pr_auc)}. ` +
        "Wide, overlapping intervals mean that the test set is too small to separate the models. " +
        "McNemar (exact) compares the threshold-based errors of each model with the baseline on the same rows.";

    // ---------- Profiles ----------
    $("cmpProfiles").innerHTML = plain
        .map((z) => {
            const info = MODEL_INFO[z.Model] || { idea: "\u2014", pros: "\u2014", cons: "\u2014", params: "\u2014" };
            return `
            <div class="model-card ${z.rank === 1 ? "tuned" : ""}">
                <div class="card-label">${esc(z.Family.toUpperCase())}</div>
                <h2>${esc(z.Model)}</h2>
                <p class="profile-line"><strong>How it works.</strong> ${info.idea}</p>
                <p class="profile-line good"><strong>Strengths.</strong> ${info.pros}</p>
                <p class="profile-line bad"><strong>Weaknesses.</strong> ${info.cons}</p>
                <p class="profile-line"><strong>Tuned parameters.</strong> ${info.params}</p>
                <div class="chips">
                    <span class="chip">CV PR-AUC ${p4(z.cv.PR_AUC)}</span>
                    <span class="chip">Test PR-AUC ${p4(z.test.PR_AUC)}</span>
                    <span class="chip">Top-3 ${pct(z.test.Top3_Hit)}</span>
                    <span class="chip">Rank ${plain.indexOf(z) + 1} of ${plain.length}</span>
                </div>
            </div>`;
        })
        .join("");
}

// ============================================================
// CHARTS (PNG, построенные на сервере в ml/pipeline.py)
// ============================================================

const plotImg = (url, alt) => `
    <a href="${url}" target="_blank" rel="noopener">
        <img class="plot-img" src="${url}" alt="${esc(alt)}" loading="lazy">
    </a>`;

const NO_CHARTS = '<div class="empty-state">Charts are not available. Run the pipeline again (/run?refresh=1).</div>';

function selectModelChart(name) {
    if (!PLOTS || !PLOTS.models[name]) return;
    document.querySelectorAll("#modelChartTabs .tab-btn").forEach((b) =>
        b.classList.toggle("active", b.dataset.model === name));
    $("modelChartImage").innerHTML = plotImg(PLOTS.models[name], name);
}

function renderCharts(d) {
    if (!PLOTS) {
        ["modelChartImage", "cmpCharts", "tunedChart"].forEach((id) => ($(id).innerHTML = NO_CHARTS));
        $("modelChartTabs").innerHTML = "";
        return;
    }

    // --- Model Training: вкладки по моделям ---
    const names = Object.keys(PLOTS.models);
    $("modelChartTabs").innerHTML = names
        .map((n) => `<button type="button" class="tab-btn${n.endsWith("Tuned") ? " tuned" : ""}" data-model="${esc(n)}">${esc(n)}</button>`)
        .join("");
    selectModelChart(d.best_model && names.includes(d.best_model) ? d.best_model : names[0]);

    // --- Tuning: график настроенной модели ---
    const tunedName = d.tuning.test_metrics.Model;
    $("tunedChart").innerHTML = PLOTS.models[tunedName] ? plotImg(PLOTS.models[tunedName], tunedName) : NO_CHARTS;

    // --- Comparison: сводные графики ---
    const s = PLOTS.summary;
    const blocks = [
        ["metrics_with_ci", "Test metrics with 95% confidence intervals",
            "Bars: test PR-AUC and Top-3 hit rate. Whiskers: bootstrap over test races. Grey band: baseline interval."],
        ["roc_pr_all_models", "ROC and precision-recall curves of all models",
            "The tuned model is drawn in red, the dashed black line is the grid-position baseline."],
        ["cv_vs_test", "Cross-validation vs test PR-AUC",
            "A large gap between the two bars hints at overfitting or an unrepresentative test period."],
        ["feature_importance", "Permutation importance of the tuned model",
            "Drop in test PR-AUC when a feature is shuffled."],
        ["shap", "SHAP importance of the tuned model",
            "Mean absolute SHAP value per feature (only when the shap package is installed)."],
        ["calibration", "Calibration, Brier score and log-loss",
            "Reliability diagram: points on the dotted diagonal mean that predicted probabilities match real podium frequencies. The isotonic version of the tuned model is dashed."],
        ["mcnemar", "McNemar test between every pair of models",
            "Green cells (p below 0.05) mean the two models make significantly different errors."],
    ];
    $("cmpCharts").innerHTML = blocks
        .filter(([key]) => s[key])
        .map(([key, title, note]) => `
        <figure class="chart-block">
            <figcaption><strong>${title}</strong><span class="note">${note}</span></figcaption>
            ${plotImg(s[key], title)}
        </figure>`)
        .join("");
}

// переключение вкладок и кнопка "VIEW CHARTS" на карточках моделей
document.addEventListener("click", (e) => {
    const tab = e.target.closest("#modelChartTabs .tab-btn");
    if (tab) {
        selectModelChart(tab.dataset.model);
        return;
    }
    const btn = e.target.closest(".chart-btn");
    if (btn) {
        selectModelChart(btn.dataset.model);
        $("modelChartsPanel").scrollIntoView({ behavior: "smooth", block: "start" });
    }
});

// ============================================================
// RUN INFO / PCA / DRIVER PROFILES / MULTILEVEL
// ============================================================

function renderRunInfo(d) {
    const cfg = d.run_config || {};
    const libs = (d.comparison && d.comparison.optional_libraries) || {};
    const missing = Object.entries(libs).filter(([, ok]) => !ok).map(([n]) => n);
    const skipped = d.skipped_models || [];
    const el = $("runInfo");
    el.hidden = false;
    el.innerHTML = `
        <span>LAST RUN <strong>${esc(d.generated_at || "\u2014")}</strong></span>
        <span>MODELS TRAINED <strong>${(cfg.models || []).length}</strong></span>
        <span>MODE <strong>${cfg.quick ? "quick" : "custom / full"}</strong></span>
        ${missing.length ? `<span class="warn">NOT INSTALLED: ${esc(missing.join(", "))} (see Literature Map)</span>` : ""}
        ${skipped.length ? `<span class="warn">SKIPPED: ${skipped.map((s) => esc(s.model)).join(", ")}</span>` : ""}`;
}

function renderPCA(d) {
    const pca = d.pca;
    if (!pca) return;
    $("pcaNote").textContent =
        `${pca.n_components_70} of ${pca.n_features} components explain 70% of the variance, ` +
        `${pca.n_components_90} explain 90%. Strong overlap between features is why correlated ones are removed.`;
    $("pcaTable").innerHTML = `
        <table class="mini-table">
            <thead><tr><th>PC</th><th>VARIANCE</th><th>CORR. WITH PODIUM</th><th>MAIN FEATURES (LOADING)</th></tr></thead>
            <tbody>${pca.components.map((c) => `
                <tr><td>PC${c.pc}</td><td>${pct(c.explained)}</td><td>${num(c.target_corr, 3)}</td>
                <td>${c.top_features.map((f) => `${esc(f.feature)} (${f.loading})`).join(", ")}</td></tr>`).join("")}
            </tbody>
        </table>`;
    $("pcaChart").innerHTML = PLOTS && PLOTS.summary.pca ? plotImg(PLOTS.summary.pca, "PCA scree plot") : "";
}

function renderProfiles(d) {
    const pr = d.driver_profiles;
    if (!pr) return;
    const sil = Object.entries(pr.silhouette_by_k).map(([k, v]) => `k=${k}: ${v}`).join(" \u00b7 ");
    $("profileNote").textContent =
        `KMeans (k=${pr.k}) on driver form, fitted on training data only. Silhouette score by k: ${sil}.`;
    $("profileTable").innerHTML = `
        <table class="mini-table">
            <thead><tr><th>PROFILE</th><th>ROWS</th><th>AVG FINISH (5)</th><th>PODIUM RATE (10)</th><th>DNF RATE</th><th>POINTS SHARE</th></tr></thead>
            <tbody>${pr.profiles.map((p) => `
                <tr><td>${esc(p.profile)}</td><td>${fmt(p.rows)}</td><td>${p.avg_finish_last5}</td>
                <td>${pct(p.podium_rate_last10)}</td><td>${pct(p.dnf_rate_last10)}</td><td>${pct(p.points_share)}</td></tr>`).join("")}
            </tbody>
        </table>`;
}

function renderMultilevel(d) {
    const el = $("cmpMultilevel");
    const ml = d.multilevel;
    if (!ml) { el.hidden = true; return; }
    const v = ml.variance_share;
    const chips = (items) => items.map(([n, e]) => `<span class="chip">${esc(n)} ${e >= 0 ? "+" : ""}${e.toFixed(2)}</span>`).join("");
    el.hidden = false;
    el.innerHTML = `
        <div class="panel-label">MULTILEVEL RANK-LOGIT · VAN KESTEREN &amp; BERGKAMP (2023)</div>
        <h2>Who matters more: the driver or the car?</h2>
        <p class="note">Share of the variance of the estimated "strength" score explained by each level of the model
            (race features, driver random effect, constructor random effect).</p>
        ${barList([["Race features (grid, form, ...)", v.features], ["Driver effect", v.driver], ["Constructor effect", v.constructor]], 1)}
        <div class="panel-label" style="margin-top:20px">STRONGEST CONSTRUCTOR EFFECTS</div>
        <div class="chips">${chips(ml.constructors)}</div>
        <div class="panel-label" style="margin-top:20px">STRONGEST DRIVER EFFECTS</div>
        <div class="chips">${chips(ml.drivers)}</div>`;
}

// ============================================================
// RACE EXPLORER
// ============================================================

let EXPLORER = [];

function renderExplorer(d) {
    EXPLORER = d.explorer || [];
    if (!EXPLORER.length) return;
    const models = Object.keys(EXPLORER[0].drivers[0].probs);
    const tuned = d.tuning.test_metrics.Model;
    const prevRace = $("exRace").value, prevModel = $("exModel").value;
    $("exRace").innerHTML = EXPLORER.map((r, i) => `<option value="${i}">${esc(r.label)}</option>`).join("");
    $("exModel").innerHTML = models.map((m) => `<option ${m === tuned ? "selected" : ""}>${esc(m)}</option>`).join("");
    if (prevRace && EXPLORER[+prevRace]) $("exRace").value = prevRace;
    if (prevModel && models.includes(prevModel)) $("exModel").value = prevModel;
    drawExplorer();
}

const hitsFor = (drivers, scoreFn) => {
    const top = [...drivers].sort((a, b) => scoreFn(b) - scoreFn(a)).slice(0, 3);
    return { top, hits: top.filter((x) => x.actual != null && x.actual <= 3).length };
};

function drawExplorer() {
    const race = EXPLORER[+$("exRace").value];
    const model = $("exModel").value;
    if (!race) return;
    const { top, hits } = hitsFor(race.drivers, (x) => x.probs[model]);
    const podium = race.drivers.filter((x) => x.actual != null && x.actual <= 3)
        .sort((a, b) => a.actual - b.actual);
    const base = hitsFor(race.drivers, (x) => -(x.grid || 99));

    $("exKpis").innerHTML = `
        <div class="kpi-card"><span>MODEL FOUND</span><strong>${hits} / ${Math.min(3, podium.length || 3)}</strong><small>podium finishers among its top 3</small></div>
        <div class="kpi-card"><span>BASELINE FOUND</span><strong>${base.hits} / ${Math.min(3, podium.length || 3)}</strong><small>drivers on the 3 best grid slots</small></div>
        <div class="kpi-card"><span>REAL PODIUM</span><strong class="model-name">${podium.map((x) => esc(x.driver)).join(" \u00b7 ") || "\u2014"}</strong><small>P1 \u00b7 P2 \u00b7 P3</small></div>`;

    const sorted = [...race.drivers].sort((a, b) => b.probs[model] - a.probs[model]);
    $("exTable").innerHTML = sorted.map((x, i) => {
        const isHit = x.actual != null && x.actual <= 3;
        return `<tr class="${i < 3 ? "picked" : ""}">
            <td>${i + 1}</td><td>${esc(x.driver)}</td><td>${esc(x.team)}</td><td>${x.grid ?? "\u2014"}</td>
            <td><div class="bar"><div class="bar-fill" style="width:${x.probs[model] * 100}%"></div></div>${pct(x.probs[model])}</td>
            <td class="${isHit ? "hit-yes" : "hit-no"}">${x.actual == null ? "DNF" : "P" + x.actual}</td></tr>`;
    }).join("");

    const rows = Object.keys(race.drivers[0].probs).map((m) => [m, hitsFor(race.drivers, (x) => x.probs[m]).hits])
        .sort((a, b) => b[1] - a[1]);
    rows.push(["Baseline (grid)", base.hits]);
    $("exModels").innerHTML = rows.map(([m, h]) => `
        <div class="prob-row ${m === model ? "tuned" : ""} ${m.startsWith("Baseline") ? "base" : ""}">
            <span class="name" title="${esc(m)}">${esc(m)}</span>
            <div class="bar"><div class="bar-fill" style="width:${(h / 3) * 100}%"></div></div>
            <span class="val">${h}/3</span>
        </div>`).join("");
}
$("exRace").addEventListener("change", drawExplorer);
$("exModel").addEventListener("change", drawExplorer);

// ============================================================
// WHAT-IF PREDICTOR (calls /api/predict)
// ============================================================

const KEY_INPUTS = [
    ["grid_clean", "Grid position", 1, 20, 1],
    ["qualifying_position", "Qualifying position", 1, 20, 1],
    ["drv_form_finish5", "Driver: average finish, last 5 races", 1, 20, 0.1],
    ["drv_podium_rate10", "Driver: podium rate, last 10 races", 0, 1, 0.01],
    ["drv_dnf_rate10", "Driver: retirement rate, last 10 races", 0, 1, 0.01],
    ["drv_pts_share", "Driver: season points vs leader", 0, 1, 0.01],
    ["drv_circuit_avg_finish", "Driver: average finish at this circuit", 1, 20, 0.1],
    ["con_form_finish5", "Team: average finish, last 5 races", 1, 20, 0.1],
    ["con_podium_rate10", "Team: podium rate per car, last 10", 0, 1, 0.01],
    ["con_pts_share", "Team: season points vs leader", 0, 1, 0.01],
];
const WI = { schema: null, base: null, features: null, teammate: null, field: 20, timer: null };

async function renderPredictor(d) {
    try {
        WI.schema = await api("/api/predict/schema");
    } catch (err) {
        $("wiSliders").innerHTML = `<div class="empty-state">${esc(err.message)}</div>`;
        return;
    }
    if (!EXPLORER.length) return;
    $("wiRace").innerHTML = EXPLORER.map((r, i) => `<option value="${i}">${esc(r.label)}</option>`).join("");
    fillWiDrivers();
}

function fillWiDrivers() {
    const race = EXPLORER[+$("wiRace").value];
    $("wiDriver").innerHTML = race.drivers.map((x, i) =>
        `<option value="${i}">${esc(x.driver)} (${esc(x.team)}, grid ${x.grid ?? "pit"})</option>`).join("");
    loadPreset();
}

function loadPreset() {
    const race = EXPLORER[+$("wiRace").value];
    const drv = race.drivers[+$("wiDriver").value];
    WI.base = { ...drv.features };
    WI.features = { ...drv.features };
    WI.teammate = drv.teammate_grid;
    WI.field = drv.field_size;
    buildSliders();
    wiPredict();
}

function buildSliders() {
    const have = new Set(WI.schema.features);
    $("wiSliders").innerHTML = KEY_INPUTS.filter(([f]) => have.has(f)).map(([f, label, lo, hi, step]) => {
        const v = WI.features[f] ?? WI.schema.defaults[f];
        WI.features[f] = v;
        return `<div class="slider-row">
            <label><span>${label}</span><strong id="wiv_${f}">${Number(v).toFixed(step < 1 ? 2 : 0)}</strong></label>
            <input type="range" data-f="${f}" min="${lo}" max="${hi}" step="${step}" value="${v}">
        </div>`;
    }).join("");
}

$("wiSliders").addEventListener("input", (e) => {
    const f = e.target.dataset.f;
    if (!f) return;
    const v = Number(e.target.value);
    WI.features[f] = v;
    const step = Number(e.target.step);
    $("wiv_" + f).textContent = v.toFixed(step < 1 ? 2 : 0);
    clearTimeout(WI.timer);
    WI.timer = setTimeout(wiPredict, 150);
});
$("wiRace").addEventListener("change", fillWiDrivers);
$("wiDriver").addEventListener("change", loadPreset);
$("wiReset").addEventListener("click", loadPreset);

async function wiPredict() {
    if (!WI.features) return;
    try {
        const res = await api("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ features: { ...WI.features, field_size: WI.field }, teammate_grid: WI.teammate }),
        });
        const thr = res.thresholds || {};
        const tuned = WI.schema.tuned_name;
        const rows = Object.entries(res.probabilities).sort((a, b) => b[1] - a[1]);
        $("wiOutput").innerHTML = rows.map(([m, p]) => {
            const yes = thr[m] != null && p >= thr[m];
            return `<div class="prob-row ${m === tuned ? "tuned" : ""} ${m.startsWith("Baseline") ? "base" : ""} ${yes ? "yes" : ""}">
                <span class="name" title="${esc(m)}">${esc(m)}</span>
                <div class="bar"><div class="bar-fill" style="width:${p * 100}%"></div></div>
                <span class="val">${(p * 100).toFixed(1)}%${yes ? '<span class="prob-mark">\u2713</span>' : ""}</span>
            </div>`;
        }).join("");
        $("wiNote").textContent =
            "A green \u2713 means the probability passes that model's decision threshold (podium predicted). " +
            "Models that need the whole grid or race history (Multilevel Rank-Logit, Pairwise Ranker, LSTM/GRU/TCN) " +
            "are shown in the Race Explorer instead.";
    } catch (err) {
        $("wiOutput").innerHTML = `<div class="empty-state">${esc(err.message)}</div>`;
    }
}

// ============================================================
// LITERATURE MAP: methods from the review -> code
// ============================================================

const LITERATURE = [
    { src: "Master thesis, National College of Ireland (2021 championship rankings)",
      method: "Learning for championship ranking prediction",
      impl: "Pairwise Ranker (RankSVM-style) and the listwise Multilevel Rank-Logit",
      models: ["Pairwise Ranker", "Multilevel Rank-Logit"], status: "done" },
    { src: "Patil et al. (2023), CCIS 1662",
      method: "Exploratory analysis, PCA, collinearity control",
      impl: "PCA analysis (scree plot, loadings), correlation filter |r| > 0.95, PCA + Logistic Regression",
      models: ["PCA + Logistic Regression"], status: "done" },
    { src: "Fatima & Johrendt (2023)",
      method: "Two embedded deep neural networks (EDNN)",
      impl: "EDNN-style: one network on start features, one on form features, outputs fed to a final network",
      models: ["EDNN-style (2 networks)"], status: "partial",
      note: "The pit-window model needs lap data that the CSV files do not contain." },
    { src: "Garc\u00eda Tejada (2023)",
      method: "Random Forest, SVM, ANN",
      impl: "Random Forest, SVM (RBF), Neural Network (MLP)",
      models: ["Random Forest", "SVM (RBF)", "Neural Network (MLP)"], status: "done" },
    { src: "Surendran (2023)",
      method: "LSTM and GRU time-series models",
      impl: "LSTM and GRU over each driver's last 5 races plus the race features",
      models: ["LSTM (last 5 races)", "GRU (last 5 races)"], status: "done", needs: "PyTorch" },
    { src: "van Kesteren & Bergkamp (2023)",
      method: "Bayesian multilevel rank-ordered logit",
      impl: "Multilevel Rank-Logit: Plackett-Luce with driver and constructor random effects",
      models: ["Multilevel Rank-Logit"], status: "partial",
      note: "MAP estimate with Gaussian priors instead of full MCMC sampling." },
    { src: "Aguad & Thraves (2024)",
      method: "Dynamic programming and Stackelberg game for pit-stop strategy",
      impl: "Not implemented", models: [], status: "na",
      note: "An optimisation model, not a supervised learner, and it needs pit-stop data." },
    { src: "Hettmann (2024)",
      method: "Driver clustering and binary classification",
      impl: "KMeans driver profiles (fit on train only) used as a categorical feature",
      models: [], status: "done", live: "profiles" },
    { src: "Leischner (2024)",
      method: "Tyre-compound decisions via clustering",
      impl: "Clustering idea reused for drivers", models: [], status: "na",
      note: "Tyre data is not in the CSV files." },
    { src: "Sasikumar, Leema & Balakrishnan (2025)",
      method: "Bi-LSTM, TCN, GRU, InceptionTime, CNN-BiLSTM; PR-AUC, bootstrap, McNemar",
      impl: "Bi-LSTM, TCN, GRU, CNN-BiLSTM; PR-AUC with bootstrap CI; exact McNemar test",
      models: ["Bi-LSTM (last 5 races)", "TCN (last 5 races)", "CNN-BiLSTM (last 5 races)", "GRU (last 5 races)"],
      status: "partial", needs: "PyTorch", note: "InceptionTime is not implemented." },
    { src: "Mistry & Smith (2025)",
      method: "RNN simulation of strategic decisions",
      impl: "Recurrent sequence models (LSTM/GRU) for prediction", models: ["LSTM (last 5 races)"],
      status: "partial", needs: "PyTorch", note: "No strategy simulation: needs detailed session data." },
    { src: "Sargut Yathishkumar (2025)",
      method: "Random Forest + LightGBM, class weights, isotonic calibration, SHAP, Brier, log-loss, chronological validation",
      impl: "Random Forest and LightGBM with class weights, isotonic calibration of the tuned model, Brier and log-loss, SHAP, chronological split",
      models: ["Random Forest (balanced)", "LightGBM (balanced)", "__calibrated__"],
      status: "done", needs: "LightGBM, SHAP" },
];

function renderLiterature(d) {
    const zoo = d ? d.model_zoo : [];
    const byName = Object.fromEntries(zoo.map((z) => [z.Model, z]));
    const libs = d ? d.comparison.optional_libraries : {};
    const counts = { done: 0, partial: 0, na: 0 };
    const pill = { done: "IMPLEMENTED", partial: "PARTIAL", na: "NOT APPLICABLE", missing: "NOT RUN" };

    $("litTable").innerHTML = LITERATURE.map((r, i) => {
        counts[r.status]++;
        const names = r.models.map((m) => (m === "__calibrated__"
            ? (zoo.find((z) => z.calibrated) || {}).Model : m)).filter(Boolean);
        const live = names.map((n) => {
            const z = byName[n];
            return z ? `<div>${esc(n)}: PR-AUC ${p4(z.test.PR_AUC)} \u00b7 Top-3 ${pct(z.test.Top3_Hit)}</div>`
                     : (d ? `<div class="muted">${esc(n)}: not trained</div>` : "");
        }).join("");
        const liveExtra = r.live === "profiles" && d && d.driver_profiles
            ? `<div>${d.driver_profiles.k} profiles, silhouette ${d.driver_profiles.silhouette_by_k[d.driver_profiles.k]}</div>` : "";
        const notRun = d && names.length && names.every((n) => !byName[n]);
        const st = notRun ? "missing" : r.status;
        const need = notRun && r.needs ? `<div class="muted">needs: ${esc(r.needs)}</div>` : "";
        return `<tr>
            <td>${i + 1}</td>
            <td>${esc(r.src)}</td>
            <td>${esc(r.method)}</td>
            <td>${esc(r.impl)}${r.note ? `<div class="muted">${esc(r.note)}</div>` : ""}</td>
            <td><span class="status-pill ${st}">${pill[st]}</span></td>
            <td>${live || liveExtra || '<span class="muted">\u2014</span>'}${live ? liveExtra : ""}${need}</td>
        </tr>`;
    }).join("");
    $("litSummary").innerHTML =
        `<span class="chip">${counts.done} implemented</span><span class="chip">${counts.partial} partly implemented</span>` +
        `<span class="chip">${counts.na} not applicable with these CSV files</span>` +
        (d ? Object.entries(libs).map(([n, ok]) => `<span class="chip">${ok ? "\u2713" : "\u2717"} ${esc(n)}</span>`).join("") : "");
}

// ============================================================
// EXPORT
// ============================================================

function download(name, text, type) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([text], { type }));
    a.download = name;
    a.click();
    URL.revokeObjectURL(a.href);
}

$("exportJson").addEventListener("click", () => {
    if (!LAST_RESULT) return alert("Run the pipeline first.");
    const copy = { ...LAST_RESULT };
    delete copy.explorer;
    download("f1_ml_results.json", JSON.stringify(copy, null, 2), "application/json");
});

$("exportCsv").addEventListener("click", () => {
    if (!LAST_RESULT) return alert("Run the pipeline first.");
    const head = ["rank", "model", "family", "cv_pr_auc", "cv_pr_auc_std", "cv_roc_auc", "cv_f1", "test_pr_auc",
                  "test_roc_auc", "test_f1", "test_top3", "test_brier", "test_logloss", "mcnemar_p_vs_baseline"];
    const rows = LAST_RESULT.model_zoo.map((z) => [z.rank, z.Model, z.Family, z.cv.PR_AUC, z.cv.PR_AUC_std,
        z.cv.ROC_AUC, z.cv.F1, z.test.PR_AUC, z.test.ROC_AUC, z.test.F1, z.test.Top3_Hit, z.test.Brier,
        z.test.LogLoss, z.mcnemar_vs_baseline ? z.mcnemar_vs_baseline.p : ""]);
    download("f1_leaderboard.csv",
        [head, ...rows].map((r) => r.map((c) => `"${String(c ?? "").replace(/"/g, '""')}"`).join(",")).join("\n"),
        "text/csv");
});

// ============================================================
// START
// ============================================================

initApp();