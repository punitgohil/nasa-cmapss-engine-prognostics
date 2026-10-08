(() => {
  const data = window.DASHBOARD_DATA;
  if (!data) {
    document.body.innerHTML = '<p>Dashboard data is missing. Run the dashboard data builder.</p>';
    return;
  }

  const colors = { global_baseline: '#94a3b8', condition_aware: '#38bdf8', positive: '#34d399', negative: '#fb7185', grid: '#1d3149', text: '#9eb0c5' };
  const labels = { global_baseline: 'Global baseline', condition_aware: 'Condition-aware' };
  const arms = Object.fromEntries(data.results.arms.map(item => [item.arm, item]));
  const predictions = data.predictions;
  const ns = 'http://www.w3.org/2000/svg';

  function svgNode(name, attrs = {}) {
    const node = document.createElementNS(ns, name);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    return node;
  }

  function clearSvg(id, height = 360) {
    const svg = document.getElementById(id);
    svg.innerHTML = '';
    const width = Math.max(320, svg.clientWidth || 700);
    svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    return { svg, width, height };
  }

  const scale = (value, domainMin, domainMax, rangeMin, rangeMax) => rangeMin + (value - domainMin) / (domainMax - domainMin) * (rangeMax - rangeMin);

  function axes(svg, width, height, xMax, yMax, xLabel, yLabel) {
    const m = { left: 56, right: 20, top: 18, bottom: 48 };
    for (let i = 0; i <= 4; i++) {
      const y = scale(i, 0, 4, height - m.bottom, m.top);
      svg.appendChild(svgNode('line', { x1: m.left, x2: width - m.right, y1: y, y2: y, stroke: colors.grid }));
      const value = (yMax * i / 4).toFixed(0);
      const text = svgNode('text', { x: m.left - 9, y: y + 4, 'text-anchor': 'end', fill: colors.text, 'font-size': 11 });
      text.textContent = value;
      svg.appendChild(text);
    }
    for (let i = 0; i <= 4; i++) {
      const x = scale(i, 0, 4, m.left, width - m.right);
      const text = svgNode('text', { x, y: height - 24, 'text-anchor': 'middle', fill: colors.text, 'font-size': 11 });
      text.textContent = (xMax * i / 4).toFixed(0);
      svg.appendChild(text);
    }
    const xl = svgNode('text', { x: (m.left + width - m.right) / 2, y: height - 4, 'text-anchor': 'middle', fill: colors.text, 'font-size': 12 });
    xl.textContent = xLabel;
    svg.appendChild(xl);
    const yl = svgNode('text', { x: 14, y: height / 2, 'text-anchor': 'middle', fill: colors.text, 'font-size': 12, transform: `rotate(-90 14 ${height / 2})` });
    yl.textContent = yLabel;
    svg.appendChild(yl);
    return m;
  }

  function renderSummary() {
    const global = arms.global_baseline.overall;
    const condition = arms.condition_aware.overall;
    const widthGlobal = arms.global_baseline.intervals['90'].mean_interval_width;
    const widthCondition = arms.condition_aware.intervals['90'].mean_interval_width;
    const bootstrap = data.analysis.paired_bootstrap;
    const ci = bootstrap.bootstrap_95_interval;
    const cards = [
      ['Condition-aware MAE', `${condition.mae.toFixed(2)} cycles`, `Observed gain ${(global.mae - condition.mae).toFixed(2)} · 95% CI ${ci[0].toFixed(2)} to ${ci[1].toFixed(2)}`],
      ['Condition-aware RMSE', `${condition.rmse.toFixed(2)} cycles`, `${(global.rmse - condition.rmse).toFixed(2)} lower than global`],
      ['Actual 90% coverage', `${(arms.condition_aware.intervals['90'].empirical_coverage * 100).toFixed(1)}%`, 'Nominal target: 90%'],
      ['Mean 90% interval', `${widthCondition.toFixed(1)} cycles`, `${(widthGlobal - widthCondition).toFixed(1)} narrower than global`]
    ];
    document.getElementById('summary-cards').innerHTML = cards.map(([label, value, delta]) => `<article class="summary-card"><div class="summary-label">${label}</div><div class="summary-value">${value}</div><div class="summary-delta">${delta}</div></article>`).join('');
  }

  function renderScatter(arm) {
    const { svg, width, height } = clearSvg('scatter-chart', 390);
    const frame = predictions.filter(row => row.arm === arm);
    const max = Math.ceil(Math.max(...frame.flatMap(row => [row.true_rul, row.predicted_rul])) / 20) * 20;
    const m = axes(svg, width, height, max, max, 'Actual RUL (cycles)', 'Predicted RUL (cycles)');
    const x = value => scale(value, 0, max, m.left, width - m.right);
    const y = value => scale(value, 0, max, height - m.bottom, m.top);
    svg.appendChild(svgNode('line', { x1: x(0), y1: y(0), x2: x(max), y2: y(max), stroke: '#64748b', 'stroke-dasharray': '6 5' }));
    frame.forEach(row => {
      const point = svgNode('circle', { cx: x(row.true_rul), cy: y(row.predicted_rul), r: 3.2, fill: colors[arm], opacity: .7 });
      const title = svgNode('title');
      title.textContent = `Engine ${row.unit_id}: actual ${row.true_rul.toFixed(0)}, predicted ${row.predicted_rul.toFixed(1)}`;
      point.appendChild(title);
      svg.appendChild(point);
    });
  }

  function groupedBars(id, rows, groupKey) {
    const { svg, width, height } = clearSvg(id, 350);
    const m = { left: 54, right: 18, top: 20, bottom: 55 };
    const max = Math.max(...rows.flatMap(row => [row.global_mae, row.condition_aware_mae])) * 1.15;
    const groupWidth = (width - m.left - m.right) / rows.length;
    for (let i = 0; i <= 4; i++) {
      const value = max * i / 4;
      const y = scale(value, 0, max, height - m.bottom, m.top);
      svg.appendChild(svgNode('line', { x1: m.left, x2: width - m.right, y1: y, y2: y, stroke: colors.grid }));
      const label = svgNode('text', { x: m.left - 8, y: y + 4, 'text-anchor': 'end', fill: colors.text, 'font-size': 11 });
      label.textContent = value.toFixed(0);
      svg.appendChild(label);
    }
    rows.forEach((row, index) => {
      const center = m.left + groupWidth * (index + .5);
      const barWidth = Math.min(30, groupWidth * .28);
      [['global_mae', 'global_baseline', -barWidth / 1.8], ['condition_aware_mae', 'condition_aware', barWidth / 1.8]].forEach(([field, arm, offset]) => {
        const value = row[field];
        const y = scale(value, 0, max, height - m.bottom, m.top);
        const rect = svgNode('rect', { x: center + offset - barWidth / 2, y, width: barWidth, height: height - m.bottom - y, rx: 3, fill: colors[arm] });
        const title = svgNode('title');
        title.textContent = `${labels[arm]}: ${value.toFixed(2)} cycles`;
        rect.appendChild(title);
        svg.appendChild(rect);
      });
      const label = svgNode('text', { x: center, y: height - 28, 'text-anchor': 'middle', fill: colors.text, 'font-size': 11 });
      label.textContent = row[groupKey];
      svg.appendChild(label);
    });
    const yLabel = svgNode('text', { x: 13, y: height / 2, 'text-anchor': 'middle', fill: colors.text, 'font-size': 12, transform: `rotate(-90 13 ${height / 2})` });
    yLabel.textContent = 'MAE (cycles)';
    svg.appendChild(yLabel);
  }

  function lineChart(id, accessor, yLabel, ideal = false) {
    const { svg, width, height } = clearSvg(id, 310);
    const m = { left: 54, right: 18, top: 20, bottom: 48 };
    const points = data.results.arms.flatMap(arm => Object.entries(arm.intervals).map(([level, values]) => ({ arm: arm.arm, x: Number(level), y: accessor(values) })));
    const yValues = points.map(point => point.y);
    const yMin = ideal ? 75 : Math.min(...yValues) * .9;
    const yMax = ideal ? 100 : Math.max(...yValues) * 1.08;
    for (let i = 0; i <= 4; i++) {
      const yValue = yMin + (yMax - yMin) * i / 4;
      const y = scale(yValue, yMin, yMax, height - m.bottom, m.top);
      svg.appendChild(svgNode('line', { x1: m.left, x2: width - m.right, y1: y, y2: y, stroke: colors.grid }));
      const text = svgNode('text', { x: m.left - 8, y: y + 4, 'text-anchor': 'end', fill: colors.text, 'font-size': 11 });
      text.textContent = yValue.toFixed(0);
      svg.appendChild(text);
    }
    [80, 90, 95].forEach(level => {
      const x = scale(level, 80, 95, m.left, width - m.right);
      const text = svgNode('text', { x, y: height - 24, 'text-anchor': 'middle', fill: colors.text, 'font-size': 11 });
      text.textContent = `${level}%`;
      svg.appendChild(text);
    });
    if (ideal) {
      svg.appendChild(svgNode('line', { x1: scale(80,80,95,m.left,width-m.right), y1: scale(80,yMin,yMax,height-m.bottom,m.top), x2: scale(95,80,95,m.left,width-m.right), y2: scale(95,yMin,yMax,height-m.bottom,m.top), stroke: '#64748b', 'stroke-dasharray': '5 5' }));
    }
    Object.keys(labels).forEach(arm => {
      const rows = points.filter(point => point.arm === arm).sort((a,b) => a.x-b.x);
      const path = rows.map((point, index) => `${index ? 'L' : 'M'} ${scale(point.x,80,95,m.left,width-m.right)} ${scale(point.y,yMin,yMax,height-m.bottom,m.top)}`).join(' ');
      svg.appendChild(svgNode('path', { d: path, fill: 'none', stroke: colors[arm], 'stroke-width': 2.5 }));
      rows.forEach(point => svg.appendChild(svgNode('circle', { cx: scale(point.x,80,95,m.left,width-m.right), cy: scale(point.y,yMin,yMax,height-m.bottom,m.top), r: 4, fill: colors[arm] })));
    });
    const label = svgNode('text', { x: 13, y: height / 2, 'text-anchor': 'middle', fill: colors.text, 'font-size': 12, transform: `rotate(-90 13 ${height / 2})` });
    label.textContent = yLabel;
    svg.appendChild(label);
  }

  function renderEngine() {
    const engine = Number(document.getElementById('engine-select').value);
    const level = document.getElementById('level-select').value;
    const rows = predictions.filter(row => Number(row.unit_id) === engine);
    const { svg, width, height } = clearSvg('engine-chart', 260);
    const values = rows.flatMap(row => [row[`lower_${level}`], row[`upper_${level}`], row.true_rul]);
    const min = Math.min(0, ...values) - 5;
    const max = Math.max(...values) + 5;
    const left = 130, right = 28;
    const x = value => scale(value, min, max, left, width - right);
    [min, (min+max)/2, max].forEach(value => {
      const pos = x(value);
      svg.appendChild(svgNode('line', { x1: pos, x2: pos, y1: 35, y2: 210, stroke: colors.grid }));
      const text = svgNode('text', { x: pos, y: 235, 'text-anchor': 'middle', fill: colors.text, 'font-size': 11 });
      text.textContent = value.toFixed(0);
      svg.appendChild(text);
    });
    rows.forEach((row, index) => {
      const y = 75 + index * 85;
      const arm = row.arm;
      const name = svgNode('text', { x: 8, y: y + 4, fill: colors.text, 'font-size': 12 });
      name.textContent = labels[arm];
      svg.appendChild(name);
      svg.appendChild(svgNode('line', { x1: x(row[`lower_${level}`]), x2: x(row[`upper_${level}`]), y1: y, y2: y, stroke: colors[arm], 'stroke-width': 8, 'stroke-linecap': 'round', opacity: .55 }));
      svg.appendChild(svgNode('circle', { cx: x(row.predicted_rul), cy: y, r: 6, fill: colors[arm] }));
      svg.appendChild(svgNode('line', { x1: x(row.true_rul), x2: x(row.true_rul), y1: y-15, y2: y+15, stroke: '#f8fafc', 'stroke-width': 2 }));
    });
    const caption = svgNode('text', { x: left, y: 20, fill: colors.text, 'font-size': 11 });
    caption.textContent = `Actual RUL ${rows[0].true_rul.toFixed(0)} cycles · vertical marker`; 
    svg.appendChild(caption);
  }

  function renderSilhouette() {
    const { svg, width, height } = clearSvg('silhouette-chart', 350);
    const rows = data.analysis.regime_structure.candidate_regime_counts;
    const m = { left: 54, right: 18, top: 20, bottom: 48 };
    const min = .55, max = 1;
    const path = rows.map((row, index) => `${index ? 'L' : 'M'} ${scale(row.regime_count,2,10,m.left,width-m.right)} ${scale(row.silhouette_score,min,max,height-m.bottom,m.top)}`).join(' ');
    svg.appendChild(svgNode('path', { d: path, fill: 'none', stroke: colors.condition_aware, 'stroke-width': 2.5 }));
    rows.forEach(row => {
      const selected = row.regime_count === 6;
      svg.appendChild(svgNode('circle', { cx: scale(row.regime_count,2,10,m.left,width-m.right), cy: scale(row.silhouette_score,min,max,height-m.bottom,m.top), r: selected ? 7 : 4, fill: selected ? colors.positive : colors.condition_aware }));
      const tick = svgNode('text', { x: scale(row.regime_count,2,10,m.left,width-m.right), y: height-24, 'text-anchor': 'middle', fill: colors.text, 'font-size': 11 });
      tick.textContent = row.regime_count;
      svg.appendChild(tick);
    });
    const note = svgNode('text', { x: scale(6,2,10,m.left,width-m.right), y: scale(.997,min,max,height-m.bottom,m.top)-13, 'text-anchor': 'middle', fill: colors.positive, 'font-size': 11 });
    note.textContent = 'k=6 · 0.997';
    svg.appendChild(note);
  }

  function renderRegimeTable() {
    const comparisons = Object.fromEntries(data.analysis.by_regime.map(row => [row.group, row]));
    const rows = data.analysis.regime_structure.regimes.map(regime => {
      const comparison = comparisons[String(regime.regime)];
      const gain = comparison.mae_improvement;
      return `<tr><td>Regime ${regime.regime}</td><td>${regime.development_rows.toLocaleString()}</td><td>${regime.test_endpoints}</td><td>${comparison.global_mae.toFixed(2)}</td><td>${comparison.condition_aware_mae.toFixed(2)}</td><td class="${gain >= 0 ? 'gain' : 'loss'}">${gain >= 0 ? '+' : ''}${gain.toFixed(2)}</td></tr>`;
    }).join('');
    document.getElementById('regime-table').innerHTML = `<table><thead><tr><th>Inferred regime</th><th>Development rows</th><th>Test engines</th><th>Global MAE</th><th>Condition MAE</th><th>MAE improvement</th></tr></thead><tbody>${rows}</tbody></table>`;
  }

  function renderFinding() {
    const bands = Object.fromEntries(data.analysis.by_true_rul_band.map(row => [row.group, row]));
    const regimes = data.analysis.by_regime;
    const best = regimes.reduce((a,b) => a.mae_improvement > b.mae_improvement ? a : b);
    const bootstrap = data.analysis.paired_bootstrap;
    const ci = bootstrap.bootstrap_95_interval;
    document.getElementById('finding-copy').innerHTML = `<p><strong>The six regimes are structurally clear, but explicit condition features add only limited predictive information.</strong> The six-cluster silhouette score is 0.997 and all regimes have thousands of development rows, so weak or tiny clusters are not the explanation. The global nonlinear model already receives the three operating-setting columns and can learn many condition-dependent splits itself.</p><p>The observed overall MAE gain is ${bootstrap.mean_mae_improvement.toFixed(2)} cycles, but its paired 95% bootstrap interval is ${ci[0].toFixed(2)} to ${ci[1].toFixed(2)} and therefore includes zero. Condition-aware prediction is more accurate for ${(bootstrap.fraction_of_engines_with_lower_condition_aware_absolute_error * 100).toFixed(1)}% of test engines. The aggregate result is modest and inconclusive.</p><p>The gain is concentrated rather than universal: regime ${best.group} improves by ${best.mae_improvement.toFixed(2)} MAE cycles, while regime 2 becomes slightly worse. By failure stage, condition-awareness improves MAE most for true RUL 51–100 (${bands['51-100'].mae_improvement.toFixed(2)} cycles), but is worse close to failure at ≤25 RUL (${Math.abs(bands['<=25'].mae_improvement).toFixed(2)} cycles). This supports a narrow conclusion: explicit regimes help in selected contexts, not across the entire degradation path.</p>`;
  }

  function initialise() {
    const ext = data.extensions;
    const rows = [
      ['FD002 Phase 1', 'Global snapshot', arms.global_baseline.overall],
      ['FD002 Phase 1', 'Condition-aware snapshot', arms.condition_aware.overall],
      ['FD002 Phase 2', 'Condition-aware snapshot', ext.method3.official_test_exploratory.arms.phase2_snapshot],
      ['FD002 Phase 2', 'Engineered history', ext.method3.official_test_exploratory.arms.phase2_engineered_history],
      ['FD002 Phase 2', 'Temporal CNN', ext.tcn.official_test_exploratory.temporal_cnn],
      ...Object.entries(ext.fd004.official_test.arms).map(([arm, metrics]) => ['FD004 replication', {global:'Global snapshot',condition_aware:'Condition-aware snapshot',engineered_history:'Engineered history',temporal_cnn:'Temporal CNN'}[arm], metrics])
    ];
    document.getElementById('complete-results').innerHTML = '<table><thead><tr><th>Experiment</th><th>Method</th><th>Test engines</th><th>MAE (cycles)</th><th>RMSE (cycles)</th></tr></thead><tbody>' + rows.map(([experiment, method, metrics]) => `<tr><td>${experiment}</td><td>${method}</td><td>${metrics.n}</td><td>${metrics.mae.toFixed(2)}</td><td>${metrics.rmse.toFixed(2)}</td></tr>`).join('') + '</tbody></table>';
    const comparison = ext.fd004.official_test.paired_comparisons.engineered_history_over_condition_aware;
    document.getElementById('complete-finding').textContent = `Engineered history is the strongest MAE method on both test subsets. On FD004 it improves MAE by ${comparison.mae_improvement.toFixed(2)} cycles over condition-aware snapshots (paired bootstrap 95% interval ${comparison.bootstrap_95_interval.map(x => x.toFixed(2)).join(' to ')}). The TCN ties engineered history on FD002 and falls behind on FD004. These fixed experiments do not measure variation across training seeds or performance on real equipment.`;
    renderSummary();
    const armControl = document.getElementById('scatter-arm');
    armControl.innerHTML = Object.keys(labels).map((arm, index) => `<button type="button" data-arm="${arm}" class="${index === 1 ? 'active' : ''}">${labels[arm]}</button>`).join('');
    armControl.addEventListener('click', event => {
      if (!event.target.dataset.arm) return;
      armControl.querySelectorAll('button').forEach(button => button.classList.toggle('active', button === event.target));
      renderScatter(event.target.dataset.arm);
    });
    renderScatter('condition_aware');
    groupedBars('rul-band-chart', data.analysis.by_true_rul_band, 'group');
    groupedBars('regime-chart', data.analysis.by_regime, 'group');
    lineChart('coverage-chart', value => value.empirical_coverage * 100, 'Actual coverage (%)', true);
    lineChart('width-chart', value => value.mean_interval_width, 'Mean width (cycles)');
    const engineSelect = document.getElementById('engine-select');
    const engineIds = [...new Set(predictions.map(row => Number(row.unit_id)))].sort((a,b) => a-b);
    engineSelect.innerHTML = engineIds.map(id => `<option value="${id}">${id}</option>`).join('');
    engineSelect.value = engineIds[Math.floor(engineIds.length / 2)];
    engineSelect.addEventListener('change', renderEngine);
    document.getElementById('level-select').addEventListener('change', renderEngine);
    renderEngine();
    renderSilhouette();
    renderRegimeTable();
    renderFinding();
  }

  initialise();
  window.addEventListener('resize', () => {
    const active = document.querySelector('#scatter-arm button.active');
    renderScatter(active ? active.dataset.arm : 'condition_aware');
    groupedBars('rul-band-chart', data.analysis.by_true_rul_band, 'group');
    groupedBars('regime-chart', data.analysis.by_regime, 'group');
    lineChart('coverage-chart', value => value.empirical_coverage * 100, 'Actual coverage (%)', true);
    lineChart('width-chart', value => value.mean_interval_width, 'Mean width (cycles)');
    renderEngine();
    renderSilhouette();
  });
})();
