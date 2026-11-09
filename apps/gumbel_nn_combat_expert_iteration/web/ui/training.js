function trainingPanel() {
  const el = id => document.getElementById(id);
  const number = (value, digits = 1) => Number.isFinite(value) ?
    value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';
  let progress = {}, sampleStage = '', sample = null, available = false;
  const showTraining = smoothView(el('trainingValues'), (values, latest) => {
    el('trainingSteps').textContent = `${number(latest.step, 0)} / ${number(latest.total, 0)}`;
    el('trainingRate').textContent = number(values.steps_s);
    el('trainingValueLoss').textContent = number(values.value_loss, 4);
    el('trainingPolicyLoss').textContent = number(values.policy_loss, 4);
    el('trainingFill').style.width = values.total > 0 ? `${Math.max(0, Math.min(100, 100 * values.step / values.total))}%` : '0';
    if (values.total > 0) {
      el('trainingProgress').setAttribute('aria-valuemin', '0');
      el('trainingProgress').setAttribute('aria-valuemax', String(values.total));
      el('trainingProgress').setAttribute('aria-valuenow', String(latest.step));
    }
  });
  const showSystem = smoothView(el('trainingResources'), values => {
    el('trainingGpu').textContent = Number.isFinite(values.gpu) ? `${number(values.gpu)}%` : '—';
    el('trainingVram').textContent = `${number(values.vram)} / ${number(values.vram_total)}`;
    el('trainingRam').textContent = `${number(values.ram)} / ${number(values.ram_total)}`;
  }, 2500);
  const active = () => progress.running && progress.phase === 'train';
  function status(message) {
    const live = active() && available && sampleStage === progress.stage;
    el('trainingValues').classList.toggle('unavailable', !live);
    el('trainingStatus').textContent = message || (live ?
      `${number(sample.window_seconds)} s loss window · 1.5 s playback delay · ${number(sample.seconds, 0)} s elapsed` :
      active() ? 'Waiting for training telemetry' : sampleStage ? `Idle · last sample from ${sampleStage}` : 'Idle');
  }
  async function get(path) {
    const response = await fetch('api/' + path, { cache: 'no-store' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();
    if (data?.error) throw new Error(data.error);
    return data;
  }
  async function pollTraining() {
    if (active()) {
      const stage = progress.stage;
      try {
        const data = await get('training');
        if (active() && progress.stage === stage) {
          available = data !== null;
          if (!data) showTraining.pause();
          if (data) {
            showTraining(data, sampleStage !== stage);
            sample = data;
            sampleStage = stage;
          }
          status();
        }
      } catch (error) {
        showTraining.pause();
        available = false;
        status('Training telemetry unavailable');
      }
    }
    setTimeout(pollTraining, 1000);
  }
  async function pollSystem() {
    try {
      showSystem(await get('system'));
      el('trainingSystemStatus').textContent = 'Device-wide GPU · host RAM · 2.5 s playback delay';
    } catch (error) {
      showSystem({ gpu: null, vram: null, vram_total: null, ram: null, ram_total: null });
      el('trainingSystemStatus').textContent = 'System telemetry unavailable';
    }
    setTimeout(pollSystem, 2000);
  }
  showTraining({ step: null, total: null, steps_s: null, value_loss: null, policy_loss: null });
  pollTraining();
  pollSystem();
  return latest => {
    if (latest.stage !== progress.stage || latest.phase !== progress.phase || !latest.running) {
      showTraining.pause(); available = false;
    }
    progress = latest; status();
  };
}
