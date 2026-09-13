const menuButton = document.querySelector(".menu-toggle");
const siteNav = document.querySelector("#site-nav");

if (siteNav) {
  const currentSection = siteNav.querySelector('[aria-current="page"]');
  const revealCurrentSection = () => {
    if (!currentSection || siteNav.scrollWidth <= siteNav.clientWidth) return;
    siteNav.scrollLeft = Math.max(0, currentSection.offsetLeft - (siteNav.clientWidth - currentSection.offsetWidth) / 2);
  };

  requestAnimationFrame(revealCurrentSection);
  window.addEventListener("resize", revealCurrentSection);
}

if (menuButton && siteNav) {
  const closeMenu = (returnFocus = false) => {
    menuButton.setAttribute("aria-expanded", "false");
    menuButton.textContent = "Menu";
    siteNav.classList.remove("open");
    document.body.classList.remove("menu-open");
    if (returnFocus) menuButton.focus();
  };

  menuButton.addEventListener("click", () => {
    const open = menuButton.getAttribute("aria-expanded") !== "true";
    if (!open) {
      closeMenu();
      return;
    }
    menuButton.setAttribute("aria-expanded", String(open));
    menuButton.textContent = "Close";
    siteNav.classList.add("open");
    document.body.classList.add("menu-open");
  });

  siteNav.querySelectorAll("a").forEach((link) => link.addEventListener("click", () => closeMenu()));
  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape" && siteNav.classList.contains("open")) closeMenu(true);
  });
  document.addEventListener("click", (event) => {
    if (!siteNav.classList.contains("open")) return;
    if (!siteNav.contains(event.target) && !menuButton.contains(event.target)) closeMenu();
  });
  window.matchMedia("(min-width: 1161px)").addEventListener("change", (event) => {
    if (event.matches) closeMenu();
  });
}

document.querySelectorAll("[data-tabs]").forEach((group) => {
  const tabs = [...group.querySelectorAll('[role="tab"]')];
  const select = (tab, focus = false) => {
    tabs.forEach((item) => {
      const selected = item === tab;
      item.setAttribute("aria-selected", String(selected));
      item.tabIndex = selected ? 0 : -1;
      const panel = document.getElementById(item.getAttribute("aria-controls"));
      if (panel) panel.hidden = !selected;
    });
    if (focus) tab.focus();
  };
  const steps = { ArrowDown: 1, ArrowRight: 1, ArrowUp: -1, ArrowLeft: -1 };

  tabs.forEach((tab, index) => {
    tab.addEventListener("click", () => select(tab));
    tab.addEventListener("keydown", (event) => {
      let next;
      if (event.key in steps) next = tabs[(index + steps[event.key] + tabs.length) % tabs.length];
      else if (event.key === "Home") next = tabs[0];
      else if (event.key === "End") next = tabs[tabs.length - 1];
      if (!next) return;
      event.preventDefault();
      select(next, true);
    });
  });
});

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  const label = button.textContent;
  button.addEventListener("click", async () => {
    const target = document.getElementById(button.dataset.copyTarget);
    if (!target) return;
    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      button.textContent = "Copied";
    } catch {
      button.textContent = "Select the text to copy";
    }
    window.setTimeout(() => { button.textContent = label; }, 1600);
  });
});

const crossbarArchitectForm = document.querySelector("#crossbar-architect-form");

if (crossbarArchitectForm) {
  const model = window.CROSSBAR_MODEL;
  const byId = (id) => document.querySelector(`#${id}`);
  const controls = {
    device: byId("xsim-device"),
    bankN: byId("xsim-bank-n"),
    wire: byId("xsim-wire"),
    input: byId("xsim-input"),
    workload: byId("xsim-workload"),
    banks: byId("xsim-banks"),
    training: byId("xsim-training")
  };
  const levelOrder = ["raw", "calibrated", "ir_aware", "chip_in_loop"];
  const statusText = {
    raw: "Accurate without compensation",
    calibrated: "Accurate with calibration",
    ir_aware: "Needs IR-aware mapping",
    chip_in_loop: "Needs chip-in-the-loop tuning"
  };

  function renderCrossbarBanks(physicalBanks, requiredBanks) {
    const bankGrid = byId("xsim-bank-grid");
    const visibleBanks = Math.min(24, physicalBanks);
    const activeBanks = Math.min(requiredBanks, physicalBanks);
    const activeVisible = Math.round(visibleBanks * activeBanks / physicalBanks);
    const fragment = document.createDocumentFragment();

    for (let index = 0; index < visibleBanks; index += 1) {
      const bank = document.createElement("span");
      bank.className = `sim-bank crossbar-bank${index < activeVisible ? " active" : ""}`;
      const verticalRail = document.createElement("span");
      verticalRail.className = "rail-v";
      const horizontalRail = document.createElement("span");
      horizontalRail.className = "rail-h";
      const resistors = document.createElement("em");
      resistors.setAttribute("aria-hidden", "true");
      const label = document.createElement("b");
      label.textContent = `X${String(index + 1).padStart(2, "0")}`;
      bank.append(verticalRail, horizontalRail, resistors, label);
      fragment.append(bank);
    }

    bankGrid.replaceChildren(fragment);
    bankGrid.setAttribute("aria-label", `${activeBanks} of ${physicalBanks} physical crossbar banks active in the current mapping wave`);
    byId("xsim-bank-caption").textContent = visibleBanks < physicalBanks
      ? `Showing ${visibleBanks} representative banks of ${physicalBanks}`
      : `Showing all ${physicalBanks} physical banks`;
  }

  function cell(tag, text, className) {
    const element = document.createElement(tag);
    element.textContent = text;
    if (className) element.className = className;
    return element;
  }

  const isBounded = (level, training) => Boolean(level.bounded && level.bounded[training]);
  const recovers = (level, training) => !isBounded(level, training)
    && model.baseline_accuracy - level.accuracy[training] <= 1;

  function renderLevels(point, training) {
    const rows = levelOrder.map((key) => {
      const level = point.levels[key];
      const accuracy = level.accuracy[training];
      const bounded = isBounded(level, training);
      const loss = model.baseline_accuracy - accuracy;
      const row = document.createElement("tr");
      row.className = recovers(level, training) ? "level-ok" : !bounded && loss <= 5 ? "level-warn" : "level-low";
      const heading = cell("th", model.levels[key]);
      heading.scope = "row";
      const accuracyText = bounded ? "Beyond measured range" : `${accuracy.toFixed(1)}%`;
      row.append(heading, cell("td", `${level.snr_db.toFixed(1)} dB`), cell("td", accuracyText));
      if (bounded) row.title = `Weight noise exceeds the measured curve; accuracy is at most ${accuracy.toFixed(1)}%`;
      return row;
    });
    byId("xsim-levels").replaceChildren(...rows);
  }

  function renderChipFits() {
    const body = byId("xsim-chip-fits");
    if (!body) return;
    const rows = [];
    model.chip_fits.forEach((chip) => {
      const results = [...(chip.fit ? [["Fitted", chip.fit]] : []), ...chip.checks.map((check) => ["Check", check])];
      results.forEach(([role, result], index) => {
        const row = document.createElement("tr");
        if (index === 0) {
          const name = cell("th", chip.chip);
          name.scope = "row";
          name.rowSpan = results.length;
          row.append(name);
        }
        const precision = "predicted_bits" in result;
        const reported = precision ? `${result.reported_bits.toFixed(1)} bits` : `${result.reported_accuracy.toFixed(2)}%`;
        const predicted = precision
          ? `${result.predicted_bits.toFixed(1)} bits (${result.error >= 0 ? "+" : ""}${result.error.toFixed(1)})`
          : `${result.predicted_accuracy.toFixed(2)}% (${result.error >= 0 ? "+" : ""}${result.error.toFixed(2)} pt)`;
        row.append(cell("td", result.observation), cell("td", role), cell("td", reported), cell("td", predicted));
        rows.push(row);
      });
    });
    body.replaceChildren(...rows);
  }

  function evaluateCrossbarArchitecture() {
    const bankN = Number(controls.bankN.value);
    const workload = Number(controls.workload.value);
    const physicalBanks = Number(controls.banks.value);
    const training = controls.training.value;
    const point = model.points.find((entry) => entry.device === controls.device.value
      && entry.wire === controls.wire.value && entry.input === controls.input.value && entry.N === bankN);

    byId("xsim-banks-value").textContent = String(physicalBanks);
    const requiredBanks = Math.ceil(workload / bankN);
    const waves = Math.ceil(requiredBanks / physicalBanks);
    renderCrossbarBanks(physicalBanks, requiredBanks);

    const device = model.devices[controls.device.value];
    byId("xsim-chip-kicker").textContent = `${device.label} · N = ${bankN} · ${model.inputs[controls.input.value].label}`;
    byId("xsim-columns").textContent = String(2 * bankN);
    byId("xsim-waves").textContent = String(waves);
    byId("xsim-gain").textContent = `${Math.max(0, Math.round((1 - point.gain) * 100))}%`;
    renderLevels(point, training);

    const bankText = `${workload.toLocaleString()} input terms partition into ${requiredBanks} local crossbar ${requiredBanks === 1 ? "bank" : "banks"} of logical dimension ${bankN}`;
    byId("xsim-mapping").textContent = `${bankText}, scheduled in ${waves} ${waves === 1 ? "wave" : "waves"}.`;

    const firstGood = levelOrder.find((key) => recovers(point.levels[key], training));
    const status = byId("xsim-status");
    status.textContent = firstGood ? statusText[firstGood] : "Loses accuracy even with tuning";
    status.className = !firstGood ? "status-low" : ["ir_aware", "chip_in_loop"].includes(firstGood) ? "status-warn" : "";

    let advice;
    if (!firstGood) {
      const tuned = point.levels.chip_in_loop;
      const gap = (model.baseline_accuracy - tuned.accuracy[training]).toFixed(1);
      advice = isBounded(tuned, training)
        ? `Even chip-in-the-loop tuning leaves the SNR beyond the measured noise range, at least ${gap} points below the ${model.baseline_accuracy}% baseline. Use a smaller bank, pulse-width inputs, or wider wires.`
        : `Even chip-in-the-loop tuning leaves ${gap} points below the ${model.baseline_accuracy}% baseline. Use a smaller bank, pulse-width inputs, or wider wires.`;
    } else {
      advice = `${model.levels[firstGood]} brings accuracy within one point of the ${model.baseline_accuracy}% baseline.`;
    }
    if (waves > 1) advice += ` The fabric needs ${waves} scheduling waves; add ${requiredBanks - physicalBanks} banks to map it in one.`;
    const adviceElement = byId("xsim-advice");
    adviceElement.replaceChildren(cell("strong", "Design reading: "), document.createTextNode(advice));
  }

  if (model) {
    crossbarArchitectForm.addEventListener("submit", (event) => {
      event.preventDefault();
      evaluateCrossbarArchitecture();
    });
    Object.values(controls).forEach((control) => control.addEventListener("input", evaluateCrossbarArchitecture));
    evaluateCrossbarArchitecture();
    renderChipFits();
  } else {
    byId("xsim-status").textContent = "Model data missing";
    byId("xsim-status").className = "status-low";
  }
}
