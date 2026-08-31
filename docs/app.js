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

const crossbarArchitectForm = document.querySelector("#crossbar-architect-form");

if (crossbarArchitectForm) {
  const controls = {
    device: document.querySelector("#xsim-device"),
    workload: document.querySelector("#xsim-workload"),
    bankN: document.querySelector("#xsim-bank-n"),
    banks: document.querySelector("#xsim-banks")
  };

  const outputs = {
    banks: document.querySelector("#xsim-banks-value")
  };

  const profiles = {
    mram: { name: "MRAM", optimumRs: { 144: 464, 288: 464, 576: 268 } },
    reram: { name: "ReRAM", optimumRs: { 144: 835, 288: 681, 576: 562 } },
    fefet: { name: "FeFET", optimumRs: { 144: 8280, 288: 10980, 576: 10980 } }
  };

  function renderCrossbarBanks(physicalBanks, requiredBanks) {
    const bankGrid = document.querySelector("#xsim-bank-grid");
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
    document.querySelector("#xsim-bank-caption").textContent = visibleBanks < physicalBanks
      ? `Showing ${visibleBanks} representative banks of ${physicalBanks}`
      : `Showing all ${physicalBanks} physical banks`;
  }

  function formatResistance(value) {
    return value >= 1000 ? `${(value / 1000).toFixed(value % 1000 === 0 ? 0 : 2)} kΩ` : `${value} Ω`;
  }

  function evaluateCrossbarArchitecture() {
    const device = controls.device.value;
    const profile = profiles[device];
    const workload = Number(controls.workload.value);
    const bankN = Number(controls.bankN.value);
    const physicalBanks = Number(controls.banks.value);

    outputs.banks.textContent = String(physicalBanks);
    const requiredBanks = Math.ceil(workload / bankN);
    const waves = Math.ceil(requiredBanks / physicalBanks);
    const optimumRs = profile.optimumRs[bankN];

    renderCrossbarBanks(physicalBanks, requiredBanks);
    document.querySelector("#xsim-chip-kicker").textContent = `${profile.name} · N = ${bankN}`;
    document.querySelector("#xsim-rs").textContent = formatResistance(optimumRs);
    document.querySelector("#xsim-columns").textContent = String(2 * bankN);
    document.querySelector("#xsim-waves").textContent = String(waves);
    document.querySelector("#xsim-mapping").textContent = `${workload.toLocaleString()} input terms partition into ${requiredBanks} local crossbar ${requiredBanks === 1 ? "bank" : "banks"} of logical dimension ${bankN}, scheduled in ${waves} ${waves === 1 ? "wave" : "waves"}.`;

    const statusElement = document.querySelector("#xsim-status");
    statusElement.textContent = waves === 1 ? "Fits in one mapping wave" : `${waves} mapping waves`;
    statusElement.className = waves === 1 ? "" : "status-warn";

    let advice;
    if (waves === 1) {
      advice = `Use the paper-selected ${profile.name} sensing point for this evaluated layer size, then combine the ${requiredBanks} converted partial sums digitally.`;
    } else {
      advice = `The local coordinate remains ${formatResistance(optimumRs)}, but this fabric needs ${waves} scheduling waves. Add at least ${requiredBanks - physicalBanks} banks to fit the mapping in one wave.`;
    }

    const adviceElement = document.querySelector("#xsim-advice");
    const adviceLead = document.createElement("strong");
    adviceLead.textContent = "Design reading: ";
    adviceElement.replaceChildren(adviceLead, document.createTextNode(advice));
  }

  crossbarArchitectForm.addEventListener("submit", (event) => {
    event.preventDefault();
    evaluateCrossbarArchitecture();
  });
  Object.values(controls).forEach((control) => control.addEventListener("input", evaluateCrossbarArchitecture));
  evaluateCrossbarArchitecture();
}
