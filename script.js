const translations = {
  en: {
    navOverview: "Overview",
    navMap: "Research map",
    navResources: "Resources",
    journalBadge: "Artificial Intelligence Review · 2026",
    eyebrow: "From model miniaturization to edge deployment",
    heroTitle: "On-device large language models",
    heroSubtitle:
      "A unified survey of model compression and system optimization for private, responsive, and resource-aware LLM intelligence at the edge.",
    authorsMore: "and 17 co-authors",
    readPaper: "Read the paper",
    browseCollection: "Browse the collection",
    acceptedLabel: "Accepted",
    publishedLabel: "Published",
    openFigure: "Open figure",
    fig1Caption:
      "The survey connects model efficiency, system optimization, evaluation, and real-world applications.",
    statPapers: "curated papers",
    statFamilies: "research families",
    statMetrics: "deployment metrics",
    statAccess: "access & community maintained",
    overviewEyebrow: "Why this survey",
    overviewTitle: "On-device intelligence is a full-stack problem.",
    overviewBody:
      "A smaller model is not automatically a deployable model. Useful edge intelligence emerges when compression, runtime, memory, hardware, and application constraints are optimized together.",
    modelSide: "Model-side efficiency",
    modelSideBody: "Reduce parameters, precision, and computation while preserving useful capability.",
    quantization: "Quantization",
    pruning: "Pruning",
    distillation: "Knowledge distillation",
    lowRank: "Low-rank factorization",
    hybrid: "Hybrid compression",
    systemSide: "System-side execution",
    systemSideBody: "Turn compact models into responsive applications under real device constraints.",
    compiler: "Compiler optimization",
    runtime: "Inference frameworks",
    memory: "Memory & KV cache",
    hardware: "Hardware support",
    edgeCloud: "Edge-cloud collaboration",
    mapEyebrow: "The research map",
    mapTitle: "A field moving from isolated techniques to co-design.",
    mapBody:
      "The literature has grown from individual compression methods into a dense ecosystem of algorithms, systems, and deployment targets. Click each original paper figure to inspect it in full resolution.",
    fig2Caption:
      "The model-compression landscape from early distillation and quantization to hybrid 2025 methods.",
    fig3Caption: "A closer view of the fast-moving LLM quantization landscape.",
    livingCollection: "Living collection",
    mapNoteTitle: "The paper is a snapshot. The repository keeps moving.",
    mapNoteBody:
      "The companion Awesome List extends the published survey with new papers, code links, venues, and community contributions.",
    exploreTaxonomy: "Explore the complete taxonomy",
    alemTitle: "Measure what deployment actually costs.",
    alemBody:
      "The survey proposes a shared evaluation lens for comparing methods across devices and workloads. No single number captures whether an on-device LLM is genuinely useful.",
    learnProtocol: "Learn about the protocol",
    accuracy: "Accuracy",
    accuracySmall: "Capability retained",
    latency: "Latency",
    latencySmall: "TTFT & throughput",
    energy: "Energy",
    energySmall: "Power per workload",
    memoryLabel: "Memory",
    memorySmall: "Weights & KV cache",
    insightsEyebrow: "Deployment takeaways",
    insightsTitle: "Three ideas to carry into the next system.",
    insight1Title: "Start with the bottleneck.",
    insight1Body:
      "Choose compression and runtime techniques only after profiling memory bandwidth, compute, latency, and energy on the target device.",
    insight2Title: "Treat KV cache as a subsystem.",
    insight2Body:
      "Long-context deployment shifts pressure from model weights to cache capacity, bandwidth, paging, and scheduling.",
    insight3Title: "Co-design beats local optimum.",
    insight3Body:
      "The best model in isolation may lose once kernels, accelerators, thermal limits, and application quality are considered together.",
    resourcesEyebrow: "Use the work",
    resourcesTitle: "Read, browse, cite, and contribute.",
    resourcesBody: "The journal article provides the synthesis. The repository keeps the map current.",
    officialArticle: "Official article",
    suggestPaper: "Suggest a paper",
    citationTitle: "Cite the survey",
    copy: "Copy",
    copied: "Citation copied",
    footerText: "A community-maintained companion to the published survey.",
    figureLicense: "Paper figures reproduced unchanged from Chen et al. (2026) under"
  },
  zh: {
    navOverview: "概览",
    navMap: "研究图谱",
    navResources: "资源",
    journalBadge: "Artificial Intelligence Review · 2026",
    eyebrow: "从模型小型化到端侧部署",
    heroTitle: "端侧大语言模型",
    heroSubtitle: "一篇贯通模型压缩与系统优化的综述，面向隐私、实时和资源受限场景下的端侧大模型智能。",
    authorsMore: "等 22 位作者",
    readPaper: "阅读正式论文",
    browseCollection: "浏览论文合集",
    acceptedLabel: "接收",
    publishedLabel: "发表",
    openFigure: "查看原图",
    fig1Caption: "论文将模型效率、系统优化、评测框架与真实应用连接为统一的端侧部署图景。",
    statPapers: "篇精选论文",
    statFamilies: "类研究方向",
    statMetrics: "项部署指标",
    statAccess: "开放获取与社区维护",
    overviewEyebrow: "为什么需要这篇综述",
    overviewTitle: "端侧智能是一个全栈问题。",
    overviewBody: "模型更小，并不意味着它就能真正部署。只有把压缩、运行时、内存、硬件和应用约束放在一起优化，端侧智能才真正可用。",
    modelSide: "模型侧效率",
    modelSideBody: "在保留有效能力的同时，降低参数规模、计算精度与总体算力需求。",
    quantization: "量化",
    pruning: "剪枝",
    distillation: "知识蒸馏",
    lowRank: "低秩分解",
    hybrid: "混合压缩",
    systemSide: "系统侧执行",
    systemSideBody: "在真实设备约束下，将紧凑模型转化为低延迟、可持续运行的应用。",
    compiler: "编译器优化",
    runtime: "推理框架",
    memory: "内存与 KV Cache",
    hardware: "硬件支持",
    edgeCloud: "端云协同",
    mapEyebrow: "研究图谱",
    mapTitle: "领域正在从单点技术走向软硬件协同设计。",
    mapBody: "相关研究已经从独立压缩算法扩展为涵盖算法、系统和部署目标的完整生态。点击论文原图可查看完整分辨率。",
    fig2Caption: "从早期蒸馏、量化方法到 2025 年混合技术的模型压缩研究演进。",
    fig3Caption: "快速演进的大模型量化技术时间线。",
    livingCollection: "持续更新",
    mapNoteTitle: "论文记录一个时间截面，仓库持续向前。",
    mapNoteBody: "配套 Awesome List 在正式论文基础上持续补充新论文、代码、发表信息与社区贡献。",
    exploreTaxonomy: "浏览完整分类体系",
    alemTitle: "衡量部署真正付出的代价。",
    alemBody: "论文提出跨设备、跨任务比较方法的统一评测视角。判断一个端侧大模型是否真正可用，不能只看单一指标。",
    learnProtocol: "了解评测协议",
    accuracy: "准确性",
    accuracySmall: "能力保留程度",
    latency: "延迟",
    latencySmall: "首字延迟与吞吐",
    energy: "能耗",
    energySmall: "单位任务功耗",
    memoryLabel: "内存",
    memorySmall: "权重与 KV Cache",
    insightsEyebrow: "部署启示",
    insightsTitle: "构建下一个系统时值得带走的三个观点。",
    insight1Title: "从瓶颈出发。",
    insight1Body: "先在目标设备上分析内存带宽、算力、延迟和能耗瓶颈，再选择压缩与运行时技术。",
    insight2Title: "把 KV Cache 当作独立子系统。",
    insight2Body: "长上下文部署会将压力从模型权重转移到缓存容量、带宽、分页和调度机制。",
    insight3Title: "协同设计优于局部最优。",
    insight3Body: "孤立评测下最好的模型，在加入算子、加速器、温控和应用质量约束后未必仍然最优。",
    resourcesEyebrow: "使用与参与",
    resourcesTitle: "阅读、浏览、引用并共同维护。",
    resourcesBody: "期刊论文提供系统总结，开源仓库让研究图谱保持更新。",
    officialArticle: "正式论文",
    suggestPaper: "推荐新论文",
    citationTitle: "引用本文",
    copy: "复制",
    copied: "引用已复制",
    footerText: "正式综述的社区维护型配套资源。",
    figureLicense: "论文原图保持不变，来自 Chen et al. (2026)，遵循"
  }
};

const root = document.documentElement;
const themeToggle = document.querySelector("#themeToggle");
const languageToggle = document.querySelector("#languageToggle");
const copyButton = document.querySelector("#copyCitation");
const citationText = document.querySelector("#citationText");
const toast = document.querySelector("#toast");
const lightbox = document.querySelector("#lightbox");
const lightboxImage = lightbox.querySelector("img");
const lightboxClose = lightbox.querySelector(".lightbox-close");

const preferredTheme = localStorage.getItem("odllm-theme") ||
  (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
const preferredLanguage = localStorage.getItem("odllm-language") || "en";

function setTheme(theme) {
  root.dataset.theme = theme;
  localStorage.setItem("odllm-theme", theme);
  themeToggle.textContent = theme === "dark" ? "☀" : "◐";
  themeToggle.setAttribute("aria-label", theme === "dark" ? "Use light theme" : "Use dark theme");
}

function setLanguage(language) {
  const dictionary = translations[language];
  document.querySelectorAll("[data-i18n]").forEach((element) => {
    const value = dictionary[element.dataset.i18n];
    if (value) element.textContent = value;
  });
  root.lang = language === "zh" ? "zh-CN" : "en";
  languageToggle.textContent = language === "en" ? "中文" : "EN";
  localStorage.setItem("odllm-language", language);
}

themeToggle.addEventListener("click", () => {
  setTheme(root.dataset.theme === "dark" ? "light" : "dark");
});

languageToggle.addEventListener("click", () => {
  setLanguage(root.lang.startsWith("zh") ? "en" : "zh");
});

copyButton.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(citationText.textContent.trim());
    toast.classList.add("show");
    window.setTimeout(() => toast.classList.remove("show"), 1800);
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(citationText);
    selection.removeAllRanges();
    selection.addRange(range);
  }
});

document.querySelectorAll("[data-lightbox]").forEach((button) => {
  button.addEventListener("click", () => {
    lightboxImage.src = button.dataset.lightbox;
    lightbox.showModal();
  });
});

lightboxClose.addEventListener("click", () => lightbox.close());
lightbox.addEventListener("click", (event) => {
  if (event.target === lightbox) lightbox.close();
});

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("visible");
        observer.unobserve(entry.target);
      }
    });
  },
  { threshold: 0.12 }
);

document.querySelectorAll(".reveal").forEach((element) => observer.observe(element));

setTheme(preferredTheme);
setLanguage(preferredLanguage);
