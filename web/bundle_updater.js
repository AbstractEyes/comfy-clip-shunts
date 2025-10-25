// js/bundle_updater.js
/*
import { app, api } from "../../scripts/app.js";

async function fetchLabels() {
  const resp = await api.fetchApi("/abs/bundle_list");
  if (!resp.ok) return [];
  const { labels } = await resp.json();
  return Array.isArray(labels) ? labels : [];
}

function refreshWidget(node, labels) {
  const w = node.widgets?.find(w => w?.name === "bundle_id" && w?.options);
  if (!w) return;

  // Comfy typically uses options.values; some forks use .choices
  if (w.options?.values) w.options.values = labels;
  else if (w.options?.choices) w.options.choices = labels;
  else w.options = { ...(w.options || {}), values: labels };

  if (labels.length && !labels.includes(w.value)) w.value = labels[0];

  node.onWidgetChanged?.(w.name, w.value, "combo");
  app.graph.setDirtyCanvas(true, true);
}

function refreshAllTargets(labels) {
  const TARGETS = new Set(["ABS_LoadEmbedding", "ABS_ShaperEmbedding", "ABS_InspectEmbedding"]);
  app.graph._nodes?.forEach(node => {
    if (TARGETS.has(node?.comfyClass)) refreshWidget(node, labels);
  });
}

app.registerExtension({
  name: "abs.bundle_list_refresher",

  async setup() {
    // If the backend broadcasts a refresh event, update immediately
    api.addEventListener("abs.bundle.refresh", async () => {
      const labels = await fetchLabels();
      if (labels.length) refreshAllTargets(labels);
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    // Populate once when a target node instance is created
    const TARGETS = new Set(["ABS_LoadEmbedding", "ABS_ShaperEmbedding", "ABS_InspectEmbedding"]);
    const cls = nodeType?.comfyClass || nodeData?.name;
    if (!TARGETS.has(cls)) return;

    const orig = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = async function () {
      const r = orig?.apply(this, arguments);
      const labels = await fetchLabels();
      if (labels.length) refreshWidget(this, labels);
      return r;
    };
  },
});
*/