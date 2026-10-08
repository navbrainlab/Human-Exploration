import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const SKILL_DIR = "C:/Users/DELL/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations";
const workspaceDir = "C:/Users/DELL/Desktop/SHJ datasets";
const sourcePath = path.join(workspaceDir, "figureA_task_design_overview_v2/figureA_task_design_overview_v2_editable_rev4.pptx");
const buildDir = path.join(workspaceDir, ".codex-pptx-build-20260923");
const finalPath = path.join(workspaceDir, "figureA_task_design_overview_v2/final/figureA_task_design_overview_publication_v9.pptx");
const candidatePath = path.join(buildDir, "candidate_publication_v9.pptx");
const previewPath = path.join(buildDir, "figureA_publication_v9_preview.png");
const layoutPath = path.join(buildDir, "figureA_publication_v9.layout.json");
const receiptPath = path.join(buildDir, "figureA_publication_v9.validation.json");
const { finalizePresentation } = await import(pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href);

const W = 1600;
const H = 720;
const FONT = "Arial";
const C = {
  ink: "#20252A", mid: "#58636D", light: "#D5DCE2", faint: "#EDF1F3",
  shj: "#68489B", grid: "#2D6DA3", masc: "#14898A", build: "#C45C32",
  blue: "#3A74B8", green: "#38865A", red: "#C6534E", white: "#FFFFFF",
};

const deck = Presentation.create({ slideSize: { width: W, height: H } });
const slide = deck.slides.add();
slide.background.fill = C.white;

function fill(v) { return v === "none" ? "none" : { type: "solid", color: v }; }
function shape(geometry, x, y, w, h, opts = {}) {
  const s = slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill: fill(opts.fill ?? "none"),
    line: opts.line ? { style: "solid", fill: opts.line, width: opts.lineWidth ?? 1 } : { style: "solid", fill: "none", width: 0 },
  });
  if (opts.text !== undefined) {
    s.text = opts.text;
    s.text.style = {
      typeface: opts.typeface ?? FONT,
      fontSize: opts.fontSize ?? 18,
      bold: opts.bold ?? false,
      color: opts.color ?? C.ink,
      alignment: opts.align ?? "left",
      verticalAlignment: opts.valign ?? "middle",
      autoFit: "none",
      wrap: "square",
      insets: opts.insets ?? { left: 1, right: 1, top: 0, bottom: 0 },
    };
  }
  return s;
}
function text(t, x, y, w, h, opts = {}) { return shape("textbox", x, y, w, h, { ...opts, text: t }); }
function rect(x, y, w, h, opts = {}) { return shape("rect", x, y, w, h, opts); }
function ellipse(x, y, w, h, opts = {}) { return shape("ellipse", x, y, w, h, opts); }
function rule(x, y, w, h, color = C.light, width = 1) { return shape("line", x, y, w, h, { line: color, lineWidth: width }); }

function eyeTag(x, y, accent) {
  ellipse(x, y + 5, 23, 12, { fill: C.white, line: accent, lineWidth: 1.4 });
  ellipse(x + 9, y + 8, 6, 6, { fill: accent });
  text("EYE TRACKING", x + 29, y, 115, 23, { fontSize: 14, bold: true, color: C.mid });
}
function panelBase(x, title, accent, descriptor, eye = false) {
  text(title, x, 27, 340, 38, { fontSize: 27, bold: true });
  if (eye) eyeTag(x + 202, 42, accent);
  rect(x, 74, 350, 3, { fill: accent });
  text(descriptor, x, 122, 350, 42, { fontSize: 18, bold: true, align: "center" });
  rule(x, 415, 350, 0, C.light, 1.2);
  text("RECORDED BEHAVIOR", x, 430, 350, 23, { fontSize: 16, bold: true, color: accent });
  rule(x, 529, 350, 0, C.light, 1.2);
  rule(x, 592, 350, 0, C.light, 1.2);
}
function bottomRow(x, y, label, value, accent) {
  text(label, x, y, 111, 28, { fontSize: 16, bold: true, color: accent });
  text(value, x + 112, y - 1, 238, 32, { fontSize: 18, bold: true });
}
function chosenSymbol(x, cy, symbol, selected, accent) {
  ellipse(x, cy - 24, 48, 48, {
    fill: selected ? "#F1ECF8" : C.white,
    line: selected ? accent : C.light,
    lineWidth: selected ? 2 : 1.2,
  });
  text(symbol, x, cy - 27, 48, 53, {
    fontSize: 27, bold: true, align: "center", typeface: "Times New Roman",
    color: selected ? accent : C.ink,
  });
}
function shjPanel(x) {
  panelBase(x, "SHJ", C.shj, "Category learning: 3 binary features", true);
  text("One value from each row forms a stimulus", x + 8, 163, 334, 23, { fontSize: 16, color: C.mid, align: "center" });
  const rows = [
    { d: "D1", v: ["$", "¢"], selected: 0 },
    { d: "D2", v: ["?", "!"], selected: 1 },
    { d: "D3", v: ["+", "−"], selected: 0 },
  ];
  rows.forEach((r, i) => {
    const cy = 213 + i * 60;
    text(r.d, x + 9, cy - 13, 42, 27, { fontSize: 17, bold: true, color: C.shj });
    chosenSymbol(x + 57, cy, r.v[0], r.selected === 0, C.shj);
    chosenSymbol(x + 119, cy, r.v[1], r.selected === 1, C.shj);
  });
  text("→", x + 195, 253, 43, 32, { fontSize: 26, color: C.shj, align: "center" });
  rect(x + 244, 190, 89, 155, { fill: C.white, line: C.shj, lineWidth: 1.5 });
  text("!", x + 273, 200, 30, 42, { fontSize: 29, bold: true, typeface: "Times New Roman", align: "center" });
  text("+", x + 253, 283, 30, 42, { fontSize: 29, bold: true, typeface: "Times New Roman", align: "center" });
  text("$", x + 295, 283, 30, 42, { fontSize: 29, bold: true, typeface: "Times New Roman", align: "center" });
  text("example stimulus", x + 230, 349, 120, 24, { fontSize: 14, color: C.mid, align: "center" });
  text("2 × 2 × 2 = 8 stimuli", x, 382, 350, 25, { fontSize: 18, color: C.mid, align: "center" });
  text("Gaze allocated to each feature\ndimension within a trial", x, 461, 350, 60, { fontSize: 20 });
  bottomRow(x, 548, "RESPONSE", "Classify stimulus", C.shj);
  bottomRow(x, 609, "FEEDBACK", "Correct / incorrect", C.shj);
}

function miniGrid(x, y, accent, pointA, pointB) {
  const n = 5, cell = 26, width = n * cell;
  rect(x, y, width, width, { fill: C.white, line: C.light, lineWidth: 1 });
  for (let k = 1; k < n; k++) {
    rule(x + k * cell, y, 0, width, C.light, 1);
    rule(x, y + k * cell, width, 0, C.light, 1);
  }
  const p = ([col, row]) => [x + col * cell + cell / 2, y + row * cell + cell / 2];
  const [aX, aY] = p(pointA), [bX, bY] = p(pointB);
  rule(Math.min(aX, bX), Math.min(aY, bY), Math.abs(bX - aX), Math.abs(bY - aY), accent, 3);
  [ [aX, aY], [bX, bY] ].forEach(([px, py]) => ellipse(px - 8, py - 8, 16, 16, { fill: accent, line: C.white, lineWidth: 1.5 }));
}
function gridPanel(x) {
  panelBase(x, "2D grid", C.grid, "Spatial exploration: 11 × 11 locations");
  text("Same row or column", x + 5, 169, 160, 24, { fontSize: 16, bold: true, align: "center" });
  text("Nearby locations", x + 185, 169, 160, 24, { fontSize: 16, bold: true, align: "center" });
  miniGrid(x + 20, 206, C.grid, [0, 2], [4, 2]);
  miniGrid(x + 200, 206, C.grid, [1, 1], [2, 2]);
  text("axis aligned", x + 10, 342, 150, 24, { fontSize: 16, color: C.mid, align: "center" });
  text("local step", x + 190, 342, 150, 24, { fontSize: 16, color: C.mid, align: "center" });
  text("The two patterns can overlap", x, 382, 350, 25, { fontSize: 17, color: C.mid, align: "center" });
  text("Sequence of sampled locations\nand their spatial relations", x, 461, 350, 60, { fontSize: 20 });
  bottomRow(x, 548, "RESPONSE", "Sample a location", C.grid);
  bottomRow(x, 609, "FEEDBACK", "Numeric reward", C.grid);
}

function dot(cx, cy, color) { ellipse(cx - 7, cy - 7, 14, 14, { fill: color, line: C.white, lineWidth: 1.4 }); }
function mascPanel(x) {
  panelBase(x, "MASC", C.masc, "Choice: 2 options × 3 attributes", true);
  text("Phone trial shown", x + 5, 167, 150, 25, { fontSize: 17, bold: true });
  dot(x + 247, 180, C.blue);
  text("A", x + 259, 167, 24, 25, { fontSize: 17, bold: true, color: C.blue });
  dot(x + 298, 180, C.green);
  text("B", x + 310, 167, 24, 25, { fontSize: 17, bold: true, color: C.green });
  text("schematic attribute values", x + 130, 195, 212, 20, { fontSize: 14, color: C.mid, align: "center" });
  const items = [
    ["Battery", 0.28, 0.77],
    ["Display", 0.66, 0.35],
    ["Storage", 0.42, 0.84],
  ];
  items.forEach(([label, a, b], i) => {
    const yy = 224 + i * 54;
    text(label, x + 8, yy - 16, 111, 31, { fontSize: 18, bold: true });
    rule(x + 136, yy, 196, 0, C.light, 2);
    dot(x + 136 + 196 * a, yy, C.blue);
    dot(x + 136 + 196 * b, yy, C.green);
  });
  text("Separate hotel trials: stars, distance, room size", x + 5, 382, 340, 25, { fontSize: 16, color: C.mid, align: "center" });
  text("Fixations to option × attribute\nlocations within each trial", x, 461, 350, 60, { fontSize: 20 });
  bottomRow(x, 548, "RESPONSE", "Rate, then choose", C.masc);
  bottomRow(x, 609, "FEEDBACK", "No trial feedback", C.masc);
}

function smallSquare(cx, cy, color) { rect(cx - 16, cy - 16, 32, 32, { fill: color, line: C.light, lineWidth: 0.8 }); }
function textureTile(cx, cy, style) {
  rect(cx - 16, cy - 16, 32, 32, { fill: C.white, line: C.light, lineWidth: 0.9 });
  if (style === 0) {
    for (let k = -8; k <= 8; k += 8) {
      rule(cx + k, cy - 15, 0, 30, C.ink, 1.6);
      rule(cx - 15, cy + k, 30, 0, C.ink, 1.6);
    }
  } else if (style === 1) {
    for (let dx of [-8, 0, 8]) for (let dy of [-8, 0, 8]) ellipse(cx + dx - 1.5, cy + dy - 1.5, 3, 3, { fill: C.ink });
  } else {
    text("≈", cx - 15, cy - 18, 30, 34, { fontSize: 30, color: C.ink, align: "center" });
  }
}
function buildPanel(x) {
  panelBase(x, "Build-an-Icon", C.build, "Value learning: color, shape, texture");
  text("Select 0–3 features; others fill at random", x + 4, 163, 342, 25, { fontSize: 16, color: C.mid, align: "center" });
  const xs = [x + 134, x + 187, x + 240];
  const ys = [220, 278, 336];
  ["Color", "Shape", "Texture"].forEach((label, i) => text(label, x + 5, ys[i] - 14, 105, 28, { fontSize: 17, bold: true }));
  [C.blue, C.red, C.green].forEach((color, i) => smallSquare(xs[i], ys[0], color));
  rect(xs[0] - 16, ys[1] - 16, 32, 32, { fill: C.white, line: C.ink, lineWidth: 1.5 });
  ellipse(xs[1] - 16, ys[1] - 16, 32, 32, { fill: C.white, line: C.ink, lineWidth: 1.5 });
  text("△", xs[2] - 19, ys[1] - 25, 38, 47, { fontSize: 40, align: "center" });
  [0, 1, 2].forEach((i) => textureTile(xs[i], ys[2], i));
  ys.forEach((yy) => rect(xs[1] - 21, yy - 21, 42, 42, { fill: "none", line: C.build, lineWidth: 2.2 }));
  text("selected", x + 146, 363, 82, 22, { fontSize: 16, color: C.build, align: "center" });
  text("→", x + 258, 256, 28, 32, { fontSize: 24, color: C.build, align: "center" });
  ellipse(x + 288, 249, 60, 60, { fill: "#E89A95", line: C.red, lineWidth: 1.3 });
  for (let dx of [-11, 0, 11]) for (let dy of [-10, 0, 10]) ellipse(x + 318 + dx - 1.5, 279 + dy - 1.5, 3, 3, { fill: C.ink });
  text("icon", x + 289, 314, 60, 24, { fontSize: 15, color: C.mid, align: "center" });
  text("3 × 3 × 3 = 27 possible icons", x, 382, 350, 25, { fontSize: 17, color: C.mid, align: "center" });
  text("Actively selected features\nacross successive trials", x, 461, 350, 60, { fontSize: 20 });
  bottomRow(x, 548, "RESPONSE", "Submit an icon", C.build);
  bottomRow(x, 609, "FEEDBACK", "Probabilistic 0 / 1", C.build);
}

text("A", 13, 13, 33, 44, { fontSize: 35, bold: true });
text("Task spaces and observations for dimensional exploration", 50, 86, 1500, 29,
  { fontSize: 18, bold: true, color: C.mid, align: "center" });
const xs = [50, 430, 810, 1190];
shjPanel(xs[0]);
gridPanel(xs[1]);
mascPanel(xs[2]);
buildPanel(xs[3]);
[410, 790, 1170].forEach((x) => {
  rule(x, 24, 0, 53, C.light, 1);
  rule(x, 119, 0, 540, C.light, 1);
});
rule(50, 663, 1470, 0, C.light, 1);
text("Schematics are illustrative; attribute positions are not measured values. Measures use different observation units across tasks.",
  50, 674, 1470, 28, { fontSize: 17, color: C.mid, align: "center" });

slide.speakerNotes.textFrame.setText(
  "Figure A | Task spaces and observations. SHJ category-learning stimuli combine one value from each of three spatially separated binary feature dimensions. The eye-tracking export summarizes gaze by dimension within a trial and does not contain ordered fixations. In the 2D grid task, participants select locations from an 11 × 11 grid and receive numeric rewards. The two sketches illustrate an axis-aligned relation and a nearby relation; the categories can overlap. MASC phone and hotel trials are separate domains, each involving two options described by three attributes. The phone example is shown; hotel attributes are stars, distance and room size. Colored marker positions are schematic, not observed attribute values. Fixations are mapped to option-by-attribute regions. Participants rate options and choose; no trial-by-trial outcome feedback is given. In Build-an-Icon, three values each of color, shape and texture yield 27 possible icons. The illustrated selected values combine into the completed icon. Participants can also leave dimensions for random fill. The record used to study exploration is the set of actively selected features across trials; feedback is a probabilistic binary reward. All task illustrations are schematic and are not participant trajectories."
);

await fs.mkdir(buildDir, { recursive: true });
await fs.mkdir(path.dirname(finalPath), { recursive: true });
await (await PresentationFile.exportPptx(deck)).save(candidatePath);
const png = await deck.export({ slide, format: "png", scale: 1 });
await fs.writeFile(previewPath, new Uint8Array(await png.arrayBuffer()));
const layout = await slide.export({ format: "layout" });
await fs.writeFile(layoutPath, await layout.text());
const result = await finalizePresentation({
  explicitTotalSlideCount: 1,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  workspaceDir,
  candidatePath,
  finalPath,
  pythonExecutable: "C:/Users/DELL/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "15240000,6858000", "--validate-heading-fit"],
  fontPolicy: {
    basis: "reference", families: ["Arial", "Times New Roman"],
    referencePath: sourcePath,
    referenceSha256: "27af173f3140575671888b5e535eb55540bae3069989eb7b744b2a09bfeebe80",
  },
  verifyArtifactToolImport: true,
  receiptPath,
});
console.log(JSON.stringify({ finalPath, previewPath, receiptPath, result }, null, 2));
