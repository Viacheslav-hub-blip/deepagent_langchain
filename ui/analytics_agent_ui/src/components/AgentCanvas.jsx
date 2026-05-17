/**
 * Canvas графа задач агента.
 *
 * Содержит:
 * - clampZoom: ограничивает масштаб графа.
 * - rowSort: сортирует строки layout.
 * - rowTitle: возвращает заголовок строки графа.
 * - rowHint: возвращает подсказку строки графа.
 * - buildGraphLayout: рассчитывает позиции узлов и связи.
 * - buildActiveBranch: выделяет активную ветку узлов.
 * - calculateFitZoom: рассчитывает масштаб по размеру контейнера.
 * - captureViewportCenter: запоминает центр текущего scroll viewport.
 * - restoreViewportCenter: восстанавливает центр viewport после изменения масштаба.
 * - centerGraph: центрирует граф в scroll-контейнере.
 * - AgentCanvas: отображает интерактивный граф выполнения агента.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Boxes, Fullscreen, Loader2, Maximize2, Minimize2, Minus, Plus, RotateCcw } from "lucide-react";
import { BranchPopover } from "./BranchPopover.jsx";
import { NodeCard } from "./NodeCard.jsx";

const CARD_WIDTH = 280;
const CARD_HEIGHT = 132;
const MIN_GRAPH_WIDTH = 760;
const H_GAP = 64;
const V_GAP = 72;
const PAD_X = 64;
const PAD_Y = 58;
const BRANCH_POPOVER_WIDTH = 340;
const BRANCH_POPOVER_HEIGHT = 360;

const MIN_ZOOM = 0.3;
const MAX_ZOOM = 1.65;
const ZOOM_STEP = 0.1;

function clampZoom(value) {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, Number(value.toFixed(3))));
}

function rowSort(a, b) {
  return a - b;
}

function rowTitle(rowNodes) {
  const first = rowNodes[0]?.node;
  const phase = first?.layout_phase || "";
  const iteration = first?.iteration;

  if (phase === "start") return "Запуск исследования";
  if (phase === "context") return "Сбор контекста";
  if (phase === "plan") return iteration && iteration > 1 ? `Планирование #${iteration}` : "Планирование";
  if (phase === "tasks") return `Задачи плана #${iteration || "?"}`;
  if (phase === "task_contracts") return `Постановка задач #${iteration || "?"}`;
  if (phase === "task_results") return `Результаты задач #${iteration || "?"}`;
  if (phase === "final") return "Финальный отчёт";

  return first?.layout_label || "Этап";
}

function rowHint(rowNodes) {
  const first = rowNodes[0]?.node;
  const phase = first?.layout_phase || "";

  if (phase === "tasks" || phase === "task_contracts") {
    return `${rowNodes.length} постановок`;
  }

  if (phase === "task_results") {
    return `${rowNodes.length} результатов`;
  }

  if (phase === "plan") {
    const rawCount = rowNodes.reduce((sum, item) => sum + (item.node.raw_event_count || 0), 0);
    return `${rawCount} событий планирования`;
  }

  if (phase === "context") {
    const rawCount = rowNodes.reduce((sum, item) => sum + (item.node.raw_event_count || 0), 0);
    return `${rawCount} context events`;
  }

  return `${rowNodes.length} блок`;
}

function buildGraphLayout(nodes) {
  if (!nodes.length) {
    return { placedNodes: [], edges: [], rows: [], width: 900, height: 560 };
  }

  const indexById = new Map(nodes.map((node, index) => [node.node_id, index]));
  const groups = new Map();

  for (const node of nodes) {
    const row = Number.isFinite(node.layout_row) ? node.layout_row : indexById.get(node.node_id);
    if (!groups.has(row)) {
      groups.set(row, []);
    }
    groups.get(row).push(node);
  }

  const sortedRows = Array.from(groups.keys()).sort(rowSort);
  const maxColumns = Math.max(...Array.from(groups.values()).map((group) => group.length));
  const maxRowWidth = maxColumns * CARD_WIDTH + Math.max(0, maxColumns - 1) * H_GAP;

  // Graph width is based only on node content. Row labels are overlays,
  // so they no longer push the graph to the right.
  const width = Math.max(MIN_GRAPH_WIDTH, PAD_X * 2 + maxRowWidth);
  const height = PAD_Y * 2 + sortedRows.length * CARD_HEIGHT + Math.max(0, sortedRows.length - 1) * V_GAP;

  const positions = new Map();
  const placedNodes = [];
  const rows = [];

  sortedRows.forEach((rowNumber, visualRowIndex) => {
    const group = groups.get(rowNumber).sort((a, b) => indexById.get(a.node_id) - indexById.get(b.node_id));
    const rowWidth = group.length * CARD_WIDTH + Math.max(0, group.length - 1) * H_GAP;
    const startX = (width - rowWidth) / 2;
    const y = PAD_Y + visualRowIndex * (CARD_HEIGHT + V_GAP);

    const rowPlaced = [];

    group.forEach((node, columnIndex) => {
      const x = startX + columnIndex * (CARD_WIDTH + H_GAP);
      const placedNode = {
        node,
        x,
        y,
        width: CARD_WIDTH,
        height: CARD_HEIGHT,
        visualRowIndex,
        sourceRow: rowNumber,
      };

      positions.set(node.node_id, placedNode);
      placedNodes.push(placedNode);
      rowPlaced.push(placedNode);
    });

    rows.push({
      key: `row-${rowNumber}`,
      sourceRow: rowNumber,
      visualRowIndex,
      y,
      height: CARD_HEIGHT,
      title: rowTitle(rowPlaced),
      hint: rowHint(rowPlaced),
      phase: rowPlaced[0]?.node.layout_phase || "default",
    });
  });

  const edges = [];

  for (const node of nodes) {
    const to = positions.get(node.node_id);
    if (!to) continue;

    for (const parentId of node.parent_ids || []) {
      const from = positions.get(parentId);
      if (!from) continue;

      const startX = from.x + from.width / 2;
      const startY = from.y + from.height;
      const endX = to.x + to.width / 2;
      const endY = to.y;
      const delta = Math.max(38, (endY - startY) / 2);

      edges.push({
        key: `${parentId}-${node.node_id}`,
        parentId,
        childId: node.node_id,
        d: `M ${startX} ${startY} C ${startX} ${startY + delta}, ${endX} ${endY - delta}, ${endX} ${endY}`,
      });
    }
  }

  return { placedNodes, edges, rows, width, height, positions };
}

function buildActiveBranch(nodes, activeNodeId) {
  const nodeById = new Map(nodes.map((node) => [node.node_id, node]));
  const branchNodeIds = new Set();
  const branchEdgeKeys = new Set();

  function visit(nodeId) {
    if (!nodeId || branchNodeIds.has(nodeId)) {
      return;
    }

    const node = nodeById.get(nodeId);
    if (!node) {
      return;
    }

    branchNodeIds.add(nodeId);

    for (const parentId of node.parent_ids || []) {
      if (!nodeById.has(parentId)) {
        continue;
      }
      branchEdgeKeys.add(`${parentId}-${nodeId}`);
      visit(parentId);
    }
  }

  visit(activeNodeId);

  return { branchNodeIds, branchEdgeKeys };
}

function calculateFitZoom(container, width, height) {
  if (!container || !width || !height) {
    return 1;
  }

  const availableWidth = Math.max(320, container.clientWidth - 36);
  const availableHeight = Math.max(360, container.clientHeight - 36);
  const zoom = Math.min(1, availableWidth / width, availableHeight / height);

  return clampZoom(zoom);
}

/**
 * Запоминает центр видимой области как долю scroll-содержимого.
 *
 * @param {HTMLElement | null} container Scroll-контейнер графа.
 * @returns {object} Относительная позиция центра viewport.
 */
function captureViewportCenter(container) {
  if (!container || !container.scrollWidth || !container.scrollHeight) {
    return { x: 0.5, y: 0.5 };
  }

  return {
    x: (container.scrollLeft + container.clientWidth / 2) / container.scrollWidth,
    y: (container.scrollTop + container.clientHeight / 2) / container.scrollHeight,
  };
}

/**
 * Восстанавливает центр видимой области после пересчета размеров графа.
 *
 * @param {HTMLElement | null} container Scroll-контейнер графа.
 * @param {object} center Относительная позиция центра viewport.
 * @returns {void}
 */
function restoreViewportCenter(container, center) {
  if (!container) {
    return;
  }

  const left = Math.max(0, center.x * container.scrollWidth - container.clientWidth / 2);
  const top = Math.max(0, center.y * container.scrollHeight - container.clientHeight / 2);

  container.scrollTo({
    left,
    top,
    behavior: "auto",
  });
}

function centerGraph(container) {
  if (!container) {
    return;
  }

  const left = Math.max(0, (container.scrollWidth - container.clientWidth) / 2);
  const top = Math.max(0, (container.scrollHeight - container.clientHeight) / 2);

  container.scrollTo({
    left,
    top,
    behavior: "smooth",
  });
}

/**
 * Отображает масштабируемый граф выполнения агента.
 *
 * @param {object} props Свойства компонента.
 * @param {Array} props.nodes Узлы пользовательского графа.
 * @param {string} props.selectedNodeId Идентификатор выбранного узла.
 * @param {Function} props.onSelectNode Обработчик выбора узла.
 * @param {Function | undefined} props.onCreateBranch Необязательный обработчик branch.
 * @param {string} props.branchingNodeId Идентификатор узла с активным branch.
 * @param {string} props.branchError Ошибка создания branch.
 * @param {string} props.phase Текущая фаза run.
 * @returns {JSX.Element} Canvas графа задач.
 */
export function AgentCanvas({ nodes, selectedNodeId, onSelectNode, onCreateBranch, branchingNodeId, branchError, phase }) {
  const scrollRef = useRef(null);
  const nodeRefs = useRef(new Map());
  const zoomModeRef = useRef("fit");
  const [zoom, setZoom] = useState(1);
  const [branchTargetId, setBranchTargetId] = useState("");
  const [graphFullscreen, setGraphFullscreen] = useState(false);

  const latestNodeId = nodes.at(-1)?.node_id || "";
  const activeNodeId = latestNodeId || selectedNodeId || "";

  const { placedNodes, edges, rows, width, height } = useMemo(
    () => buildGraphLayout(nodes),
    [nodes]
  );

  const { branchNodeIds, branchEdgeKeys } = useMemo(
    () => buildActiveBranch(nodes, activeNodeId),
    [nodes, activeNodeId]
  );

  const branchTarget = useMemo(
    () => placedNodes.find((item) => item.node.node_id === branchTargetId) || null,
    [placedNodes, branchTargetId]
  );

  const branchPopoverPosition = useMemo(() => {
    if (!branchTarget) {
      return null;
    }

    const preferredRight = branchTarget.x + branchTarget.width + 18;
    const hasRightSpace = preferredRight + BRANCH_POPOVER_WIDTH < width - 24;
    const x = hasRightSpace
      ? preferredRight
      : Math.max(24, branchTarget.x - BRANCH_POPOVER_WIDTH - 18);
    const y = Math.min(
      Math.max(24, branchTarget.y),
      Math.max(24, height - BRANCH_POPOVER_HEIGHT - 24)
    );

    return { x, y };
  }, [branchTarget, width, height]);

  const handleNodeClick = (nodeId) => {
    setBranchTargetId("");
    onSelectNode(nodeId);
  };

  const handleBranchClick = (nodeId) => {
    if (!onCreateBranch) {
      return;
    }
    setBranchTargetId(nodeId);
    onSelectNode(nodeId);
  };

  useEffect(() => {
    if (!branchTargetId) {
      return;
    }

    const stillExists = nodes.some((node) => node.node_id === branchTargetId);
    if (!stillExists) {
      setBranchTargetId("");
    }
  }, [nodes, branchTargetId]);

  const applyZoom = (nextZoom, mode = "manual") => {
    const container = scrollRef.current;
    const center = captureViewportCenter(container);
    zoomModeRef.current = mode;
    setZoom((currentZoom) => {
      const resolvedZoom = typeof nextZoom === "function" ? nextZoom(currentZoom) : nextZoom;
      return clampZoom(resolvedZoom);
    });

    window.requestAnimationFrame(() => {
      restoreViewportCenter(container, center);
    });
  };

  const fitGraph = () => {
    const nextZoom = calculateFitZoom(scrollRef.current, width, height);
    applyZoom(nextZoom, "fit");
  };

  const zoomIn = () => applyZoom((currentZoom) => currentZoom + ZOOM_STEP);
  const zoomOut = () => applyZoom((currentZoom) => currentZoom - ZOOM_STEP);
  const resetZoom = () => applyZoom(1);

  useEffect(() => {
    if (zoomModeRef.current !== "fit") {
      return;
    }

    fitGraph();

    const element = scrollRef.current;
    if (!element || typeof ResizeObserver === "undefined") {
      return;
    }

    const observer = new ResizeObserver(() => {
      if (zoomModeRef.current === "fit") {
        fitGraph();
      }
    });
    observer.observe(element);

    return () => observer.disconnect();
  }, [width, height, placedNodes.length]);

  useEffect(() => {
    const frameId = window.requestAnimationFrame(() => {
      if (zoomModeRef.current === "fit") {
        fitGraph();
      } else {
        centerGraph(scrollRef.current);
      }
    });

    return () => window.cancelAnimationFrame(frameId);
  }, [placedNodes.length, activeNodeId]);

  useEffect(() => {
    if (!graphFullscreen) {
      document.body.style.overflow = "";
      return;
    }
    document.body.style.overflow = "hidden";
    const onKey = (event) => {
      if (event.key === "Escape") {
        setGraphFullscreen(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = "";
      window.removeEventListener("keydown", onKey);
    };
  }, [graphFullscreen]);

  useEffect(() => {
    if (!graphFullscreen || !nodes.length) {
      return;
    }
    const id = window.requestAnimationFrame(() => {
      if (zoomModeRef.current === "fit") {
        fitGraph();
      }
    });
    return () => window.cancelAnimationFrame(id);
  }, [graphFullscreen, nodes.length, width, height]);

  function toggleGraphFullscreen() {
    setGraphFullscreen((value) => !value);
    window.requestAnimationFrame(() => {
      if (zoomModeRef.current === "fit") {
        fitGraph();
      } else {
        centerGraph(scrollRef.current);
      }
    });
  }

  return (
    <section
      className={[
        "canvas-panel",
        "canvas-panel--graph",
        graphFullscreen ? "canvas-panel--graph-fullscreen" : "",
      ].filter(Boolean).join(" ")}
    >
      <div className="canvas-backdrop">
        <div className="canvas-grid" />
      </div>

      <div className="canvas-header">
        <div>
          <div className="section-label">Execution canvas</div>
          <h2>Agent graph</h2>
        </div>

        <div className="canvas-header-actions canvas-header-actions--zoom">
          <button
            type="button"
            className="canvas-mini-button"
            onClick={zoomOut}
            disabled={!nodes.length || zoom <= MIN_ZOOM}
            title="Отдалить граф внутри canvas"
          >
            <Minus size={14} />
          </button>

          <div className="zoom-readout">{Math.round(zoom * 100)}%</div>

          <button
            type="button"
            className="canvas-mini-button"
            onClick={zoomIn}
            disabled={!nodes.length || zoom >= MAX_ZOOM}
            title="Приблизить граф внутри canvas"
          >
            <Plus size={14} />
          </button>

          <button
            type="button"
            className="canvas-mini-button"
            onClick={fitGraph}
            disabled={!nodes.length}
            title="Показать весь граф"
          >
            <Maximize2 size={14} />
            Fit
          </button>

          <button
            type="button"
            className="canvas-mini-button"
            onClick={toggleGraphFullscreen}
            disabled={!nodes.length}
            title={graphFullscreen ? "Свернуть граф (Esc)" : "Открыть граф на весь экран"}
          >
            {graphFullscreen ? <Minimize2 size={14} /> : <Fullscreen size={14} />}
            {graphFullscreen ? "Свернуть" : "Экран"}
          </button>

          <button
            type="button"
            className="canvas-mini-button"
            onClick={resetZoom}
            disabled={!nodes.length}
            title="Вернуть масштаб 100%"
          >
            <RotateCcw size={14} />
            100%
          </button>

          <div className="canvas-counter">{nodes.length} steps</div>
        </div>
      </div>

      {!nodes.length ? (
        <div className="empty-canvas">
          {phase === "starting" || phase === "running" ? <Loader2 className="spin" size={26} /> : <Boxes size={26} />}
          <strong>Ожидаю первые узлы</strong>
          <span>Когда агент начнет писать lineage, здесь появится сценарный граф сверху вниз.</span>
        </div>
      ) : (
        <div ref={scrollRef} className="graph-scroll-area graph-scroll-area--fit graph-scroll-area--centered">
          <div
            className="graph-zoom-shell graph-zoom-shell--centered"
            style={{
              width: `${width * zoom}px`,
              height: `${height * zoom}px`,
            }}
          >
            <div
              className="graph-stage graph-stage--vertical graph-stage--phase-rows"
              style={{
                width: `${width}px`,
                height: `${height}px`,
                transform: `scale(${zoom})`,
              }}
            >
              {rows.map((row) => (
                <div
                  key={row.key}
                  className={`graph-row-band graph-row-band--${row.phase}`}
                  style={{ top: `${row.y - 18}px`, height: `${row.height + 36}px` }}
                >
                  <div className="graph-row-label">
                    <strong>{row.title}</strong>
                  </div>
                </div>
              ))}

              <svg className="graph-lines" width={width} height={height} viewBox={`0 0 ${width} ${height}`} aria-hidden="true">
                <defs>
                  <filter id="lineGlow">
                    <feGaussianBlur stdDeviation="3.5" result="blur" />
                    <feMerge>
                      <feMergeNode in="blur" />
                      <feMergeNode in="SourceGraphic" />
                    </feMerge>
                  </filter>
                  <linearGradient id="activeBranchGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#9ca3af" />
                    <stop offset="55%" stopColor="#8b8495" />
                    <stop offset="100%" stopColor="#71717a" />
                  </linearGradient>
                </defs>

                {edges.map((edge) => {
                  const inBranch = branchEdgeKeys.has(edge.key);
                  return (
                    <path
                      key={`${edge.key}-glow`}
                      d={edge.d}
                      className={inBranch ? "graph-line graph-line--branch-glow" : "graph-line graph-line--glow"}
                      filter={inBranch ? "url(#lineGlow)" : undefined}
                    />
                  );
                })}

                {edges.map((edge) => {
                  const inBranch = branchEdgeKeys.has(edge.key);
                  return (
                    <path
                      key={edge.key}
                      d={edge.d}
                      className={inBranch ? "graph-line graph-line--branch" : "graph-line"}
                    />
                  );
                })}
              </svg>

              <AnimatePresence initial={false}>
                {placedNodes.map(({ node, x, y }, index) => {
                  const isCurrent = node.node_id === activeNodeId;
                  const isInBranch = branchNodeIds.has(node.node_id);

                  return (
                    <motion.div
                      key={node.node_id}
                      ref={(element) => {
                        if (element) {
                          nodeRefs.current.set(node.node_id, element);
                        } else {
                          nodeRefs.current.delete(node.node_id);
                        }
                      }}
                      className="graph-node"
                      style={{ left: `${x}px`, top: `${y}px`, width: `${CARD_WIDTH}px`, height: `${CARD_HEIGHT}px` }}
                      initial={{ opacity: 0, y: 28, scale: 0.94, filter: "blur(8px)" }}
                      animate={{ opacity: 1, y: 0, scale: 1, filter: "blur(0px)" }}
                      exit={{ opacity: 0, scale: 0.96 }}
                      transition={{ duration: 0.4, delay: Math.min(index * 0.04, 0.35) }}
                    >
                      <NodeCard
                        node={node}
                        index={index}
                        active={selectedNodeId === node.node_id}
                        current={isCurrent}
                        inBranch={isInBranch}
                        onClick={() => handleNodeClick(node.node_id)}
                        onBranchClick={onCreateBranch ? () => handleBranchClick(node.node_id) : undefined}
                      />
                    </motion.div>
                  );
                })}
              </AnimatePresence>

              {onCreateBranch ? (
                <BranchPopover
                  node={branchTarget?.node}
                  position={branchPopoverPosition}
                  busy={branchingNodeId === branchTargetId}
                  error={branchTargetId ? branchError : ""}
                  onCreateBranch={onCreateBranch}
                  onClose={() => setBranchTargetId("")}
                  zoom={zoom}
                />
              ) : null}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
