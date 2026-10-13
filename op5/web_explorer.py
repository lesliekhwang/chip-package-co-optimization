"""
web_explorer.py
---------------------------------
Exports Chiplet Placement Optimization runs to an interactive HTML/JS dashboard.
"""

import sys
import os
import json
import argparse
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for file generation
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Attempt to import layout renderer components
try:
    from layout_renderer import draw_single_placement
except ImportError:
    print("Error: layout_renderer.py not found in path. Ensure it is accessible.")
    sys.exit(1)

def nsga2_fast_non_dominated_sort(fitness: np.ndarray) -> np.ndarray:
    n_samples = fitness.shape[0]
    ranks = np.zeros(n_samples, dtype=int)
    dominated_counts = np.zeros(n_samples, dtype=int)
    dominates_list = [[] for _ in range(n_samples)]

    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            if np.all(fitness[i] <= fitness[j]) and np.any(fitness[i] < fitness[j]):
                dominates_list[i].append(j)
                dominated_counts[j] += 1
            elif np.all(fitness[j] <= fitness[i]) and np.any(fitness[j] < fitness[i]):
                dominates_list[j].append(i)
                dominated_counts[i] += 1

    current_rank = 0
    current_front = [i for i in range(n_samples) if dominated_counts[i] == 0]

    while current_front:
        next_front = []
        for i in current_front:
            ranks[i] = current_rank
            for j in dominates_list[i]:
                dominated_counts[j] -= 1
                if dominated_counts[j] == 0:
                    next_front.append(j)
        current_rank += 1
        current_front = next_front

    return ranks

def find_global_pareto_front(fitness: np.ndarray) -> np.ndarray:
    n = fitness.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if is_pareto[i]:
            dominated = np.all(fitness <= fitness[i], axis=1) & np.any(fitness < fitness[i], axis=1)
            if np.any(dominated):
                is_pareto[i] = False
            else:
                dominates = np.all(fitness[i] <= fitness, axis=1) & np.any(fitness[i] < fitness, axis=1)
                is_pareto[dominates] = False
    return is_pareto

class ChipletInfo:
    def __init__(self, name: str, w: float, h: float, adj: list = None):
        self.name = str(name)
        self.w = float(w)
        self.h = float(h)
        self.adj = adj if adj is not None else []

def render_worker(task_args):
    abs_img_path, title, row_dict, base_chiplets = task_args
    if abs_img_path.exists():
        return True 

    fig, ax = plt.subplots(figsize=(4, 4))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.9, bottom=0.05)
    
    draw_single_placement(ax, fig, row_dict, base_chiplets, 
                          show_heatmap=True, show_hpwl=True, font_size=8, is_grid=False)
    ax.set_title(title, fontsize=10)
    
    fig.savefig(abs_img_path, dpi=75)
    plt.close(fig)
    return True

def generate_html_dashboard(out_dir: Path, data_json: str, external_data_json: str = "[]"):
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Chiplet Optimization Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 0; display: flex; height: 100vh; background: #f4f4f9; color: #333; }
        .panel { padding: 15px; box-sizing: border-box; overflow-y: auto; border-right: 1px solid #ccc; background: white; }
        #left-panel { width: 35%; display: flex; flex-direction: column; }
        #mid-panel { width: 38%; display: flex; flex-direction: column; }
        #right-panel { width: 27%; font-size: 0.85em; display: flex; flex-direction: column; }
        
        .controls { display: flex; flex-direction: column; gap: 8px; margin-bottom: 10px; padding: 10px; background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; }
        .controls-row { display: flex; flex-wrap: wrap; gap: 10px; align-items: center; }
        .control-group { display: flex; flex-direction: column; gap: 3px; }
        
        .filter-row { display: flex; align-items: center; justify-content: space-between; gap: 6px; margin-bottom: 4px; }
        .filter-row label { font-weight: 600; min-width: 60px; }
        .range-inputs { display: flex; align-items: center; gap: 4px; }
        .range-inputs input { width: 65px; padding: 3px 5px; border: 1px solid #ccc; border-radius: 3px; font-size: 0.9em; }
        .range-inputs span { font-size: 0.85em; color: #555; }

        #plot { flex-grow: 1; width: 100%; min-height: 350px; }
        
        #grid-container { 
            display: grid; gap: 12px; flex-grow: 1; overflow-y: auto; background: #fafafa; 
            padding: 12px; border: 1px solid #ddd; border-radius: 4px; align-content: start; box-sizing: border-box;
        }
        .grid-item { 
            border: 1px solid #ccc; background: white; padding: 8px; border-radius: 6px; display: flex; 
            flex-direction: column; align-items: center; justify-content: space-between; aspect-ratio: 1 / 1; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.06); box-sizing: border-box; overflow: hidden; width: 100%;
        }
        .grid-item-header { font-size: 0.78em; font-weight: bold; text-align: center; margin-bottom: 4px; line-height: 1.35; width: 100%; }
        .sel-badge { background: #007bff; color: white; border-radius: 10px; padding: 1px 6px; font-size: 0.85em; margin-right: 4px; display: inline-block; }
        .grid-item img { max-width: 100%; max-height: calc(100% - 32px); object-fit: contain; }
        
        .no-img-placeholder { 
            display: flex; flex-direction: column; align-items: center; justify-content: center; height: calc(100% - 32px); 
            width: 100%; background: #f8f9fa; color: #6c757d; font-size: 0.75em; text-align: center; 
            border: 1px dashed #ced4da; border-radius: 4px; padding: 6px; box-sizing: border-box; line-height: 1.3;
        }

        .table-container { flex-grow: 1; overflow-y: auto; border: 1px solid #ddd; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; }
        th, td { border: 1px solid #ddd; padding: 5px; text-align: left; }
        th { 
            background-color: #007bff; 
            color: white; 
            position: sticky; 
            top: 0; 
            z-index: 1; 
            font-size: 0.88em; 
            cursor: pointer; 
            user-select: none;
            padding-right: 20px;
            transition: background-color 0.2s;
        }
        th:hover { background-color: #0056b3; }
        .sort-icon { 
            position: absolute; 
            right: 4px; 
            top: 50%; 
            transform: translateY(-50%); 
            font-size: 1.2em; 
            transition: opacity 0.2s;
        }
        
        tr.selected { background-color: #d3e3fc; font-weight: bold; }
        tr:hover { background-color: #f1f1f1; cursor: pointer; }
        
        input[type="number"], select { padding: 4px; border: 1px solid #ccc; border-radius: 3px; }
        button { padding: 5px 10px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 4px; font-size: 0.85em; }
        button:hover { background: #0056b3; }
        
        .header { font-weight: bold; font-size: 1.05em; margin-bottom: 8px; border-bottom: 2px solid #007bff; padding-bottom: 4px; color: #111; }
    </style>
</head>
<body>

<div id="left-panel" class="panel">
    <div class="header">Pareto Exploration</div>
    <div class="controls">
        <div class="controls-row">
            <div class="control-group">
                <label>View Mode:</label>
                <select id="plot-mode" onchange="togglePlotMode()">
                    <option value="2d">2D Projection</option>
                    <option value="3d">3D Pareto Space</option>
                </select>
            </div>
            <div class="control-group">
                <label>X-Axis:</label>
                <select id="x-axis"></select>
            </div>
            <div class="control-group">
                <label>Y-Axis:</label>
                <select id="y-axis"></select>
            </div>
            <div class="control-group" id="z-axis-group" style="display: none;">
                <label>Z-Axis:</label>
                <select id="z-axis"></select>
            </div>
            <div class="control-group" style="margin-left: auto;">
                <button onclick="clearSelection()">Clear Selection</button>
            </div>
        </div>
    </div>
    <div id="plot"></div>
</div>

<div id="mid-panel" class="panel">
    <div class="header">Layout Grid Viewer</div>
    <div class="controls">
        <div class="controls-row" style="justify-content: space-between;">
            <div>
                <label>Grid Size (NxN):</label>
                <input type="number" id="grid-size" value="3" min="1" max="10" style="width: 50px;">
            </div>
            <div style="display: flex; align-items: center; gap: 6px;">
                <button onclick="changePage(-1)">&#9664;</button>
                <span id="page-indicator" style="font-size: 0.85em; font-weight: bold;">1 / 1</span>
                <button onclick="changePage(1)">&#9654;</button>
            </div>
        </div>
    </div>
    <div id="grid-container"></div>
</div>

<div id="right-panel" class="panel">
    <div class="header">Individuals & Cost Filters</div>
    <div class="controls">
        <div class="filter-row">
            <label>HPWL:</label>
            <div class="range-inputs">
                <input type="number" id="filt-hpwl-min" placeholder="Min" step="10">
                <span>&le; x &le;</span>
                <input type="number" id="filt-hpwl-max" placeholder="Max" step="10">
            </div>
        </div>
        <div class="filter-row">
            <label>Area:</label>
            <div class="range-inputs">
                <input type="number" id="filt-area-min" placeholder="Min" step="10">
                <span>&le; x &le;</span>
                <input type="number" id="filt-area-max" placeholder="Max" step="10">
            </div>
        </div>
        <div class="filter-row">
            <label>Warpage:</label>
            <div class="range-inputs">
                <input type="number" id="filt-warp-min" placeholder="Min" step="0.001">
                <span>&le; x &le;</span>
                <input type="number" id="filt-warp-max" placeholder="Max" step="0.001">
            </div>
        </div>
        <div style="text-align: right; margin-top: 4px;">
            <button onclick="resetFilters()">Reset Range</button>
        </div>
    </div>
    <div class="table-container">
        <table id="data-table">
            <thead>
                <tr>
                    <th>#</th>
                    <th id="th-gen" onclick="sortTable('gen')">Gen <span class="sort-icon"></span></th>
                    <th id="th-ind" onclick="sortTable('ind')">Ind <span class="sort-icon"></span></th>
                    <th id="th-rank" onclick="sortTable('rank')">Rank <span class="sort-icon"></span></th>
                    <th id="th-hpwl" onclick="sortTable('hpwl')">HPWL <span class="sort-icon"></span></th>
                    <th id="th-area" onclick="sortTable('area')">Area <span class="sort-icon"></span></th>
                    <th id="th-warpage" onclick="sortTable('warpage')">Warp <span class="sort-icon"></span></th>
                </tr>
            </thead>
            <tbody></tbody>
        </table>
    </div>
</div>

<script>
    const ALL_DATA = DATA_PLACEHOLDER;
    const EXTERNAL_DATA = EXTERNAL_DATA_PLACEHOLDER;
    const METRICS = ["hpwl", "area", "warpage"];
    
    let filteredData = [];
    let selectedOrder = []; 
    let selectedMap = new Map(); 
    let currentPage = 0;
    
    // Sorting state
    let sortCol = "rank";
    let sortAsc = true;
    
    window.onload = () => {
        const xAxis = document.getElementById("x-axis");
        const yAxis = document.getElementById("y-axis");
        const zAxis = document.getElementById("z-axis");
        
        METRICS.forEach(m => {
            xAxis.add(new Option(m.toUpperCase(), m));
            yAxis.add(new Option(m.toUpperCase(), m));
            zAxis.add(new Option(m.toUpperCase(), m));
        });
        
        xAxis.value = METRICS[0];
        yAxis.value = METRICS[1];
        zAxis.value = METRICS[2];
        
        xAxis.addEventListener("change", updatePlot);
        yAxis.addEventListener("change", updatePlot);
        zAxis.addEventListener("change", updatePlot);
        
        const filterInputs = [
            "filt-hpwl-min", "filt-hpwl-max",
            "filt-area-min", "filt-area-max",
            "filt-warp-min", "filt-warp-max"
        ];
        filterInputs.forEach(id => {
            document.getElementById(id).addEventListener("input", applyFilters);
        });
        
        document.getElementById("grid-size").addEventListener("change", () => {
            currentPage = 0;
            renderGrid();
        });
        
        resetFilters();
    };

    function togglePlotMode() {
        const is3D = document.getElementById("plot-mode").value === "3d";
        document.getElementById("z-axis-group").style.display = is3D ? "flex" : "none";
        updatePlot();
    }

    function resetFilters() {
        const hpwls = ALL_DATA.map(d => d.hpwl);
        const areas = ALL_DATA.map(d => d.area);
        const warps = ALL_DATA.map(d => d.warpage);
        
        document.getElementById("filt-hpwl-min").value = Math.floor(Math.min(...hpwls));
        document.getElementById("filt-hpwl-max").value = Math.ceil(Math.max(...hpwls));
        
        document.getElementById("filt-area-min").value = Math.floor(Math.min(...areas));
        document.getElementById("filt-area-max").value = Math.ceil(Math.max(...areas));
        
        document.getElementById("filt-warp-min").value = (Math.min(...warps)).toFixed(4);
        document.getElementById("filt-warp-max").value = (Math.max(...warps) + 0.001).toFixed(4);
        
        applyFilters();
    }

    function applyFilters() {
        const minHpwl = parseFloat(document.getElementById("filt-hpwl-min").value);
        const maxHpwl = parseFloat(document.getElementById("filt-hpwl-max").value);
        
        const minArea = parseFloat(document.getElementById("filt-area-min").value);
        const maxArea = parseFloat(document.getElementById("filt-area-max").value);
        
        const minWarp = parseFloat(document.getElementById("filt-warp-min").value);
        const maxWarp = parseFloat(document.getElementById("filt-warp-max").value);
        
        filteredData = ALL_DATA.filter(d => 
            (isNaN(minHpwl) || d.hpwl >= minHpwl) &&
            (isNaN(maxHpwl) || d.hpwl <= maxHpwl) &&
            (isNaN(minArea) || d.area >= minArea) &&
            (isNaN(maxArea) || d.area <= maxArea) &&
            (isNaN(minWarp) || d.warpage >= minWarp) &&
            (isNaN(maxWarp) || d.warpage <= maxWarp)
        );
        
        const validIds = new Set(filteredData.map(d => d.id));
        selectedOrder = selectedOrder.filter(id => validIds.has(id));
        rebuildSelectedMap();
        
        sortData();
        updatePlot();
        renderTable();
        updateSortIcons();
        renderGrid();
    }

    function sortTable(col) {
        if (sortCol === col) {
            sortAsc = !sortAsc;
        } else {
            sortCol = col;
            sortAsc = true;
        }
        sortData();
        renderTable();
        updateSortIcons();
    }

    function sortData() {
        filteredData.sort((a, b) => {
            let valA = a[sortCol];
            let valB = b[sortCol];
            if (valA < valB) return sortAsc ? -1 : 1;
            if (valA > valB) return sortAsc ? 1 : -1;
            return 0;
        });
    }

    function updateSortIcons() {
        const ths = document.querySelectorAll("#data-table th");
        ths.forEach(th => {
            const icon = th.querySelector(".sort-icon");
            if (icon) {
                icon.innerHTML = "&#8693;"; 
                icon.style.opacity = "0.4";
                icon.style.color = "inherit";
            }
        });
        const activeTh = document.getElementById("th-" + sortCol);
        if (activeTh) {
            const activeIcon = activeTh.querySelector(".sort-icon");
            activeIcon.innerHTML = sortAsc ? "&#9650;" : "&#9660;"; 
            activeIcon.style.opacity = "1";
            activeIcon.style.color = "#ffeb3b"; 
        }
    }

    function rebuildSelectedMap() {
        selectedMap.clear();
        selectedOrder.forEach((id, idx) => {
            selectedMap.set(id, idx + 1);
        });
    }

    function updatePlot() {
        const is3D = document.getElementById("plot-mode").value === "3d";
        const xKey = document.getElementById("x-axis").value;
        const yKey = document.getElementById("y-axis").value;
        const zKey = document.getElementById("z-axis").value;
        
        const globalParetoData = [];
        const finalGenData = [];
        const otherData = [];
        
        filteredData.forEach(d => {
            if (d.is_global_pareto) {
                globalParetoData.push(d);
            } else if (d.is_final_gen) {
                finalGenData.push(d);
            } else {
                otherData.push(d);
            }
        });

        function makeTrace(name, data, color, defaultSize) {
            const trace = {
                name: name,
                x: data.map(d => d[xKey]),
                y: data.map(d => d[yKey]),
                mode: 'markers',
                type: is3D ? 'scatter3d' : 'scatter',
                text: data.map(d => `Gen: ${d.gen}<br>Ind: ${d.ind}<br>Rank: ${d.rank}${d.is_global_pareto ? '<br><b>Global Pareto</b>' : ''}`),
                hoverinfo: is3D ? 'text+x+y+z' : 'text+x+y',
                customdata: data.map(d => d.id),
                marker: {
                    color: color,
                    size: is3D ? defaultSize * 0.7 : defaultSize,
                    opacity: 0.75
                }
            };
            if (is3D) trace.z = data.map(d => d[zKey]);
            return trace;
        }

        const traces = [
            makeTrace('Other Generations', otherData, '#8d99ae', 5),
            makeTrace('Final Generation', finalGenData, '#f77f00', 7),
            makeTrace('Global Pareto Front', globalParetoData, '#e63946', 9)
        ];

        // Ideal and Nadir pulls exclusively from the Global Pareto Front
        if (globalParetoData.length > 0) {
            const minX = Math.min(...globalParetoData.map(d => d[xKey]));
            const maxX = Math.max(...globalParetoData.map(d => d[xKey]));
            const minY = Math.min(...globalParetoData.map(d => d[yKey]));
            const maxY = Math.max(...globalParetoData.map(d => d[yKey]));

            const idealTrace = {
                name: 'Ideal Point',
                x: [minX], y: [minY],
                mode: 'text',
                text: ['★'], 
                textposition: 'middle center',
                type: is3D ? 'scatter3d' : 'scatter',
                textfont: { size: is3D ? 40 : 40, color: '#D000F0', family: 'Arial, sans-serif' },
                hoverinfo: is3D ? 'name+x+y+z' : 'name+x+y'
            };

            const nadirTrace = {
                name: 'Nadir Point',
                x: [maxX], y: [maxY],
                mode: 'text',
                text: ['🕱'],
                textposition: 'middle center',
                type: is3D ? 'scatter3d' : 'scatter',
                textfont: { size: is3D ? 40 : 40, color: '#D000F0', family: 'Arial, sans-serif' },
                hoverinfo: is3D ? 'name+x+y+z' : 'name+x+y'
            };

            if (is3D) {
                idealTrace.z = [Math.min(...globalParetoData.map(d => d[zKey]))];
                nadirTrace.z = [Math.max(...globalParetoData.map(d => d[zKey]))];
            }

            traces.push(idealTrace);
            traces.push(nadirTrace);
        }

        // External / Competitor points plotted as red crossed swords (⚔)
        if (EXTERNAL_DATA && EXTERNAL_DATA.length > 0) {
            const extTrace = {
                name: 'Competitor Metrics',
                x: EXTERNAL_DATA.map(d => d[xKey]),
                y: EXTERNAL_DATA.map(d => d[yKey]),
                mode: 'text',
                text: EXTERNAL_DATA.map(d => '⚔'),
                type: is3D ? 'scatter3d' : 'scatter',
                textposition: 'middle center',
                textfont: { size: 30, color: '#00C853', family: 'Arial, sans-serif', weight: 'bold' },
                hovertext: EXTERNAL_DATA.map(d => `<b>Competitor ID: ${d.placement_id}</b><br>HPWL: ${d.hpwl.toFixed(1)}<br>Area: ${d.area.toFixed(1)}<br>Warp: ${d.warpage.toFixed(4)}`),
                hoverinfo: 'text'
            };
            if (is3D) extTrace.z = EXTERNAL_DATA.map(d => d[zKey]);
            traces.push(extTrace);
        }

        const selectedPointsData = selectedOrder.map(id => ALL_DATA.find(d => d.id === id)).filter(Boolean);
        if (selectedPointsData.length > 0) {
            const selTrace = {
                name: 'Selected Items',
                x: selectedPointsData.map(d => d[xKey]),
                y: selectedPointsData.map(d => d[yKey]),
                mode: 'markers+text',
                type: is3D ? 'scatter3d' : 'scatter',
                text: selectedPointsData.map(d => `${selectedMap.get(d.id)}`),
                textposition: 'top center',
                textfont: { family: 'Segoe UI, sans-serif', size: 12, color: '#000000', weight: 'bold' },
                customdata: selectedPointsData.map(d => d.id),
                marker: {
                    color: '#ffc107',
                    size: is3D ? 8 : 11,
                    line: { color: '#000000', width: 2 }
                },
                hoverinfo: is3D ? 'text+x+y+z' : 'text+x+y'
            };
            if (is3D) selTrace.z = selectedPointsData.map(d => d[zKey]);
            traces.push(selTrace);
        }

        const layout = is3D ? {
            margin: { l: 0, r: 0, t: 0, b: 0 },
            scene: {
                xaxis: { title: xKey.toUpperCase() },
                yaxis: { title: yKey.toUpperCase() },
                zaxis: { title: zKey.toUpperCase() }
            },
            legend: { orientation: 'h', y: 1.05, x: 0 }
        } : {
            margin: { l: 50, r: 20, t: 20, b: 50 },
            xaxis: { title: xKey.toUpperCase() },
            yaxis: { title: yKey.toUpperCase() },
            hovermode: 'closest',
            dragmode: 'lasso',
            legend: { orientation: 'h', y: 1.12, x: 0 }
        };

        Plotly.newPlot('plot', traces, layout, {responsive: true});

        const plotEl = document.getElementById('plot');
        plotEl.removeAllListeners('plotly_selected');
        plotEl.removeAllListeners('plotly_click');

        plotEl.on('plotly_selected', (eventData) => {
            if (!eventData || !eventData.points) return;
            eventData.points.forEach(pt => {
                const id = pt.customdata;
                if (id && !selectedMap.has(id)) {
                    selectedOrder.push(id);
                }
            });
            rebuildSelectedMap();
            syncUI();
        });

        plotEl.on('plotly_click', (eventData) => {
            if (!eventData || !eventData.points.length) return;
            const id = eventData.points[0].customdata;
            if (id) toggleSelection(id);
        });
    }

    function toggleSelection(id) {
        if (selectedMap.has(id)) {
            selectedOrder = selectedOrder.filter(x => x !== id);
        } else {
            selectedOrder.push(id);
        }
        rebuildSelectedMap();
        syncUI();
    }

    function clearSelection() {
        selectedOrder = [];
        selectedMap.clear();
        syncUI();
    }

    function syncUI() {
        currentPage = 0;
        updatePlot();
        renderTable();
        renderGrid();
    }

    function renderTable() {
        const tbody = document.querySelector("#data-table tbody");
        tbody.innerHTML = "";
        filteredData.forEach(d => {
            const tr = document.createElement("tr");
            const isSel = selectedMap.has(d.id);
            if (isSel) tr.className = "selected";
            tr.onclick = () => toggleSelection(d.id);
            
            const numLabel = isSel ? `#${selectedMap.get(d.id)}` : '-';
            
            tr.innerHTML = `
                <td style="font-weight: bold; color: ${isSel ? '#007bff' : '#888'}">${numLabel}</td>
                <td>${d.gen}</td>
                <td>${d.ind}</td>
                <td>${d.rank}</td>
                <td>${d.hpwl.toFixed(1)}</td>
                <td>${d.area.toFixed(1)}</td>
                <td>${d.warpage.toFixed(4)}</td>
            `;
            tbody.appendChild(tr);
        });
    }

    function renderGrid() {
        const container = document.getElementById("grid-container");
        const n = Math.max(1, parseInt(document.getElementById("grid-size").value) || 3);
        
        container.style.gridTemplateColumns = `repeat(${n}, 1fr)`;
        container.innerHTML = "";

        const selectionArray = selectedOrder.map(id => ALL_DATA.find(d => d.id === id)).filter(Boolean);
        
        const itemsPerPage = n * n;
        const totalPages = Math.max(1, Math.ceil(selectionArray.length / itemsPerPage));
        if (currentPage >= totalPages) currentPage = totalPages - 1;
        if (currentPage < 0) currentPage = 0;
        
        document.getElementById("page-indicator").innerText = `${currentPage + 1} / ${totalPages} (${selectionArray.length} items)`;

        const start = currentPage * itemsPerPage;
        const pageItems = selectionArray.slice(start, start + itemsPerPage);

        pageItems.forEach(d => {
            const div = document.createElement("div");
            div.className = "grid-item";
            const selNum = selectedMap.get(d.id);
            
            let imgHtml = "";
            if (d.has_img) {
                imgHtml = `
                    <img src="${d.img_path}" alt="Layout ${d.id}" 
                         onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                    <div class="no-img-placeholder" style="display:none;">
                        <b>Image file missing</b><br><small>${d.img_path}</small>
                    </div>
                `;
            } else {
                imgHtml = `
                    <div class="no-img-placeholder">
                        <b>Image not rendered</b>
                        <span style="margin-top: 4px; font-size: 0.85em;">(Run with <code>--render-mode all</code> to render all images)</span>
                    </div>
                `;
            }

            div.innerHTML = `
                <div class="grid-item-header">
                    <span class="sel-badge">#${selNum}</span>
                    Gen ${d.gen} | Ind ${d.ind}<br>H:${d.hpwl.toFixed(0)} | A:${d.area.toFixed(0)} | W:${d.warpage.toFixed(4)}
                </div>
                ${imgHtml}
            `;
            container.appendChild(div);
        });
    }

    function changePage(delta) {
        const n = Math.max(1, parseInt(document.getElementById("grid-size").value) || 3);
        const itemsPerPage = n * n;
        const maxPage = Math.max(0, Math.ceil(selectedOrder.length / itemsPerPage) - 1);
        
        currentPage += delta;
        if (currentPage < 0) currentPage = 0;
        if (currentPage > maxPage) currentPage = maxPage;
        renderGrid();
    }
</script>
</body>
</html>
"""
    # Replace external data placeholder before ALL_DATA placeholder to avoid substring collisions
    html_content = html_content.replace("EXTERNAL_DATA_PLACEHOLDER", external_data_json)
    html_content = html_content.replace("DATA_PLACEHOLDER", data_json)
    
    index_path = out_dir / "index.html"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n[+] Successfully generated dashboard at: {index_path.resolve()}")

def main():
    parser = argparse.ArgumentParser(description="Export Chiplet Placement Run to HTML/JS Dashboard.")
    parser.add_argument("run_dir", type=str, help="Path to the directory containing run.h5 and jsons")
    parser.add_argument("--out", type=str, default="html_export", help="Output directory for HTML and images")
    parser.add_argument(
        "--render-mode", 
        choices=["all", "pareto-and-final", "none"], 
        default="all", 
        help="Image rendering scope: 'all' (default), 'pareto-and-final' (only global Pareto front or final generation), or 'none'"
    )
    parser.add_argument("--skip-images", action="store_true", help="Shortcut for --render-mode none")
    parser.add_argument("--metrics-csv", type=str, default=None, help="Path to competitor metrics CSV file to overlay on scatter plot")
    args = parser.parse_args()

    render_mode = "none" if args.skip_images else args.render_mode

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out)
    images_dir = out_dir / "images"

    h5_path = run_dir / "run.h5"
    chiplets_path = run_dir / "chiplets.json"

    if not h5_path.exists() or not chiplets_path.exists():
        print(f"Error: Missing run.h5 or chiplets.json in {run_dir}")
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    if render_mode != "none":
        images_dir.mkdir(parents=True, exist_ok=True)

    with open(chiplets_path, "r") as f: 
        chiplet_cfg = json.load(f)
    
    id_map = [k for k in chiplet_cfg.keys() if k.lower() != "interposer"]
    base_chiplets = {}
    for name in id_map:
        c = chiplet_cfg[name]
        base_chiplets[name] = ChipletInfo(name, float(c["dim"][0]), float(c["dim"][1]), c.get("adjacency", []))

    # Process competitor metrics CSV if available
    external_records = []
    metrics_csv_path = None
    if args.metrics_csv:
        metrics_csv_path = Path(args.metrics_csv)
    elif (run_dir / "placement_metrics.csv").exists():
        metrics_csv_path = run_dir / "placement_metrics.csv"
    elif Path("placement_metrics.csv").exists():
        metrics_csv_path = Path("placement_metrics.csv")

    if metrics_csv_path and metrics_csv_path.exists():
        print(f"\n[+] Processing external competitor metrics from: {metrics_csv_path}")
        try:
            df = pd.read_csv(metrics_csv_path)
            required_cols = {'placement_id', 'hpwl', 'area', 'warpage'}
            if required_cols.issubset(df.columns):
                out_csv_path = out_dir / "external_placements_costs.csv"
                out_df = df[['placement_id', 'hpwl', 'area', 'warpage']]
                out_df.to_csv(out_csv_path, index=False)
                print(f"    Exported reformatted costs to: {out_csv_path}")
                
                for _, row in out_df.iterrows():
                    external_records.append({
                        "placement_id": int(row['placement_id']),
                        "hpwl": float(row['hpwl']),
                        "area": float(row['area']),
                        "warpage": float(row['warpage'])
                    })
            else:
                print(f"    [!] Warning: Missing required columns in CSV. Expected {required_cols}, found {set(df.columns)}")
        except Exception as e:
            print(f"    [!] Error reading CSV: {e}")

    records = []
    render_tasks = []
    
    print("\n[+] Reading HDF5 Data and Extracting Metrics...")
    
    with h5py.File(h5_path, "r") as h5f:
        gen_keys = sorted([k for k in h5f["generations"].keys() if k.startswith("gen") and k[3:].isdigit()], 
                          key=lambda x: int(x.replace("gen", "")))
        
        gen_numbers = [int(k.replace("gen", "")) for k in gen_keys]
        max_gen = max(gen_numbers) if gen_numbers else 0
        
        all_fitness = []
        global_idx_map = {}
        idx_counter = 0

        for gen_key in gen_keys:
            gen = int(gen_key.replace("gen", ""))
            fitness = h5f["generations"][gen_key]["fitness"]["F"][:]
            for sol_idx in range(len(fitness)):
                all_fitness.append(fitness[sol_idx])
                global_idx_map[(gen, sol_idx)] = idx_counter
                idx_counter += 1

        all_fitness_arr = np.array(all_fitness)
        total_samples = len(all_fitness_arr)
        
        print(f"    Found {len(gen_keys)} generations, {total_samples} total individuals.")
        
        print("    Computing global Pareto front across all samples...")
        global_pareto_mask = find_global_pareto_front(all_fitness_arr)
        num_global_pareto = int(np.sum(global_pareto_mask))
        print(f"    Found {num_global_pareto} individuals on the global Pareto front.")

        print(f"    Render mode: '{render_mode}' (Max Gen: {max_gen})")
        
        processed_count = 0
        
        for gen_key in gen_keys:
            gen = int(gen_key.replace("gen", ""))
            gen_grp = h5f["generations"][gen_key]
            
            fitness = gen_grp["fitness"]["F"][:]
            pop_size = len(fitness)
            ranks_arr = nsga2_fast_non_dominated_sort(fitness)
            
            pheno_grp = gen_grp["phenotype"]
            geno_grp = gen_grp["genotype"]
            
            bboxes_arr = pheno_grp["bboxes"][:]
            total_w_arr = pheno_grp["total_w"][:]
            total_h_arr = pheno_grp["total_h"][:]
            slack_arr = geno_grp["slack"][:]
            warpage_maps = pheno_grp.get("warpage_maps", None)

            gen_img_dir = images_dir / f"gen_{gen}"

            for sol_idx in range(pop_size):
                rel_img_path = f"images/gen_{gen}/ind_{sol_idx}.png"
                abs_img_path = out_dir / rel_img_path
                
                g_idx = global_idx_map[(gen, sol_idx)]
                is_global_pareto = bool(global_pareto_mask[g_idx])
                is_final_gen = bool(gen == max_gen)
                
                should_render = False
                if render_mode == "all":
                    should_render = True
                elif render_mode == "pareto-and-final":
                    should_render = is_global_pareto or is_final_gen
                
                if should_render:
                    gen_img_dir.mkdir(exist_ok=True)
                    
                    sol_bboxes = {id_map[i]: bboxes_arr[sol_idx, i].tolist() for i in range(len(id_map))}
                    
                    slack_mat = slack_arr[sol_idx]
                    ws_dict = {
                        name: {
                            "ws_n": float(slack_mat[0, i_c]), 
                            "ws_e": float(slack_mat[1, i_c]), 
                            "ws_s": float(slack_mat[2, i_c]), 
                            "ws_w": float(slack_mat[3, i_c])
                        } for i_c, name in enumerate(id_map)
                    }

                    row_dict = {
                        "bboxes": sol_bboxes,
                        "total_w": float(total_w_arr[sol_idx]),
                        "total_h": float(total_h_arr[sol_idx]),
                        "chiplet_whitespace": ws_dict,
                        "warpage_matrix": warpage_maps[sol_idx] if warpage_maps is not None else None,
                    }
                    
                    title = f"Gen {gen} | Ind {sol_idx}"
                    render_tasks.append((abs_img_path, title, row_dict, base_chiplets))

                records.append({
                    "id": f"g{gen}_i{sol_idx}",
                    "gen": gen,
                    "ind": sol_idx,
                    "rank": int(ranks_arr[sol_idx]),
                    "hpwl": float(fitness[sol_idx, 0]),
                    "area": float(fitness[sol_idx, 1]),
                    "warpage": float(fitness[sol_idx, 2]) if fitness.shape[1] > 2 else 0.0,
                    "img_path": rel_img_path,
                    "has_img": should_render,
                    "is_global_pareto": is_global_pareto,
                    "is_final_gen": is_final_gen
                })
                
                processed_count += 1
                
    rendered_count = 0
    if render_tasks:
        num_workers = min(multiprocessing.cpu_count() or 4, 16)
        print(f"\n[+] Dispatching {len(render_tasks)} image generation tasks across {num_workers} CPU cores...")
        
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(render_worker, task) for task in render_tasks]
            
            for future in as_completed(futures):
                try:
                    future.result()
                    rendered_count += 1
                    if rendered_count % 100 == 0:
                        print(f"    Rendered {rendered_count}/{len(render_tasks)} images...")
                except Exception as exc:
                    print(f"    [!] Render task generated an exception: {exc}")
    
    print(f"\n[+] Processing complete: {processed_count} total records exported, {rendered_count} layout images queued/rendered.")
    print("[+] Generating HTML/JS Application...")
    generate_html_dashboard(out_dir, json.dumps(records), json.dumps(external_records))

if __name__ == "__main__":
    main()