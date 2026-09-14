"""
ga.py
---------------------------------
Genetic algorithm operators and problem definition for pure sequence optimization.
Includes HPWL, Area, and Thermo-Mechanical Warpage objectives.
Now includes continuous genes for dynamically optimizing additional chiplet whitespace
in four independent directions (N, E, S, W), mapped explicitly by chiplet name.
"""

import copy
import random
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path

from pymoo.core.callback import Callback
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling

import seqpair
from cost import area, hpwl, warpage

class ChipletPlacementProblem(Problem):
    def __init__(
        self, 
        base_chiplets: dict, 
        use_hpwl: bool = True, 
        use_area: bool = True, 
        use_warpage: bool = True
    ):
        self.base_chiplets = base_chiplets
        self.names = list(base_chiplets.keys())
        self.n_chiplets = len(self.names)
        
        self.use_hpwl = use_hpwl
        self.use_area = use_area
        self.use_warpage = use_warpage
        
        # Keep n_obj fixed at 3 so array shapes match the tracker callback
        super().__init__(n_var=1, n_obj=3, n_constr=0)

    def _evaluate(self, X_pop: np.ndarray, out: dict, *args, **kwargs) -> None:
        F = []
        real_F = []
        warpage_tensors = []
        
        for ind in X_pop:
            actual_ind = ind[0] if isinstance(ind, (np.ndarray, list)) else ind
            x_seq = actual_ind["X"]
            y_seq = actual_ind["Y"]
            ws_n = actual_ind["WS_N"] 
            ws_e = actual_ind["WS_E"] 
            ws_s = actual_ind["WS_S"] 
            ws_w = actual_ind["WS_W"] 
            
            eval_chiplets = copy.deepcopy(self.base_chiplets)
            
            for name in self.names:
                eval_chiplets[name].ws_n += ws_n[name]
                eval_chiplets[name].ws_e += ws_e[name]
                eval_chiplets[name].ws_s += ws_s[name]
                eval_chiplets[name].ws_w += ws_w[name]
            
            x_coords, y_coords, total_w, total_h = seqpair.sequence_pair_to_placement(
                x_seq, y_seq, eval_chiplets
            )
            
            # --- Always compute the actual values for logging/plots ---
            real_hpwl = hpwl(x_coords, y_coords, eval_chiplets)
            real_area = area(total_w, total_h)
            warp, real_warp_val = warpage(x_coords, y_coords, total_w, total_h, eval_chiplets, warpagemap=True)

            # --- Mask values for PyMoo's optimization vector (0.0 if turned off) ---
            c1 = real_hpwl if self.use_hpwl else 0.0
            c2 = real_area if self.use_area else 0.0
            c3 = real_warp_val if self.use_warpage else 0.0

            warpage_tensors.append(warp)
            F.append([c1, c2, c3])
            real_F.append([real_hpwl, real_area, real_warp_val])

        out["warpage_tensor"] = warpage_tensors
        out["F"] = np.array(F, dtype=float)
        # Pass the real unmasked metrics out so the callback can capture them
        out["real_metrics"] = np.array(real_F, dtype=float)

class PermutationPlacementSampling(Sampling):
    def __init__(self, names: list, ws_max: float):
        super().__init__()
        self.names = names
        self.n_chiplets = len(names)
        self.ws_max = ws_max

    def _do(self, problem: Problem, n_samples: int, **kwargs) -> np.ndarray:
        X_out = []
        for _ in range(n_samples):
            ind = {
                "X": random.sample(self.names, self.n_chiplets),
                "Y": random.sample(self.names, self.n_chiplets),
                # Initialize whitespace as dictionaries mapped by chiplet name
                "WS_N": {name: random.uniform(0.0, self.ws_max) for name in self.names},
                "WS_E": {name: random.uniform(0.0, self.ws_max) for name in self.names},
                "WS_S": {name: random.uniform(0.0, self.ws_max) for name in self.names},
                "WS_W": {name: random.uniform(0.0, self.ws_max) for name in self.names}
            }
            X_out.append([ind])
        return np.array(X_out, dtype=object)


class PermutationPlacementCrossover(Crossover):
    def __init__(self):
        super().__init__(n_parents=2, n_offsprings=2)

    def _do(self, problem: Problem, X: np.ndarray, **kwargs) -> np.ndarray:
        if len(X.shape) == 2:
            n_parents, n_matings = X.shape
        else:
            n_parents, n_matings, _ = X.shape
            
        Y_out = np.empty((self.n_offsprings, n_matings, problem.n_var), dtype=object)
        
        for i in range(n_matings):
            if len(X.shape) == 2:
                p1_raw, p2_raw = X[0, i], X[1, i]
            else:
                p1_raw, p2_raw = X[0, i, 0], X[1, i, 0]
                
            p1 = p1_raw[0] if isinstance(p1_raw, (list, np.ndarray)) else p1_raw
            p2 = p2_raw[0] if isinstance(p2_raw, (list, np.ndarray)) else p2_raw
            
            def ox_crossover(parent1: list, parent2: list) -> list:
                """Order Crossover (OX1) for permutation sequences."""
                idx = sorted(random.sample(range(len(parent1)), 2))
                child = [None] * len(parent1)
                
                # Copy segment from parent 1
                child[idx[0]:idx[1]+1] = parent1[idx[0]:idx[1]+1]
                
                # Fill remaining from parent 2 in order, skipping duplicates
                p2_filtered = [item for item in parent2 if item not in child]
                cur = 0
                for j in range(len(child)):
                    if child[j] is None:
                        child[j] = p2_filtered[cur]
                        cur += 1
                return child

            c1_x = ox_crossover(p1["X"], p2["X"])
            c2_x = ox_crossover(p2["X"], p1["X"])
            c1_y = ox_crossover(p1["Y"], p2["Y"])
            c2_y = ox_crossover(p2["Y"], p1["Y"])
            
            # Uniform Crossover for Dictionary Whitespace Genes
            c1_wsn, c2_wsn = {}, {}
            c1_wse, c2_wse = {}, {}
            c1_wss, c2_wss = {}, {}
            c1_wsw, c2_wsw = {}, {}
            
            for name in p1["WS_N"].keys():
                # Crossover North
                if random.random() < 0.5: c1_wsn[name], c2_wsn[name] = p1["WS_N"][name], p2["WS_N"][name]
                else: c1_wsn[name], c2_wsn[name] = p2["WS_N"][name], p1["WS_N"][name]
                
                # Crossover East
                if random.random() < 0.5: c1_wse[name], c2_wse[name] = p1["WS_E"][name], p2["WS_E"][name]
                else: c1_wse[name], c2_wse[name] = p2["WS_E"][name], p1["WS_E"][name]

                # Crossover South
                if random.random() < 0.5: c1_wss[name], c2_wss[name] = p1["WS_S"][name], p2["WS_S"][name]
                else: c1_wss[name], c2_wss[name] = p2["WS_S"][name], p1["WS_S"][name]

                # Crossover West
                if random.random() < 0.5: c1_wsw[name], c2_wsw[name] = p1["WS_W"][name], p2["WS_W"][name]
                else: c1_wsw[name], c2_wsw[name] = p2["WS_W"][name], p1["WS_W"][name]

            Y_out[0, i, 0] = {"X": c1_x, "Y": c1_y, "WS_N": c1_wsn, "WS_E": c1_wse, "WS_S": c1_wss, "WS_W": c1_wsw}
            Y_out[1, i, 0] = {"X": c2_x, "Y": c2_y, "WS_N": c2_wsn, "WS_E": c2_wse, "WS_S": c2_wss, "WS_W": c2_wsw}
            
        return Y_out


class PermutationPlacementMutation(Mutation):
    def __init__(self, names: list, mut_prob_x: float, mut_prob_y: float, mut_prob_ws: float, ws_max: float):
        super().__init__()
        self.names = names
        self.n_chiplets = len(names)
        
        self.mut_prob_x = mut_prob_x
        self.mut_prob_y = mut_prob_y
        self.mut_prob_ws = mut_prob_ws
        self.ws_max = ws_max

    def _do(self, problem: Problem, X: np.ndarray, **kwargs) -> np.ndarray:
        Y = np.copy(X)
        for i in range(len(Y)):
            is_nested = isinstance(Y[i], (np.ndarray, list)) and len(Y[i]) == 1
            ind = copy.deepcopy(Y[i][0] if is_nested else Y[i])
            
            # Sequence X Swap Mutation
            if random.random() < self.mut_prob_x:
                idx1, idx2 = random.sample(range(self.n_chiplets), 2)
                ind["X"][idx1], ind["X"][idx2] = ind["X"][idx2], ind["X"][idx1]
                
            # Sequence Y Swap Mutation
            if random.random() < self.mut_prob_y:
                idx1, idx2 = random.sample(range(self.n_chiplets), 2)
                ind["Y"][idx1], ind["Y"][idx2] = ind["Y"][idx2], ind["Y"][idx1]
                
            # Random Reset Mutation for Dictionary Whitespace Genes
            for name in self.names:
                if random.random() < self.mut_prob_ws: ind["WS_N"][name] = random.uniform(0.0, self.ws_max)
                if random.random() < self.mut_prob_ws: ind["WS_E"][name] = random.uniform(0.0, self.ws_max)
                if random.random() < self.mut_prob_ws: ind["WS_S"][name] = random.uniform(0.0, self.ws_max)
                if random.random() < self.mut_prob_ws: ind["WS_W"][name] = random.uniform(0.0, self.ws_max)
                
            # Re-wrap for PyMoo if necessary
            if is_nested:
                Y[i][0] = ind
            else:
                Y[i] = ind
                
        return Y


class ParetoFrontTrackerCallback(Callback):
    """Logs the ENTIRE population at each generation into a binary Parquet file and saves warpage tensors."""

    def __init__(self, filename: str):
        super().__init__()
        self.filename = Path(filename)
        self.writer = None
        
        # Create the directory for warpage tensors next to the parquet file
        self.tensors_dir = self.filename.parent / "warpage_tensors"
        self.tensors_dir.mkdir(parents=True, exist_ok=True)

    def notify(self, algorithm) -> None:
        population = algorithm.pop
        generation = algorithm.n_gen
        chiplet_names = getattr(algorithm.problem, "names", [])
        base_chiplets = getattr(algorithm.problem, "base_chiplets", {})

        batch_data = {
            "generation": [],
            "solution_index": [],
            "hpwl": [],
            "area": [],
            "warpage": [],
            "total_w": [],
            "total_h": [],
            "sequence_x": [],
            "sequence_y": [],
            "chiplet_whitespace": [],
            "warpage_tensor_path": []  # New column for the npz file pointer
        }

        for idx, ind in enumerate(population):
            
            # --- Extract real metrics for logging instead of masked F ---
            real_metrics = ind.get("real_metrics")
            if real_metrics is not None:
                hpwl_val, area_val, warpage_val = float(real_metrics[0]), float(real_metrics[1]), float(real_metrics[2])
            else:
                # Fallback to PyMoo's F if real_metrics isn't present
                hpwl_val, area_val, warpage_val = float(ind.F[0]), float(ind.F[1]), float(ind.F[2])

            chrom_wrapper = ind.X
            raw_dict = chrom_wrapper[0] if isinstance(chrom_wrapper, (list, np.ndarray)) else chrom_wrapper

            seq_x = list(raw_dict["X"])
            seq_y = list(raw_dict["Y"])
            ws_n = raw_dict["WS_N"]
            ws_e = raw_dict["WS_E"]
            ws_s = raw_dict["WS_S"]
            ws_w = raw_dict["WS_W"]

            # Pull cleanly from the dictionary gene representation
            chiplet_ws = {
                name: {
                    "ws_n": float(ws_n[name]),
                    "ws_e": float(ws_e[name]),
                    "ws_s": float(ws_s[name]),
                    "ws_w": float(ws_w[name])
                }
                for name in chiplet_names
            }

            # Evaluate bounding box dimensions (total_w, total_h)
            eval_chiplets = copy.deepcopy(base_chiplets)
            for name in chiplet_names:
                eval_chiplets[name].ws_n += ws_n[name]
                eval_chiplets[name].ws_e += ws_e[name]
                eval_chiplets[name].ws_s += ws_s[name]
                eval_chiplets[name].ws_w += ws_w[name]

            _, _, total_w, total_h = seqpair.sequence_pair_to_placement(
                seq_x, seq_y, eval_chiplets
            )

            # Extract the real generated tensor. 
            tensor = ind.get("warpage_tensor") 
            
            # Fallback if the tensor wasn't attached to the individual
            if tensor is None:
                tensor = np.zeros((256, 256)) 

            tensor_filename = f"gen_{generation}_idx_{idx + 1}.npz"
            tensor_path = self.tensors_dir / tensor_filename
            
            # Save the tensor as a compressed npz file
            np.savez_compressed(tensor_path, warpage=tensor)

            # --- Append to Parquet batch ---
            batch_data["generation"].append(int(generation))
            batch_data["solution_index"].append(int(idx + 1))
            batch_data["hpwl"].append(hpwl_val)
            batch_data["area"].append(area_val)
            batch_data["warpage"].append(warpage_val)
            batch_data["total_w"].append(float(total_w))
            batch_data["total_h"].append(float(total_h))
            batch_data["sequence_x"].append(seq_x)
            batch_data["sequence_y"].append(seq_y)
            batch_data["chiplet_whitespace"].append(chiplet_ws)
            
            # Store the relative path (e.g., "warpage_tensors/gen_1_idx_1.npz")
            batch_data["warpage_tensor_path"].append(str(tensor_path.relative_to(self.filename.parent)))

        table = pa.Table.from_pydict(batch_data)

        if self.writer is None:
            self.writer = pq.ParquetWriter(self.filename, table.schema, compression="SNAPPY")
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __del__(self) -> None:
        self.close()