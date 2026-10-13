"""
ga.py
---------------------------------
Genetic algorithm operators, problem definition, and HDF5 logging callback 
for Sequence Pair placement. Supports full lineage, physical coordinates, 
warpage maps, mutation metrics, pre-computed Pareto ranks, and organized 
hierarchical HDF5 logging.
"""

import copy
import json
import random
import time
import uuid
import numpy as np
import h5py

from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.core.callback import Callback

import seqpair
from cost import area_batch, hpwl_batch, warpage_batch


# --- Helper Function for JSON -> HDF5 Mapping ---
def _save_dict_to_hdf5(group: h5py.Group, data_dict: dict) -> None:
    """Recursively saves nested dictionary key-values into HDF5 datasets and attributes."""
    for key, val in data_dict.items():
        if isinstance(val, dict):
            sub_group = group.create_group(str(key))
            _save_dict_to_hdf5(sub_group, val)
        elif isinstance(val, (list, tuple, np.ndarray)):
            try:
                arr = np.array(val)
                if arr.dtype.kind in ['U', 'S', 'O']:
                    dt = h5py.string_dtype(encoding='utf-8')
                    group.create_dataset(str(key), data=np.array(val, dtype=dt))
                else:
                    group.create_dataset(str(key), data=arr)
            except Exception:
                group.attrs[str(key)] = str(val)
        elif isinstance(val, (int, float, str, bool)):
            group.attrs[str(key)] = val


# --- Topological Operators ---
def ox_crossover(parent1: list, parent2: list) -> list:
    n = len(parent1)
    if n < 2: return list(parent1)
    idx1, idx2 = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[idx1:idx2 + 1] = parent1[idx1:idx2 + 1]
    p2_filtered = [item for item in parent2 if item not in child]
    p2_idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = p2_filtered[p2_idx]
            p2_idx += 1
    return child

def mutate_swap(sequence: list) -> list:
    seq = list(sequence)
    if len(seq) >= 2:
        i, j = random.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]
    return seq

def mutate_insert(sequence: list) -> list:
    seq = list(sequence)
    if len(seq) >= 2:
        elem = seq.pop(random.randrange(len(seq)))
        seq.insert(random.randint(0, len(seq)), elem)
    return seq

def mutate_inversion(sequence: list) -> list:
    seq = list(sequence)
    if len(seq) >= 2:
        i, j = sorted(random.sample(range(len(seq)), 2))
        seq[i:j + 1] = reversed(seq[i:j + 1])
    return seq

def mutate_scramble(sequence: list) -> list:
    seq = list(sequence)
    if len(seq) >= 2:
        i, j = sorted(random.sample(range(len(seq)), 2))
        sub_seq = seq[i:j + 1]
        random.shuffle(sub_seq)
        seq[i:j + 1] = sub_seq
    return seq


# --- GA Operators with Lineage Support ---
class PermutationPlacementSampling(Sampling):
    def __init__(self, N: int, slack_bounds: np.ndarray):
        super().__init__()
        self.N = N
        self.min_s = slack_bounds[0]
        self.max_s = slack_bounds[1]

    def _do(self, problem, n_samples, **kwargs):
        X_out = []
        for _ in range(n_samples):
            ind = {
                "id": uuid.uuid4().hex,
                "parents": ["None", "None"],
                "X": np.random.permutation(self.N),
                "Y": np.random.permutation(self.N),
                "slack": np.random.uniform(self.min_s, self.max_s, (4, self.N)),
                "rot": np.random.randint(0, 2, self.N)
            }
            X_out.append([ind])
        return np.array(X_out, dtype=object)


class PermutationPlacementCrossover(Crossover):
    def __init__(self, config: dict):
        super().__init__(n_parents=2, n_offsprings=2)
        self.prob_top = config["topology"]["crossover"]["probability"]
        self.prob_slack = config["slack"]["crossover"]["probability"]
        self.blx_alpha = config["slack"]["crossover"]["alpha"]
        self.rot_prob = config["rotation"]["crossover"]["probability"]

    def _do(self, problem: Problem, X: np.ndarray, **kwargs) -> np.ndarray:
        n_parents, n_matings = X.shape[:2]
        Y_out = np.empty((self.n_offsprings, n_matings, problem.n_var), dtype=object)

        min_s, max_s = problem.min_s, problem.max_s

        for i in range(n_matings):
            p1_raw, p2_raw = X[0, i], X[1, i]
            p1 = p1_raw[0] if isinstance(p1_raw, (list, np.ndarray)) else p1_raw
            p2 = p2_raw[0] if isinstance(p2_raw, (list, np.ndarray)) else p2_raw

            p1_id = p1.get("id", "Unknown")
            p2_id = p2.get("id", "Unknown")

            if random.random() < self.prob_top:
                c1_x = np.array(ox_crossover(p1["X"].tolist(), p2["X"].tolist()))
                c2_x = np.array(ox_crossover(p2["X"].tolist(), p1["X"].tolist()))
                c1_y = np.array(ox_crossover(p1["Y"].tolist(), p2["Y"].tolist()))
                c2_y = np.array(ox_crossover(p2["Y"].tolist(), p1["Y"].tolist()))
            else:
                c1_x, c2_x = np.copy(p1["X"]), np.copy(p2["X"])
                c1_y, c2_y = np.copy(p1["Y"]), np.copy(p2["Y"])

            N_chiplets = p1["slack"].shape[1]
            slack_mask = np.random.rand(N_chiplets) < self.prob_slack
            
            gamma = np.minimum(p1["slack"], p2["slack"])
            delta = np.maximum(p1["slack"], p2["slack"])
            diff = delta - gamma
            
            lower = gamma - self.blx_alpha * diff
            upper = delta + self.blx_alpha * diff
            
            blended_c1 = np.clip(np.random.uniform(lower, upper), min_s, max_s)
            blended_c2 = np.clip(np.random.uniform(lower, upper), min_s, max_s)
            
            c1_slack = np.where(slack_mask, blended_c1, p1["slack"])
            c2_slack = np.where(slack_mask, blended_c2, p2["slack"])

            if random.random() < self.rot_prob:
                rot_mask = np.random.randint(0, 2, N_chiplets).astype(bool)
                c1_rot = np.where(rot_mask, p1["rot"], p2["rot"])
                c2_rot = np.where(rot_mask, p2["rot"], p1["rot"])
            else:
                c1_rot, c2_rot = np.copy(p1["rot"]), np.copy(p2["rot"])

            Y_out[0, i, 0] = {"id": uuid.uuid4().hex, "parents": [p1_id, p2_id], "X": c1_x, "Y": c1_y, "slack": c1_slack, "rot": c1_rot}
            Y_out[1, i, 0] = {"id": uuid.uuid4().hex, "parents": [p1_id, p2_id], "X": c2_x, "Y": c2_y, "slack": c2_slack, "rot": c2_rot}

        return Y_out

class PermutationPlacementMutation(Mutation):
    def __init__(self, N: int, config: dict):
        super().__init__()
        self.N = N
        self.immigrant_prob = config["immigrant_probability"]
        
        top_mut = config["topology"]["mutation"]
        self.top_prob = top_mut["probability"]
        top_weights = top_mut["weights"]
        self.top_ops = ["swap", "insert", "inversion", "joint_sequence_swap", "scramble"]
        self.top_weights = [
            top_weights["swap"], 
            top_weights["insert"], 
            top_weights["inversion"], 
            top_weights["joint_sequence_swap"],
            top_weights["scramble"]
        ]

        slack_mut = config["slack"]["mutation"]
        self.slack_prob = slack_mut["probability"]
        slack_weights = slack_mut["weights"]
        self.slack_ops = ["noise", "uniform_reset"]
        self.slack_weights = [slack_weights["noise"], slack_weights["uniform_reset"]]
        self.noise_scale = slack_mut["params"]["scale"]
        
        self.rot_prob = config["rotation"]["mutation"]["probability"]
        
        self.generation_counts = {op: 0 for op in self.top_ops + self.slack_ops}

    def reset_counts(self):
        self.generation_counts = {op: 0 for op in self.top_ops + self.slack_ops}

    def _do(self, problem, X, **kwargs) -> np.ndarray:
        Y = np.copy(X)
        min_s, max_s = problem.min_s, problem.max_s

        for i in range(len(Y)):
            is_nested = isinstance(Y[i], (np.ndarray, list)) and len(Y[i]) == 1
            ind = copy.deepcopy(Y[i][0] if is_nested else Y[i])

            if random.random() < self.immigrant_prob:
                ind = {
                    "id": uuid.uuid4().hex,
                    "parents": ["Immigrant", "Immigrant"],
                    "X": np.random.permutation(self.N),
                    "Y": np.random.permutation(self.N),
                    "slack": np.random.uniform(min_s, max_s, (4, self.N)),
                    "rot": np.random.randint(0, 2, self.N)
                }
                if is_nested: Y[i][0] = ind
                else: Y[i] = ind
                continue 

            mutated = False

            flip_mask = np.random.rand(self.N) < self.rot_prob
            if np.any(flip_mask):
                ind["rot"] = np.where(flip_mask, 1 - ind["rot"], ind["rot"])
                mutated = True

            if random.random() < self.top_prob:
                chosen_top = random.choices(self.top_ops, weights=self.top_weights, k=1)[0]
                self.generation_counts[chosen_top] += 1
                mutated = True
                
                if chosen_top == "swap":
                    ind["X"] = np.array(mutate_swap(ind["X"].tolist()))
                    ind["Y"] = np.array(mutate_swap(ind["Y"].tolist()))
                elif chosen_top == "insert":
                    ind["X"] = np.array(mutate_insert(ind["X"].tolist()))
                    ind["Y"] = np.array(mutate_insert(ind["Y"].tolist()))
                elif chosen_top == "inversion":
                    ind["X"] = np.array(mutate_inversion(ind["X"].tolist()))
                    ind["Y"] = np.array(mutate_inversion(ind["Y"].tolist()))
                elif chosen_top == "joint_sequence_swap" and self.N >= 2:
                    idx1, idx2 = random.sample(range(self.N), 2)
                    ind["X"][idx1], ind["X"][idx2] = ind["X"][idx2], ind["X"][idx1]
                    ind["Y"][idx1], ind["Y"][idx2] = ind["Y"][idx2], ind["Y"][idx1]
                elif chosen_top == "scramble":
                    ind["X"] = np.array(mutate_scramble(ind["X"].tolist()))
                    ind["Y"] = np.array(mutate_scramble(ind["Y"].tolist()))

            if random.random() < self.slack_prob:
                chosen_slack = random.choices(self.slack_ops, weights=self.slack_weights, k=1)[0]
                self.generation_counts[chosen_slack] += 1
                mutated = True
                
                if chosen_slack == "noise":
                    sigma = self.noise_scale * (max_s - min_s)
                    noise = np.random.normal(0, sigma, (4, self.N))
                    ind["slack"] = np.clip(ind["slack"] + noise, min_s, max_s)
                elif chosen_slack == "uniform_reset":
                    ind["slack"] = np.random.uniform(min_s, max_s, (4, self.N))

            if mutated:
                ind["parents"] = [ind.get("id", "Unknown"), "Mutated"]
                ind["id"] = uuid.uuid4().hex

            if is_nested: Y[i][0] = ind
            else: Y[i] = ind

        return Y


# --- Evaluation Problem ---
class ChipletPlacementProblem(Problem):
    def __init__(self, N: int, base_dims: np.ndarray, slack_bounds: np.ndarray, adj_list: list, ar_bounds: tuple, **kwargs):
        self.N = N
        self.base_widths = base_dims[0]
        self.base_heights = base_dims[1]
        self.min_s = slack_bounds[0]
        self.max_s = slack_bounds[1]
        self.adj_list = adj_list
        
        self.use_hpwl = kwargs.get('use_hpwl', True)
        self.use_area = kwargs.get('use_area', True)
        self.use_warpage = kwargs.get('use_warpage', True)
        self.ar_bounds = ar_bounds
        self.device = kwargs.get("device", "cuda")
        self.chunk_size = kwargs.get("chunk_size", 100)
        
        super().__init__(n_var=1, n_obj=3, n_constr=2)

    def _evaluate(self, X_pop: np.ndarray, out: dict, *args, **kwargs) -> None:
        pop_size = len(X_pop)
        
        x_coords_list = np.zeros((pop_size, self.N), dtype=np.float32)
        y_coords_list = np.zeros((pop_size, self.N), dtype=np.float32)
        total_w_list = np.zeros(pop_size, dtype=np.float32)
        total_h_list = np.zeros(pop_size, dtype=np.float32)
        slack_mats = np.zeros((pop_size, 4, self.N), dtype=np.float32)
        eff_w_list = np.zeros((pop_size, self.N), dtype=np.float32)
        eff_h_list = np.zeros((pop_size, self.N), dtype=np.float32)

        for idx, ind in enumerate(X_pop):
            actual_ind = ind[0] if isinstance(ind, (np.ndarray, list)) else ind
            r = actual_ind["rot"]
            
            eff_w = np.where(r, self.base_heights, self.base_widths)
            eff_h = np.where(r, self.base_widths, self.base_heights)
            
            eff_w_list[idx] = eff_w
            eff_h_list[idx] = eff_h
            
            eff_slack = np.copy(actual_ind["slack"])
            for i in range(self.N):
                if r[i]:
                    eff_slack[:, i] = np.roll(actual_ind["slack"][:, i], shift=1)
            
            slack_mats[idx] = eff_slack
            
            x_coords, y_coords, total_w, total_h = seqpair.sequence_pair_to_placement(
                actual_ind["X"], actual_ind["Y"], eff_w, eff_h, eff_slack
            )
            
            x_coords_list[idx] = x_coords
            y_coords_list[idx] = y_coords
            total_w_list[idx] = total_w
            total_h_list[idx] = total_h

        ar = np.where(total_h_list > 0, total_w_list / total_h_list, 1.0)
        G = np.column_stack([self.ar_bounds[0] - ar, ar - self.ar_bounds[1]])
        
        F = np.zeros((pop_size, 3), dtype=np.float32)
        W = None
        
        if self.use_hpwl:
            F[:, 0] = hpwl_batch(x_coords_list, y_coords_list, eff_w_list, eff_h_list, self.adj_list, slack_mats)
            
        if self.use_area:
            F[:, 1] = area_batch(total_w_list, total_h_list)
            
        if self.use_warpage:
            W, rms_vals = warpage_batch(
                x_coords_list, y_coords_list, total_w_list, total_h_list, 
                eff_w_list, eff_h_list, slack_mats, 
                device=self.device, chunk_size=self.chunk_size
            )
            F[:, 2] = rms_vals

        out["F"] = F
        out["G"] = G
        if W is not None:
            out["warpage_maps"] = W
        
        out["x_coords"] = x_coords_list
        out["y_coords"] = y_coords_list
        out["eff_w"] = eff_w_list  
        out["eff_h"] = eff_h_list  
        out["total_w"] = total_w_list
        out["total_h"] = total_h_list


class HDF5HistoryCallback(Callback):
    def __init__(self, 
                 mutation_operator, 
                 filename: str = "history.h5", 
                 seed: int = None, 
                 id_to_name: dict = None,
                 adj_list: list = None,
                 model_file: str = None):
        super().__init__()
        self.filename = filename
        self.mutation_op = mutation_operator
        self.last_gen_time = time.time()

        self.topo_ops = list(getattr(mutation_operator, 'top_ops', []))
        self.slack_ops = list(getattr(mutation_operator, 'slack_ops', []))

        mut_fields = [('generation', 'i4')]
        for op in self.topo_ops:
            mut_fields.append((f"topo_{op}", 'i4'))
        for op in self.slack_ops:
            mut_fields.append((f"slack_{op}", 'i4'))

        self.dt_mut = np.dtype(mut_fields)

        with h5py.File(self.filename, "w") as h5f:
            str_dt = h5py.string_dtype(encoding='utf-8')

            meta_group = h5f.create_group("metadata")
            
            if seed is not None:
                meta_group.create_dataset("seed", data=seed)
            if model_file is not None:
                meta_group.create_dataset("model_file", data=model_file, dtype=str_dt)
                
            start_timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.last_gen_time))
            meta_group.create_dataset("start_timestamp", data=start_timestamp_str, dtype=str_dt)

            if id_to_name is not None:
                names = [id_to_name.get(i, "") for i in range(len(id_to_name))]
                meta_group.create_dataset("chiplet_id_map", data=np.array(names, dtype=str_dt))

            if adj_list is not None:
                num_nodes = len(adj_list)
                adj_matrix = np.zeros((num_nodes, num_nodes), dtype=np.int32)
                for src_id, neighbors in enumerate(adj_list):
                    for dst_id in neighbors:
                        adj_matrix[src_id, dst_id] = 1
                meta_group.create_dataset("adjacency_matrix", data=adj_matrix)
            
            generations_group = h5f.create_group("generations")
            
            dt_time = np.dtype([('generation', 'i4'), ('timestamp', 'S20'), ('wall_clock_sec', 'f8')])
            generations_group.create_dataset("generation_times", shape=(0,), maxshape=(None,), dtype=dt_time)

            generations_group.create_dataset("generations_mutation", shape=(0,), maxshape=(None,), dtype=self.dt_mut)

            dt_lineage = np.dtype([
                ('generation', 'i4'), 
                ('child_id', 'S32'), 
                ('parent1_id', 'S32'), 
                ('parent2_id', 'S32')
            ])
            generations_group.create_dataset("global_lineage", shape=(0,), maxshape=(None,), dtype=dt_lineage)

    def notify(self, algorithm) -> None:
        current_time = time.time()
        wall_clock_sec = current_time - self.last_gen_time
        self.last_gen_time = current_time

        gen = algorithm.n_gen
        pop = algorithm.pop

        F = pop.get("F")
        G = pop.get("G")
        W = pop.get("warpage_maps")
        
        ranks = pop.get("rank")
        if ranks is None or (isinstance(ranks, np.ndarray) and ranks.dtype == object and (ranks == None).any()):
            ranks = np.zeros(len(pop), dtype=np.int32)
        else:
            ranks = ranks.astype(np.int32)

        x1 = pop.get("x_coords")
        y1 = pop.get("y_coords")
        eff_w = pop.get("eff_w")
        eff_h = pop.get("eff_h")
        
        if x1 is not None and y1 is not None and eff_w is not None and eff_h is not None:
            x2 = x1 + eff_w
            y2 = y1 + eff_h
            bboxes = np.stack([x1, y1, x2, y2], axis=-1)
        else:
            bboxes = None
            
        total_w = pop.get("total_w")
        total_h = pop.get("total_h")

        X_seq, Y_seq, slack_mat = [], [], []
        lineage_data = []
        rot_seq = []

        dt_lineage = np.dtype([
            ('generation', 'i4'), 
            ('child_id', 'S32'), 
            ('parent1_id', 'S32'), 
            ('parent2_id', 'S32')
        ])

        for ind in pop:
            actual_ind = ind.X[0] if isinstance(ind.X, (np.ndarray, list)) else ind.X
            X_seq.append(actual_ind["X"])
            Y_seq.append(actual_ind["Y"])
            
            r = actual_ind["rot"]
            eff_slack = np.copy(actual_ind["slack"])
            for i in range(len(r)):
                if r[i]:
                    eff_slack[:, i] = np.roll(actual_ind["slack"][:, i], shift=1)
            slack_mat.append(eff_slack)
            
            child_id = actual_ind.get("id", "None").encode("utf-8")
            p_list = actual_ind.get("parents", ["None", "None"])
            p1_id = p_list[0].encode("utf-8")
            p2_id = p_list[1].encode("utf-8")
            
            lineage_data.append((gen, child_id, p1_id, p2_id))
            rot_seq.append(actual_ind["rot"])

        X_seq = np.array(X_seq)
        Y_seq = np.array(Y_seq)
        slack_mat = np.array(slack_mat)
        lineage_arr = np.array(lineage_data, dtype=dt_lineage)
        num_individuals = len(lineage_arr)
        rot_seq = np.array(rot_seq)

        active_mutation_op = getattr(getattr(algorithm, "mating", None), "mutation", self.mutation_op)

        mut_row_values = [gen]
        for op in self.topo_ops:
            mut_row_values.append(active_mutation_op.generation_counts.get(op, 0))
        for op in self.slack_ops:
            mut_row_values.append(active_mutation_op.generation_counts.get(op, 0))

        mut_row_arr = np.array([tuple(mut_row_values)], dtype=self.dt_mut)

        with h5py.File(self.filename, "a") as h5f:
            generations_group = h5f["generations"]
            
            timing_ds = generations_group["generation_times"]
            idx = timing_ds.shape[0]
            timing_ds.resize((idx + 1,))
            timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time)).encode('utf-8')
            timing_ds[idx] = (gen, timestamp_str, wall_clock_sec)

            mut_ds = generations_group["generations_mutation"]
            mut_ds.resize((idx + 1,))
            mut_ds[idx:] = mut_row_arr

            lin_ds = generations_group["global_lineage"]
            idx_lin = lin_ds.shape[0]
            lin_ds.resize((idx_lin + num_individuals,))
            lin_ds[idx_lin:] = lineage_arr

            gen_group = generations_group.create_group(f"gen{gen}")

            geno_group = gen_group.create_group("genotype")
            geno_group.create_dataset("X", data=X_seq, compression="gzip")
            geno_group.create_dataset("Y", data=Y_seq, compression="gzip")
            geno_group.create_dataset("slack", data=slack_mat, compression="gzip")
            geno_group.create_dataset("rot", data=rot_seq, compression="gzip")

            pheno_group = gen_group.create_group("phenotype")
            
            if bboxes is not None:
                pheno_group.create_dataset("bboxes", data=bboxes, compression="gzip")
                
            pheno_group.create_dataset("total_w", data=total_w, compression="gzip")
            pheno_group.create_dataset("total_h", data=total_h, compression="gzip")
            
            if W is not None:
                pheno_group.create_dataset("warpage_maps", data=W, compression="gzip")

            fit_group = gen_group.create_group("fitness")
            fit_group.create_dataset("F", data=F, compression="gzip")
            fit_group.create_dataset("G", data=G, compression="gzip")
            fit_group.create_dataset("rank", data=ranks.astype(np.int32), compression="gzip")
            
            active_mutation_op.reset_counts()