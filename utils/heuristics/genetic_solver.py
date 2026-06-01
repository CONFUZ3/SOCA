from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GAConfig:
    """Configuration parameters shared by all GA solvers."""

    population_size: int = 60
    elite_fraction: float = 0.15
    mutation_rate: float = 0.02
    crossover_rate: float = 0.9
    tournament_size: int = 3
    max_generations: int = 750
    stagnation_generations: int = 80
    time_limit_seconds: float = 60.0
    random_seed: Optional[int] = None


class _BinaryGeneticOptimizer:
    """Generic binary genetic algorithm with optional cardinality constraints."""

    def __init__(
        self,
        n_genes: int,
        config: GAConfig,
        fixed_ones: Optional[int] = None,
        minimum_ones: int = 1,
    ) -> None:
        self.n_genes = int(n_genes)
        self.config = config
        self.fixed_ones = fixed_ones
        self.minimum_ones = max(0, minimum_ones)
        self.rng = np.random.default_rng(config.random_seed)

        if self.fixed_ones is not None and self.fixed_ones > self.n_genes:
            raise ValueError("fixed_ones cannot exceed number of genes")

    def run(
        self,
        evaluate_fn: Callable[[np.ndarray], Tuple[float, Dict[str, Any]]],
        initial_solution: Optional[np.ndarray] = None,
        time_limit: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Run the GA and return statistics plus best solution payload."""

        if self.n_genes <= 0:
            raise ValueError("Number of genes must be positive")

        time_budget = float(time_limit or self.config.time_limit_seconds or 0)
        logger.info(f"GA: Starting optimization with time budget: {time_budget:.2f} seconds, max generations: {self.config.max_generations}")
        pop = self._initialize_population(initial_solution)
        logger.info(f"GA: Initialized population of size {pop.shape[0]} with {self.n_genes} genes")

        best_individual: Optional[np.ndarray] = None
        best_objective = np.inf
        best_payload: Optional[Dict[str, Any]] = None

        generation = 0
        stagnant_generations = 0
        evaluations = 0
        start = time.time()
        timed_out = False

        while generation < self.config.max_generations:
            now = time.time()
            elapsed = now - start
            if time_budget and elapsed >= time_budget:
                timed_out = True
                logger.info(f"GA: Time budget reached at generation {generation}, elapsed: {elapsed:.2f}s")
                break

            objectives = np.empty(pop.shape[0], dtype=float)
            payloads: List[Optional[Dict[str, Any]]] = [None] * pop.shape[0]

            for idx, individual in enumerate(pop):
                obj, payload = evaluate_fn(individual)
                evaluations += 1
                objectives[idx] = obj
                payloads[idx] = payload

                if obj < best_objective:
                    best_objective = obj
                    best_individual = individual.copy()
                    best_payload = payload
                    stagnant_generations = 0

            stagnant_generations += 1
            if stagnant_generations >= self.config.stagnation_generations:
                logger.info(f"GA: Stagnation limit reached at generation {generation} (stagnant for {stagnant_generations} generations)")
                break

            next_population = self._build_next_generation(pop, objectives)
            pop = next_population
            generation += 1
            
            # Log progress every 50 generations
            if generation % 50 == 0:
                elapsed = time.time() - start
                logger.info(f"GA: Generation {generation}, best objective: {best_objective:.4f}, elapsed: {elapsed:.2f}s, evaluations: {evaluations}")

        elapsed = time.time() - start
        best_obj_str = f"{best_objective:.4f}" if np.isfinite(best_objective) else "inf"
        logger.info(f"GA: Optimization completed - generations: {generation}, evaluations: {evaluations}, elapsed: {elapsed:.2f}s, timed_out: {timed_out}, best_objective: {best_obj_str}")
        return {
            "best_individual": best_individual,
            "best_objective": float(best_objective) if np.isfinite(best_objective) else None,
            "best_payload": best_payload,
            "generations": generation,
            "evaluations": evaluations,
            "elapsed": elapsed,
            "timed_out": timed_out,
        }

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _initialize_population(self, initial_solution: Optional[np.ndarray]) -> np.ndarray:
        pop_size = max(4, self.config.population_size)
        population = np.zeros((pop_size, self.n_genes), dtype=np.int8)

        start_idx = 0
        if initial_solution is not None:
            seed = np.array(initial_solution, dtype=np.int8)
            if seed.shape[0] != self.n_genes:
                raise ValueError("Initial solution length mismatch")
            population[0] = self._repair(seed)
            start_idx = 1

        for i in range(start_idx, pop_size):
            population[i] = self._create_individual()

        return population

    def _create_individual(self) -> np.ndarray:
        if self.fixed_ones is not None:
            idxs = self.rng.choice(self.n_genes, size=self.fixed_ones, replace=False)
            individual = np.zeros(self.n_genes, dtype=np.int8)
            individual[idxs] = 1
        else:
            probability = max(self.minimum_ones / max(1, self.n_genes), 0.2)
            individual = (self.rng.random(self.n_genes) < probability).astype(np.int8)
        return self._repair(individual)

    def _repair(self, individual: np.ndarray) -> np.ndarray:
        repaired = individual.copy()
        ones = repaired.sum()

        if self.fixed_ones is not None:
            target = self.fixed_ones
            if ones > target:
                ones_idx = np.where(repaired == 1)[0]
                drop = self.rng.choice(ones_idx, size=ones - target, replace=False)
                repaired[drop] = 0
            elif ones < target:
                zeros_idx = np.where(repaired == 0)[0]
                if zeros_idx.size < (target - ones):
                    repaired[:] = 0
                    repaired[:target] = 1
                else:
                    add = self.rng.choice(zeros_idx, size=target - ones, replace=False)
                    repaired[add] = 1
            return repaired

        # Variable-cardinality case: enforce minimum number of facilities
        if ones < self.minimum_ones:
            zeros_idx = np.where(repaired == 0)[0]
            add_count = min(len(zeros_idx), self.minimum_ones - ones)
            if add_count > 0:
                add = self.rng.choice(zeros_idx, size=add_count, replace=False)
                repaired[add] = 1
        elif ones == 0 and self.minimum_ones == 0:
            repaired[self.rng.integers(0, self.n_genes)] = 1
        return repaired

    def _build_next_generation(self, population: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        pop_size = population.shape[0]
        elite_count = max(1, int(pop_size * self.config.elite_fraction))
        elite_indices = np.argsort(objectives)[:elite_count]
        next_pop = [population[idx].copy() for idx in elite_indices]

        while len(next_pop) < pop_size:
            parent1 = self._select_parent(population, objectives)
            if self.rng.random() < self.config.crossover_rate:
                parent2 = self._select_parent(population, objectives)
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1 = parent1.copy()
                child2 = self._select_parent(population, objectives).copy()

            next_pop.append(self._mutate(child1))
            if len(next_pop) < pop_size:
                next_pop.append(self._mutate(child2))

        return np.array(next_pop, dtype=np.int8)

    def _select_parent(self, population: np.ndarray, objectives: np.ndarray) -> np.ndarray:
        best_idx = None
        for _ in range(self.config.tournament_size):
            cand_idx = int(self.rng.integers(0, population.shape[0]))
            if best_idx is None or objectives[cand_idx] < objectives[best_idx]:
                best_idx = cand_idx
        return population[best_idx].copy()

    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.n_genes <= 1:
            return parent1.copy(), parent2.copy()
        point = int(self.rng.integers(1, self.n_genes))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return self._repair(child1), self._repair(child2)

    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        mutated = individual.copy()
        if self.fixed_ones is not None:
            if self.rng.random() < self.config.mutation_rate:
                ones_idx = np.where(mutated == 1)[0]
                zeros_idx = np.where(mutated == 0)[0]
                if ones_idx.size > 0 and zeros_idx.size > 0:
                    off = self.rng.choice(ones_idx)
                    on = self.rng.choice(zeros_idx)
                    mutated[off] = 0
                    mutated[on] = 1
            return mutated

        mutation_mask = self.rng.random(self.n_genes) < self.config.mutation_rate
        mutated[mutation_mask] = 1 - mutated[mutation_mask]
        return self._repair(mutated)


# -------------------------------------------------------------------------- #
# Problem-specific heuristics
# -------------------------------------------------------------------------- #

class PMedianGeneticSolver:
    """Genetic algorithm fallback for the P-Median problem."""

    def __init__(self, config: GAConfig) -> None:
        self.config = config

    def solve(
        self,
        distance_matrix: np.ndarray,
        demand_weights: np.ndarray,
        p: int,
        objective_type: str = "total",
        initial_solution: Optional[np.ndarray] = None,
        time_budget_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        logger.info(f"P-Median GA: Starting solve with p={p}, objective_type={objective_type}, time_budget={time_budget_seconds:.2f}s" if time_budget_seconds else f"P-Median GA: Starting solve with p={p}, objective_type={objective_type}")
        n_demand, n_candidates = distance_matrix.shape
        optimizer = _BinaryGeneticOptimizer(
            n_genes=n_candidates,
            config=self.config,
            fixed_ones=int(p),
            minimum_ones=int(min(1, p)),
        )

        weights = demand_weights.astype(float)
        total_weight = float(np.sum(weights))

        def evaluate(individual: np.ndarray) -> Tuple[float, Dict[str, Any]]:
            open_idx = np.where(individual == 1)[0]
            if open_idx.size == 0:
                return np.inf, {}

            dm = distance_matrix[:, open_idx]
            assign_cols = np.argmin(dm, axis=1)
            selected_facilities = open_idx[assign_cols]
            distances = dm[np.arange(n_demand), assign_cols]
            weighted_total = float(np.dot(distances, weights))
            avg_distance = weighted_total / total_weight if total_weight > 0 else weighted_total

            objective = weighted_total if objective_type == "total" else avg_distance
            assignments = {i: int(selected_facilities[i]) for i in range(n_demand)}
            payload = {
                "selected": open_idx.tolist(),
                "assignments": assignments,
                "total_weighted_distance": weighted_total,
                "average_distance": avg_distance,
                "feasible": True,
            }
            return objective, payload

        result = optimizer.run(
            evaluate_fn=evaluate,
            initial_solution=initial_solution,
            time_limit=time_budget_seconds,
        )

        if not result.get("best_payload"):
            raise RuntimeError("GA failed to produce a feasible P-Median solution")

        payload = result["best_payload"]
        objective_value = result["best_objective"]
        return {
            "status": "feasible",
            "objective_value": float(objective_value),
            "selected_facilities": payload["selected"],
            "assignments": payload["assignments"],
            "solver_details": self._solver_details(result, "p-median"),
        }

    def _solver_details(self, result: Dict[str, Any], formulation: str) -> Dict[str, Any]:
        return {
            "solver": "ga",
            "formulation": formulation,
            "population_size": self.config.population_size,
            "generations": result.get("generations"),
            "evaluations": result.get("evaluations"),
            "elapsed_seconds": result.get("elapsed"),
            "timed_out": result.get("timed_out", False),
        }


class PCenterGeneticSolver:
    """Genetic algorithm fallback for the P-Center problem."""

    SUPPORTED_VARIANTS = {"vertex", "weighted", "conditional"}

    def __init__(self, config: GAConfig) -> None:
        self.config = config

    def supports_variant(self, variant: str) -> bool:
        return variant in self.SUPPORTED_VARIANTS

    def solve(
        self,
        distance_matrix: np.ndarray,
        p: int,
        initial_solution: Optional[np.ndarray] = None,
        time_budget_seconds: Optional[float] = None,
        variant: str = "vertex",
        demand_weights: Optional[np.ndarray] = None,
        existing: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        variant = variant or "vertex"
        logger.info(f"P-Center GA: Starting solve with variant={variant}, p={p}, time_budget={time_budget_seconds:.2f}s" if time_budget_seconds else f"P-Center GA: Starting solve with variant={variant}, p={p}")
        n_demand, n_candidates = distance_matrix.shape

        if not self.supports_variant(variant):
            raise ValueError(f"Variant '{variant}' is not supported by GA fallback")

        weights = demand_weights.astype(float) if demand_weights is not None else np.ones(n_demand)

        # Conditional variant: optimise over the non-existing candidates only and
        # union the (always-open) existing facilities back in when evaluating.
        existing_idx = sorted(set(int(j) for j in (existing or []) if 0 <= int(j) < n_candidates))
        if variant == "conditional":
            free = np.array([j for j in range(n_candidates) if j not in set(existing_idx)], dtype=int)
            n_genes = int(free.size)
            existing_arr = np.array(existing_idx, dtype=int)
            # Translate the incumbent mask into the restricted gene space.
            restricted_initial: Optional[np.ndarray] = None
            if initial_solution is not None:
                projected = np.asarray(initial_solution, dtype=np.int8)[free]
                if int(projected.sum()) == int(p):
                    restricted_initial = projected
        else:
            free = np.arange(n_candidates, dtype=int)
            n_genes = n_candidates
            existing_arr = np.array([], dtype=int)
            restricted_initial = initial_solution

        optimizer = _BinaryGeneticOptimizer(
            n_genes=n_genes,
            config=self.config,
            fixed_ones=int(p),
            minimum_ones=int(min(1, p)),
        )

        def evaluate(individual: np.ndarray) -> Tuple[float, Dict[str, Any]]:
            new_idx = free[np.where(individual == 1)[0]]
            open_idx = np.concatenate([new_idx, existing_arr]) if existing_arr.size else new_idx
            if open_idx.size == 0:
                return np.inf, {}
            dm = distance_matrix[:, open_idx]
            assign_cols = np.argmin(dm, axis=1)
            selected_facilities = open_idx[assign_cols]
            nearest = dm[np.arange(n_demand), assign_cols]
            if variant == "weighted":
                objective = float(np.max(weights * nearest))
            else:
                objective = float(np.max(nearest))
            average_distance = float(np.mean(nearest))
            assignments = {i: int(selected_facilities[i]) for i in range(n_demand)}
            payload = {
                "selected": [int(j) for j in open_idx],
                "assignments": assignments,
                "max_distance": objective,
                "average_distance": average_distance,
                "feasible": True,
            }
            return objective, payload

        result = optimizer.run(
            evaluate_fn=evaluate,
            initial_solution=restricted_initial,
            time_limit=time_budget_seconds,
        )

        if not result.get("best_payload"):
            raise RuntimeError("GA failed to produce a feasible P-Center solution")

        payload = result["best_payload"]
        return {
            "status": "feasible",
            "objective_value": payload["max_distance"],
            "selected_facilities": payload["selected"],
            "assignments": payload["assignments"],
            "solver_details": self._solver_details(result, f"p-center-{variant}"),
        }

    def _solver_details(self, result: Dict[str, Any], formulation: str) -> Dict[str, Any]:
        return {
            "solver": "ga",
            "formulation": formulation,
            "population_size": self.config.population_size,
            "generations": result.get("generations"),
            "evaluations": result.get("evaluations"),
            "elapsed_seconds": result.get("elapsed"),
            "timed_out": result.get("timed_out", False),
        }


class LSCPGeneticSolver:
    """Genetic algorithm fallback for the LSCP solver."""

    SUPPORTED_VARIANTS = {"base", "backup", "conditional", "probabilistic", "partial"}

    def __init__(self, config: GAConfig) -> None:
        self.config = config

    def supports_variant(self, variant: str) -> bool:
        return variant in self.SUPPORTED_VARIANTS

    def solve(
        self,
        coverage_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        time_budget_seconds: Optional[float] = None,
        initial_solution: Optional[np.ndarray] = None,
        variant: str = "base",
        k_coverage: int = 2,
        demand_weights: Optional[np.ndarray] = None,
        reliability: Optional[np.ndarray] = None,
        coverage_reliability: float = 0.95,
        coverage_fraction: float = 0.95,
        existing: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        variant = variant or "base"
        logger.info(f"LSCP GA: Starting solve with variant={variant}, time_budget={time_budget_seconds:.2f}s" if time_budget_seconds else f"LSCP GA: Starting solve with variant={variant}")
        n_demand, n_candidates = coverage_matrix.shape

        if not self.supports_variant(variant):
            raise ValueError(f"Variant '{variant}' is not supported by GA fallback")

        weights = demand_weights.astype(float) if demand_weights is not None else np.ones(n_demand)
        total_weight = float(np.sum(weights))
        reliability = reliability.astype(float) if reliability is not None else np.ones(n_candidates)
        k_required = max(1, int(k_coverage))
        alpha = float(coverage_reliability)
        fraction = float(coverage_fraction)
        penalty_scale = max(10, n_candidates)

        # Conditional variant: optimise over the non-existing candidates only and
        # union the (always-open) existing facilities back in when evaluating; the
        # objective counts only the additional facilities.
        existing_idx = sorted(set(int(j) for j in (existing or []) if 0 <= int(j) < n_candidates))
        if variant == "conditional":
            free = np.array([j for j in range(n_candidates) if j not in set(existing_idx)], dtype=int)
            existing_arr = np.array(existing_idx, dtype=int)
            restricted_initial: Optional[np.ndarray] = None
            if initial_solution is not None and free.size:
                restricted_initial = np.asarray(initial_solution, dtype=np.int8)[free]
        else:
            free = np.arange(n_candidates, dtype=int)
            existing_arr = np.array([], dtype=int)
            restricted_initial = initial_solution

        optimizer = _BinaryGeneticOptimizer(
            n_genes=int(free.size),
            config=self.config,
            fixed_ones=None,
            minimum_ones=1,
        )

        def evaluate(individual: np.ndarray) -> Tuple[float, Dict[str, Any]]:
            new_idx = free[np.where(individual == 1)[0]]
            open_idx = np.concatenate([new_idx, existing_arr]) if existing_arr.size else new_idx
            if open_idx.size == 0:
                return np.inf, {}

            cover_slice = coverage_matrix[:, open_idx]
            cover_counts = cover_slice.sum(axis=1)
            # Facility count for the objective: conditional counts only the additions.
            facility_count = int(new_idx.size) if variant == "conditional" else int(open_idx.size)

            if variant == "backup":
                covered_mask = cover_counts >= k_required
                violations = int(np.count_nonzero(~covered_mask))
                penalty = violations * penalty_scale
            elif variant == "probabilistic":
                covered_mask = cover_counts >= 1
                prob_not_served = np.ones(n_demand)
                for facility in open_idx:
                    facility_rel = float(reliability[facility])
                    prob_not_served *= np.where(coverage_matrix[:, facility] == 1, 1.0 - facility_rel, 1.0)
                achieved = 1.0 - prob_not_served
                violations = int(np.count_nonzero(achieved < alpha - 1e-9))
                penalty = violations * penalty_scale
            elif variant == "partial":
                covered_mask = cover_counts >= 1
                covered_weight = float(np.sum(weights[covered_mask]))
                required = fraction * total_weight
                penalty = penalty_scale * max(0.0, required - covered_weight)
            else:
                # base / conditional: every demand needs at least one covering facility
                covered_mask = cover_counts >= 1
                violations = int(np.count_nonzero(~covered_mask))
                penalty = violations * penalty_scale

            objective = facility_count + penalty

            assignments: Dict[int, int] = {}
            for demand_idx in range(n_demand):
                covering = np.where(cover_slice[demand_idx] == 1)[0]
                if covering.size > 0:
                    assignments[demand_idx] = int(open_idx[covering[0]])

            uncovered = int(np.count_nonzero(~(cover_counts >= 1)))
            payload = {
                "selected": [int(j) for j in open_idx],
                "assignments": assignments,
                "uncovered": uncovered,
                "covered_pct": float((1 - uncovered / n_demand) * 100 if n_demand else 100.0),
                "feasible": penalty == 0,
            }
            return objective, payload

        result = optimizer.run(
            evaluate_fn=evaluate,
            initial_solution=restricted_initial,
            time_limit=time_budget_seconds,
        )

        payload = result.get("best_payload")
        if not payload:
            raise RuntimeError("GA failed to produce a feasible LSCP solution")

        status = "feasible" if payload.get("feasible") else "approximate"
        return {
            "status": status,
            "objective_value": len(payload["selected"]),
            "selected_facilities": payload["selected"],
            "assignments": payload["assignments"],
            "solver_details": self._solver_details(result, f"lscp-{variant}"),
        }

    def _solver_details(self, result: Dict[str, Any], formulation: str) -> Dict[str, Any]:
        return {
            "solver": "ga",
            "formulation": formulation,
            "population_size": self.config.population_size,
            "generations": result.get("generations"),
            "evaluations": result.get("evaluations"),
            "elapsed_seconds": result.get("elapsed"),
            "timed_out": result.get("timed_out", False),
        }


class MCLPGeneticSolver:
    """Genetic algorithm fallback for selected MCLP variants."""

    SUPPORTED_VARIANTS = {"classical", "budget", "multi_coverage", "backup", "probabilistic"}

    def __init__(self, config: GAConfig) -> None:
        self.config = config

    def supports_variant(self, variant: str) -> bool:
        return variant in self.SUPPORTED_VARIANTS

    def solve(
        self,
        coverage_matrix: np.ndarray,
        distance_matrix: np.ndarray,
        demand_weights: np.ndarray,
        variant: str,
        p: Optional[int],
        facility_costs: Optional[np.ndarray],
        budget: Optional[float],
        k_coverage: int,
        reliability: Optional[np.ndarray],
        initial_solution: Optional[np.ndarray] = None,
        time_budget_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        variant = variant or "classical"
        logger.info(f"MCLP GA: Starting solve with variant={variant}, p={p}, time_budget={time_budget_seconds:.2f}s" if time_budget_seconds else f"MCLP GA: Starting solve with variant={variant}, p={p}")
        n_demand, n_candidates = coverage_matrix.shape

        if not self.supports_variant(variant):
            raise ValueError(f"Variant '{variant}' is not supported by GA fallback")

        fixed_ones = None
        min_ones = 1
        if variant != "budget":
            if p is None:
                raise ValueError(f"Variant '{variant}' requires n_facilities")
            fixed_ones = int(p)
            min_ones = min(1, fixed_ones)

        optimizer = _BinaryGeneticOptimizer(
            n_genes=n_candidates,
            config=self.config,
            fixed_ones=fixed_ones,
            minimum_ones=min_ones,
        )

        weights = demand_weights.astype(float)
        total_weight = float(np.sum(weights)) if np.sum(weights) > 0 else 1.0
        facility_costs = facility_costs.astype(float) if facility_costs is not None else np.ones(n_candidates)
        reliability = reliability.astype(float) if reliability is not None else np.ones(n_candidates)
        k_required = max(1, int(k_coverage))

        budget_value = float(budget) if budget is not None else None
        penalty_scale = max(100.0, np.sum(weights))

        def evaluate(individual: np.ndarray) -> Tuple[float, Dict[str, Any]]:
            open_idx = np.where(individual == 1)[0]
            if open_idx.size == 0:
                return np.inf, {}

            cost = float(np.sum(facility_costs[open_idx]))
            budget_penalty = 0.0
            if variant == "budget" and budget_value is not None and cost > budget_value:
                budget_penalty = (cost - budget_value) * penalty_scale

            cover_slice = coverage_matrix[:, open_idx]
            cover_counts = cover_slice.sum(axis=1)
            if variant in {"multi_coverage", "backup"}:
                coverage_mask = cover_counts >= k_required
            else:
                coverage_mask = cover_slice.any(axis=1)

            if variant == "probabilistic":
                # Match the MIP's linear coverage surrogate  z_i ≤ Σ_{j∈Nᵢ} r_j x_j
                # (capped at 1) so the GA fallback optimises the *same* objective
                # the MIP was approximating, instead of the exact product form
                # (which would make MIP and GA disagree on the same instance).
                rel_open = reliability[open_idx]
                rel_sum = coverage_matrix[:, open_idx] @ rel_open
                z_lin = np.minimum(1.0, rel_sum)
                covered_weight = float(np.sum(weights * z_lin))
            else:
                covered_weight = float(np.sum(weights[coverage_mask]))

            objective = -covered_weight + budget_penalty + max(0, 0.1 * len(open_idx))

            assignments: Dict[int, int] = {}
            if open_idx.size > 0:
                dm = distance_matrix[:, open_idx]
                assign_cols = np.argmin(dm, axis=1)
                for demand_idx in range(n_demand):
                    if coverage_mask[demand_idx]:
                        facility = int(open_idx[assign_cols[demand_idx]])
                        assignments[demand_idx] = facility

            payload = {
                "selected": open_idx.tolist(),
                "assignments": assignments,
                "covered_weight": covered_weight,
                "coverage_pct": float((covered_weight / total_weight) * 100 if total_weight else 0),
                "objective_value": covered_weight,
                "z_values": {i: 1.0 if coverage_mask[i] else 0.0 for i in range(n_demand)},
            }
            return objective, payload

        result = optimizer.run(
            evaluate_fn=evaluate,
            initial_solution=initial_solution,
            time_limit=time_budget_seconds,
        )

        payload = result.get("best_payload")
        if not payload:
            raise RuntimeError("GA failed to produce a feasible MCLP solution")

        return {
            "status": "feasible",
            "objective_value": float(payload["objective_value"]),
            "selected_facilities": payload["selected"],
            "assignments": payload["assignments"],
            "z_values": payload["z_values"],
            "solver_details": self._solver_details(result, f"mclp-{variant}"),
        }

    def _solver_details(self, result: Dict[str, Any], formulation: str) -> Dict[str, Any]:
        return {
            "solver": "ga",
            "formulation": formulation,
            "population_size": self.config.population_size,
            "generations": result.get("generations"),
            "evaluations": result.get("evaluations"),
            "elapsed_seconds": result.get("elapsed"),
            "timed_out": result.get("timed_out", False),
        }


__all__ = [
    "GAConfig",
    "PMedianGeneticSolver",
    "PCenterGeneticSolver",
    "LSCPGeneticSolver",
    "MCLPGeneticSolver",
]

