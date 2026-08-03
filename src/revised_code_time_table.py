import simpy
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set
import itertools


RANDOM_SEED         = 42
SIM_HOURS           = 30 * 24
REPLICATIONS        = 16
DELAY_THRESH        = 4.0
WARMUP_HOURS        = 7 * 24
MANEUVER_SPEED_KMH  = 11  


FIXED_TRAVEL_TIMES_H: Dict[Tuple[str, str], float] = {
    ("BS", "SW1"):                     15/60,
    ("SW1", "BS"):                     15/60,
    ("SW1", "Barko"):                  15/60,
    ("Barko", "SW1"):                  15/60,
    ("SW1", "Steel_Hormozgan"):        15/60,
    ("Steel_Hormozgan", "SW1"):        15/60,
    ("SW1", "Yard_Port"):              15/60,
    ("Yard_Port", "SW2"):              15/60,
    ("SW2", "Yard_Port"):              15/60,
    ("SW2", "old_steel_and_copper"):   15/60,
    ("old_steel_and_copper", "SW2"):   15/60,
    ("SW2", "SW3"):                     5/60,
    ("SW3", "SW2"):                     5/60,
    ("SW3", "Sulfer"):                 10/60,
    ("Sulfer", "SW3"):                 10/60,
    ("SW3", "BS"):                     15/60,
}

FIXED_TRAVEL_TIMES_H_scenario_A: Dict[Tuple[str, str], float] = {
    ("BS", "Support_Stations"):                     30/60,
    ("Support_Stations", "BS"):                     30/60,
    ("Support_Stations", "Barko"):                  15/60,
    ("Barko", "Support_Stations"):                  15/60,
    ("Support_Stations", "Steel_Hormozgan"):        15/60,
    ("Steel_Hormozgan", "Support_Stations"):        15/60,
    ("Support_Stations", "Yard_Port"):              30/60,
    ("Yard_Port", "Support_Stations"):              30/60,
    ("Support_Stations", "old_steel_and_copper"):   30/60,
    ("old_steel_and_copper", "Support_Stations"):   30/60,
    ("Support_Stations", "Sulfer"):                 30/60,
    ("Sulfer", "Support_Stations"):                 30/60,
}

FIXED_TRAVEL_TIMES_H_scenario_B: Dict[Tuple[str, str], float] = {
    ("BS", "SW1"):                     15/60,
    ("SW1", "BS"):                     15/60,
    ("SW1", "Barko"):                  15/60,
    ("Barko", "SW1"):                  15/60,
    ("SW1", "Steel_Hormozgan"):        15/60,
    ("Steel_Hormozgan", "SW1"):        15/60,
    ("BS", "Yard_Port"):               30/60,   
    ("Yard_Port", "BS"):               30/60,
    ("SW1", "Yard_Port"):              15/60,
    ("Yard_Port", "SW1"):              15/60,
    ("Yard_Port", "SW2"):              15/60,
    ("SW2", "Yard_Port"):              15/60,
    ("SW2", "old_steel_and_copper"):   15/60,
    ("old_steel_and_copper", "SW2"):   15/60,
    ("SW2", "SW3"):                     5/60,
    ("SW3", "SW2"):                     5/60,
    ("SW3", "Sulfer"):                 10/60,
    ("Sulfer", "SW3"):                 10/60,
}

FIXED_TRAVEL_TIMES_H_scenario_C: Dict[Tuple[str, str], float] = {
    ("BS", "North_Yard"):                 5/60,
    ("North_Yard", "BS"):                 5/60,
    ("North_Yard", "Barko"):             15/60,
    ("Barko", "North_Yard"):             15/60,
    ("North_Yard", "Steel_Hormozgan"):   15/60,
    ("Steel_Hormozgan", "North_Yard"):   15/60,
    ("BS", "Yard_Port"):                 30/60,
    ("Yard_Port", "BS"):                 30/60,
    ("BS", "South_Yard"):                15/60,
    ("South_Yard", "BS"):                15/60,
    ("South_Yard", "old_steel_and_copper"): 15/60,
    ("old_steel_and_copper", "South_Yard"): 15/60,
    ("South_Yard", "Sulfer"):            10/60,
    ("Sulfer", "South_Yard"):            10/60,
}

REAL_DISTANCES_M: Dict[Tuple[str, str], float] = {
    ("BS",    "SW1"):                   6000,
    ("SW1",   "Barko"):                374,
    ("SW1",   "Steel_Hormozgan"):      374,
    ("SW1",   "Yard_Port"):            4000,
    ("Yard_Port", "SW2"):              2000,
    ("SW2",   "old_steel_and_copper"):  300,
    ("SW2",   "SW3"):                   1000,
    ("SW3",   "Sulfer"):                300,
    ("BS",    "Yard_Port"):            10000,
}

CALIBRATED_RATES = {
    "port_inbound":        0.0667,
    "port_outbound":       0.0500,
    "barko_inbound":       0.0417,
    "barko_outbound":      0.0333,
    "golgohar_inbound":    0.1333,
    "golgohar_outbound":   0.1250,
    "interregional_in":    0.2750,
    "interregional_out":   0.2250,
    "petroleum_inbound":   0.0250,
    "petroleum_outbound":  0.0250,
}

BASE_PORT_RATE  = 4   * (CALIBRATED_RATES["port_inbound"]     + CALIBRATED_RATES["port_outbound"])     / 2
BASE_NORTH_RATE = 4 * (CALIBRATED_RATES["barko_inbound"]    + CALIBRATED_RATES["barko_outbound"])    / 2
BASE_SOUTH_RATE = 4   * (CALIBRATED_RATES["golgohar_inbound"] + CALIBRATED_RATES["golgohar_outbound"]) / 2

AVG_WAGONS = {
    "Yard_Port":              44.4,
    "Barko":                  42.2,
    "Steel_Hormozgan":        46.9,
    "old_steel_and_copper":   43.0,
    "Sulfer":                 43.0,
}

# =============================================================================
# Travel-time lookups
# =============================================================================
def get_travel_time(u: str, v: str, speed_kmh: float = MANEUVER_SPEED_KMH) -> float:
    """Baseline-only lookup, kept for backward compatibility."""
    key = (u, v)
    if key in FIXED_TRAVEL_TIMES_H:
        return FIXED_TRAVEL_TIMES_H[key]
    rev = (v, u)
    dist_m = REAL_DISTANCES_M.get(key) or REAL_DISTANCES_M.get(rev)
    if dist_m is None:
        return 0.025
    return (dist_m / 1000.0) / speed_kmh


def _make_scenario_travel_lookup(table: Dict[Tuple[str, str], float]):
    def _get(u, v, speed_kmh: float = MANEUVER_SPEED_KMH) -> float:
        key = (u, v)
        if key in table:
            return table[key]
        rev = (v, u)
        dist_m = REAL_DISTANCES_M.get(key) or REAL_DISTANCES_M.get(rev)
        if dist_m is None:
            return 0.025
        return (dist_m / 1000.0) / speed_kmh
    return _get


def _stochastic_travel_time_for(lookup_fn, u, v, speed_kmh: float = MANEUVER_SPEED_KMH, cv: float = 0.15) -> float:
    """Return a log‑normal travel time based on the mean from lookup_fn."""
    mu = lookup_fn(u, v, speed_kmh)
    sigma = cv * mu
    if mu <= 0:
        return 0.001
    log_mu    = np.log(mu ** 2 / np.sqrt(mu ** 2 + sigma ** 2))
    log_sigma = np.sqrt(np.log(1 + (sigma / mu) ** 2))
    return max(0.001, np.random.lognormal(log_mu, log_sigma))


def stochastic_travel_time(u, v, speed_kmh=MANEUVER_SPEED_KMH, cv=0.15):
    return _stochastic_travel_time_for(get_travel_time, u, v, speed_kmh, cv)


get_travel_time_baseline   = _make_scenario_travel_lookup(FIXED_TRAVEL_TIMES_H)
get_travel_time_scenario_a = _make_scenario_travel_lookup(FIXED_TRAVEL_TIMES_H_scenario_A)

FIXED_TRAVEL_TIMES_H_scenario_B = dict(FIXED_TRAVEL_TIMES_H)
FIXED_TRAVEL_TIMES_H_scenario_B[("BS", "Yard_Port")] = 20 / 60
FIXED_TRAVEL_TIMES_H_scenario_B[("Yard_Port", "BS")] = 20 / 60
get_travel_time_scenario_b = _make_scenario_travel_lookup(FIXED_TRAVEL_TIMES_H_scenario_B)
get_travel_time_scenario_c = _make_scenario_travel_lookup(FIXED_TRAVEL_TIMES_H_scenario_C)


# =============================================================================
# Resources
# =============================================================================
class MonitoredResource(simpy.Resource):
    """Resource that tracks busy time, queue length integral, and request count."""
    def __init__(self, env, capacity):
        super().__init__(env, capacity)
        self.busy_time      = 0.0
        self.queue_integral = 0.0
        self.last_event     = env.now
        self._queue_len     = 0
        self.total_requests = 0
    def request(self, *args, **kwargs):
        self.total_requests += 1
        return super().request(*args, **kwargs)
    def _update(self, now):
        dt = now - self.last_event
        self.busy_time      += self.count * dt
        self.queue_integral += self._queue_len * dt
        self.last_event      = now
    def _do_put(self, event):
        self._update(self._env.now); super()._do_put(event)
        self._queue_len = len(self.queue); self._update(self._env.now)
    def _do_get(self, event):
        self._update(self._env.now); super()._do_get(event)
        self._queue_len = len(self.queue); self._update(self._env.now)


# =============================================================================
# Graph builders
# =============================================================================
def build_baseline_graph():
    G = nx.DiGraph()
    for (u, v), t in FIXED_TRAVEL_TIMES_H.items():
        G.add_edge(u, v, travel_time=t, distance_m=REAL_DISTANCES_M.get((u, v), REAL_DISTANCES_M.get((v, u), 250)))
    return G

def build_scenario_a_graph():
    G = nx.DiGraph()
    for (u, v), t in FIXED_TRAVEL_TIMES_H_scenario_A.items():
        G.add_edge(u, v, travel_time=t)
    return G

def build_scenario_b_graph():
    G = build_baseline_graph()
    G.add_edge("BS", "Yard_Port", travel_time=20 / 60)
    G.add_edge("Yard_Port", "BS", travel_time=20 / 60)
    return G

def build_scenario_c_graph():
    G = nx.DiGraph()
    for (u, v), t in FIXED_TRAVEL_TIMES_H_scenario_C.items():
        G.add_edge(u, v, travel_time=t)
    return G


# =============================================================================
# Switch / resource-conflict maps
# =============================================================================
def build_baseline_switches(env, cfg):
    sw = {s: MonitoredResource(env, 1) for s in ["SW1", "SW2", "SW3"]}
    esw = {
        ("BS",   "SW1"):  ["SW1"], ("SW1",  "BS"):   ["SW1"],
        ("SW1",  "Barko"):["SW1"], ("Barko","SW1"):  ["SW1"],
        ("SW1",  "Steel_Hormozgan"): ["SW1"], ("Steel_Hormozgan", "SW1"): ["SW1"],
        ("SW1",  "Yard_Port"): ["SW1"], ("Yard_Port", "SW1"): ["SW1"],
        ("Yard_Port", "SW2"): ["SW2"], ("SW2", "Yard_Port"):  ["SW2"],
        ("SW2",  "old_steel_and_copper"): ["SW2"], ("old_steel_and_copper", "SW2"):  ["SW2"],
        ("SW2",  "SW3"):   ["SW2", "SW3"], ("SW3", "SW2"):    ["SW2", "SW3"],
        ("SW3",  "Sulfer"):["SW3"],        ("Sulfer", "SW3"): ["SW3"],
        ("SW3",  "BS"): ["SW3"],
    }
    return sw, esw, {}

def build_scenario_a_switches(env, cfg):
    G = build_scenario_a_graph()
    return {}, {(u, v): [] for u, v in G.edges}, {}

def build_scenario_b_switches(env, cfg):
    sw, esw, yr = build_baseline_switches(env, cfg)
    esw[("BS", "Yard_Port")] = []
    esw[("Yard_Port", "BS")] = []
    return sw, esw, yr

def build_scenario_c_switches(env, cfg):
    esw = {
        ("BS",          "North_Yard"): [],  ("North_Yard",          "BS"): [],
        ("North_Yard",  "Barko"):      [],  ("Barko",        "North_Yard"): [],
        ("North_Yard",  "Steel_Hormozgan"): [], ("Steel_Hormozgan", "North_Yard"):  [],
        ("BS",          "Yard_Port"):  [],  ("Yard_Port",           "BS"): [],
        ("BS",          "South_Yard"): [],  ("South_Yard",          "BS"): [],
        ("South_Yard",  "old_steel_and_copper"): [], ("old_steel_and_copper", "South_Yard"):  [],
        ("South_Yard",  "Sulfer"): [], ("Sulfer", "South_Yard"): [],
    }
    yr = {yn: MonitoredResource(env, cap) for yn, cap in cfg.yard_capacities.items()}
    return {}, esw, yr


# =============================================================================
# Scenario configuration
# =============================================================================
@dataclass
class ScenarioConfig:
    name:  str
    color: str
    short: str
    bs_track_cap:  int
    bs_loco_cap:   int
    yard_track_cap: int
    yard_loco_cap:  int
    port_train_rate:  float
    north_ind_rate:   float
    south_ind_rate:   float
    port_load_time:  Tuple[float, float]
    north_load_time: Tuple[float, float]
    south_load_time: Tuple[float, float]
    port_conflict:   float = 0.0
    north_conflict:  float = 0.0
    south_conflict:  float = 0.0
    edge_capacities: dict  = field(default_factory=dict)
    graph_builder:   object = None
    switch_builder:  object = None
    yard_requirements:    dict = field(default_factory=dict)
    yard_capacities:      dict = field(default_factory=dict)
    yard_loco_capacities: dict = field(default_factory=dict)
    support_station_cap: int  = 1
    dest_loco_cap:       int  = 1
    use_paired_arrivals: bool  = False
    within_gap_hours: float = 0.67
    maneuver_time: float = 0.083

BASELINE = ScenarioConfig(
    name="Baseline_v6_1_fixed", short="Baseline", color="#E05252",
    bs_track_cap=2,  bs_loco_cap=3, yard_track_cap=2, yard_loco_cap=3,
    port_train_rate=BASE_PORT_RATE, north_ind_rate=BASE_NORTH_RATE, south_ind_rate=BASE_SOUTH_RATE,
    port_load_time=(2.17, 0.24), north_load_time=(2.22, 0.29), south_load_time=(2.40, 0.62),
    port_conflict=0.070, north_conflict=0.05, south_conflict=0.070,
    graph_builder=build_baseline_graph, switch_builder=build_baseline_switches,
    use_paired_arrivals=True, maneuver_time=0.5,
)

SCENARIO_A = ScenarioConfig(
    name="Scenario_A_v6_1_fixed", short="Scenario A", color="#4ADE80",
    bs_track_cap=2, bs_loco_cap=4, yard_track_cap=2, yard_loco_cap=2,
    port_train_rate=BASE_PORT_RATE, north_ind_rate=BASE_NORTH_RATE, south_ind_rate=BASE_SOUTH_RATE,
    port_load_time=(2.17 * 0.85, 0.20), north_load_time=(2.22 * 0.85, 0.22), south_load_time=(2.40 * 0.85, 0.45),
    port_conflict=0.03, north_conflict=0.02, south_conflict=0.05,
    edge_capacities={
        ("BS", "Support_Stations"): 2, ("Support_Stations", "BS"): 2,
        ("Support_Stations", "Yard_Port"): 2, ("Yard_Port", "Support_Stations"): 2,
    },
    graph_builder=build_scenario_a_graph, switch_builder=build_scenario_a_switches,
    support_station_cap=5, use_paired_arrivals=True, maneuver_time=0.050,
)

SCENARIO_B = ScenarioConfig(
    name="Scenario_B_v6_1_fixed", short="Scenario B", color="#FBBF24",
    bs_track_cap=2, bs_loco_cap=5, yard_track_cap=2, yard_loco_cap=5,
    port_train_rate=BASE_PORT_RATE, north_ind_rate=BASE_NORTH_RATE, south_ind_rate=BASE_SOUTH_RATE,
    port_load_time=(1.17, 0.24), north_load_time=(1.22, 0.22), south_load_time=(1.22, 0.22),
    port_conflict=0.05, north_conflict=0.05, south_conflict=0.10,
    graph_builder=build_scenario_b_graph, switch_builder=build_scenario_b_switches,
    use_paired_arrivals=True, maneuver_time=0.067,
)

SCENARIO_C = ScenarioConfig(
    name="Scenario_C_v6_1_fixed", short="Scenario C", color="#60A5FA",
    bs_track_cap=2, bs_loco_cap=3, yard_track_cap=2, yard_loco_cap=2,
    port_train_rate=BASE_PORT_RATE, north_ind_rate=BASE_NORTH_RATE, south_ind_rate=BASE_SOUTH_RATE,
    port_load_time=(0.85, 0.20), north_load_time=(0.85, 0.22), south_load_time=(0.85, 0.45),
    port_conflict=0.03, north_conflict=0.03, south_conflict=0.08,
    edge_capacities={
        ("BS", "Yard_Port"): 2, ("Yard_Port", "BS"): 2,
        ("BS", "North_Yard"): 2, ("North_Yard", "BS"): 2,
        ("BS", "South_Yard"): 2, ("South_Yard", "BS"): 2,
    },
    graph_builder=build_scenario_c_graph, switch_builder=build_scenario_c_switches,
    yard_requirements={
        "Barko":               "North_Yard", "Steel_Hormozgan":     "North_Yard",
        "old_steel_and_copper":"South_Yard", "Sulfer":              "South_Yard",
    },
    yard_capacities={"North_Yard": 2, "South_Yard": 2},
    yard_loco_capacities={"North_Yard": 2, "South_Yard": 2},
    use_paired_arrivals=True, maneuver_time=0.042,
)

ALL_SCENARIOS = [BASELINE, SCENARIO_A, SCENARIO_B, SCENARIO_C]
DEST_LIST = ["Barko", "Steel_Hormozgan", "Sulfer", "old_steel_and_copper", "Yard_Port"]

SCENARIO_TRAVEL_LOOKUP = {
    "Baseline":   get_travel_time_baseline,
    "Scenario A": get_travel_time_scenario_a,
    "Scenario B": get_travel_time_scenario_b,
    "Scenario C": get_travel_time_scenario_c,
}


def get_path_edges(G, src, dst):
    """Return list of (u, v, travel_time) for the shortest path."""
    path = nx.shortest_path(G, src, dst, weight="travel_time")
    return [(u, v, G[u][v]["travel_time"]) for u, v in zip(path[:-1], path[1:])]


# =============================================================================
# Train process and replication runner
# =============================================================================
@dataclass
class RunStats:
    dwells:           list = field(default_factory=list)
    delays:           list = field(default_factory=list)
    travel_times:      list = field(default_factory=list)
    completed:        int = 0
    moves:            int = 0
    dest_counts:      dict = field(default_factory=dict)
    station_arrivals: dict = field(default_factory=dict)
    total_generated:   int = 0
    completed_overall: int = 0
    dwells_by_dest:     dict = field(default_factory=dict)
    resource_util:      dict = field(default_factory=dict)
    resource_wait:       dict = field(default_factory=dict)


SWITCH_CROSSING_TIME = 1 / 60


def travel_path(env, G, edge_resources, switch_resources, edge_switches, src, dst,
                travel_lookup, stats=None):
    """
    Travel along the shortest path from src to dst, acquiring edge and switch resources.
    Logs station arrivals if stats is provided and after warmup.
    """
    total_travel = 0.0
    for u, v, _t_mean in get_path_edges(G, src, dst):
        t_actual = _stochastic_travel_time_for(travel_lookup, u, v, MANEUVER_SPEED_KMH)
        # Acquire switch resources if any for this edge
        for sw in edge_switches.get((u, v), []):
            r = switch_resources[sw].request()
            yield r
            yield env.timeout(SWITCH_CROSSING_TIME)
            switch_resources[sw].release(r)
        # Acquire edge capacity
        with edge_resources[(u, v)].request() as er:
            yield er
            yield env.timeout(t_actual)
            total_travel += t_actual
            if stats is not None and env.now >= WARMUP_HOURS:
                stats.station_arrivals[v] = stats.station_arrivals.get(v, 0) + 1
    if stats is not None:
        stats.travel_times.append(total_travel)


def travel_path_locked(env, G, src, dst, travel_lookup, stats=None):

    total_travel = 0.0
    for u, v, _t_mean in get_path_edges(G, src, dst):
        t_actual = _stochastic_travel_time_for(travel_lookup, u, v, MANEUVER_SPEED_KMH)
        yield env.timeout(t_actual)
        total_travel += t_actual
        if stats is not None and env.now >= WARMUP_HOURS:
            stats.station_arrivals[v] = stats.station_arrivals.get(v, 0) + 1
    if stats is not None:
        stats.travel_times.append(total_travel)


def train_process(env, cfg, train_id, destination, graph, edge_resources, bs_tracks, bs_locos,
                yard_track, yard_loco, load_mean, load_std, conflict, stats,
                switch_resources, edge_switches, yard_resources,
                support_station_res, dest_loco_res, yard_loco_res,
                corridor1_res=None, corridor2_res=None, dedicated_port_res=None):

    travel_lookup = SCENARIO_TRAVEL_LOOKUP[cfg.short]

    def tp(src, dst):
        return travel_path(env, graph, edge_resources, switch_resources, edge_switches,
                            src, dst, travel_lookup, stats)

    def tp_locked(src, dst):

        return travel_path_locked(env, graph, src, dst, travel_lookup, stats)

    arrival_time = env.now

    # =========================================================================
    # BASELINE – Corridor 1 / Corridor 2 + Yard-Port capacity rebuild
    # =========================================================================
    if cfg.name == "Baseline_v6_1_fixed":
        # --- Barco / Steel Hormozgan (north industries) – UNCHANGED ---
        if destination in ("Barko", "Steel_Hormozgan"):
            c1_req = corridor1_res.request()
            yield c1_req
            with bs_tracks.request() as bt:
                yield bt
                yield env.timeout(cfg.maneuver_time)
                with bs_locos.request() as lr:
                    yield lr
                    yield from tp("BS", "SW1")
                    corridor1_res.release(c1_req)
                    yield from tp("SW1", destination)

                    service = max(0.1, random.gauss(load_mean, load_std))
                    yield env.timeout(service + random.uniform(0, conflict))

                    yield from tp(destination, "BS")
                yield env.timeout(cfg.maneuver_time)
            stats.moves += 1

        # --- Yard-Port only (port terminal) ---
        elif destination == "Yard_Port":

            c1_req = corridor1_res.request()
            yield c1_req
            with bs_tracks.request() as bt:
                yield bt
                yt_req = yard_track.request()
                yield yt_req                       # wait at BS if Yard-Port is full
                yield from tp_locked("BS", "Yard_Port")
                corridor1_res.release(c1_req)       # train has fully left Corridor 1

                # Loading/unloading at Yard-Port - still holding the yard slot
                service = max(0.1, random.gauss(load_mean, load_std))
                yield env.timeout(service + random.uniform(0, conflict))
            # bs_track released (train left the BS platform)

            # Enter Corridor 2 only once it is free; the train stays at
            # Yard-Port (keeping its slot) while it waits here for Corridor 2.
            c2_req = corridor2_res.request()
            yield c2_req
            with bs_tracks.request() as bl:
                yield bl

                yard_track.release(yt_req)
                yield from tp_locked("Yard_Port", "BS")
            corridor2_res.release(c2_req)
            stats.moves += 1

        # --- Sulfer / old_steel_and_copper (south industries) ---
        else:
            # ---- 1. Enter Corridor 1: BS -> SW1 -> Yard-Port ----
            c1_req = corridor1_res.request()
            yield c1_req
            with bs_tracks.request() as bt:
                yield bt
                yt_req = yard_track.request()
                yield yt_req                       # wait at BS if Yard-Port is full
                yield from tp_locked("BS", "Yard_Port")
                corridor1_res.release(c1_req)       # train has fully left Corridor 1

                # Shunting-locomotive exchange at Yard-Port - still holding the slot
                yield env.timeout(cfg.maneuver_time)
            # bs_track released here

            # ---- 2. Shunting locomotive takes the wagons to the destination.
            # This is an internal yard move, not a corridor - the Yard-Port
            # slot is only vacated the instant the train actually departs. ----
            yard_track.release(yt_req)
            yield from tp_locked("Yard_Port", destination)

            # ---- 3. Unloading at the destination ----
            service = max(0.1, random.gauss(load_mean, load_std))
            yield env.timeout(service + random.uniform(0, conflict))

            # ---- 4. Before returning, verify/reserve a free Yard-Port slot.
            # If Yard-Port is full, the train waits here at the destination -
            # not on the way to, or just outside, Yard-Port. ----
            yt_req = yard_track.request()
            yield yt_req
            yield from tp_locked(destination, "Yard_Port")

            # ---- 5. Main locomotive re-attachment at Yard-Port - still
            # holding the slot ----
            yield env.timeout(cfg.maneuver_time)

            # ---- 6. Only now may the train enter Corridor 2 ----
            c2_req = corridor2_res.request()
            yield c2_req
            with bs_tracks.request() as bl:
                yield bl

                yard_track.release(yt_req)
                yield from tp_locked("Yard_Port", "BS")
            corridor2_res.release(c2_req)

            stats.moves += 1

    # =========================================================================
    # SCENARIO B – Same dynamic rules, with dedicated port line (deadlock fixed)
    # =========================================================================
    elif cfg.name == "Scenario_B_v6_1_fixed":
        # --- Barco / Steel Hormozgan ---
        if destination in ("Barko", "Steel_Hormozgan"):
            c1_req = corridor1_res.request()
            yield c1_req
            with bs_tracks.request() as bt:
                yield bt
                yield env.timeout(cfg.maneuver_time)
                with bs_locos.request() as lr:
                    yield lr
                    yield from tp("BS", "SW1")
                    corridor1_res.release(c1_req)
                    yield from tp("SW1", destination)

                    service = max(0.1, random.gauss(load_mean, load_std))
                    yield env.timeout(service + random.uniform(0, conflict))

                    yield from tp(destination, "BS")
                yield env.timeout(cfg.maneuver_time)
            stats.moves += 1

        # --- Yard-Port only – Dedicated Port Line, DEADLOCK FIXED ---
        elif destination == "Yard_Port":
            dp_req = dedicated_port_res.request()
            yield dp_req
            with bs_tracks.request() as bt:
                yield bt
                with yard_track.request() as yt:
                    yield yt
                    yield from tp("BS", "Yard_Port")
                    dedicated_port_res.release(dp_req)

                    service = max(0.1, random.gauss(load_mean, load_std))
                    yield env.timeout(service + random.uniform(0, conflict))

                # Yard track and bs_track released
            # bs_track free – safe to request return corridor
            c3_req = corridor2_res.request()
            yield c3_req
            with bs_tracks.request() as bl:
                yield bl
                yield from tp("Yard_Port", "BS")
            corridor2_res.release(c3_req)
            stats.moves += 1

        # --- Sulfer / old_steel_and_copper – CAPACITY FIX APPLIED ---
        else:
            # 1. Outbound: BS → SW1 → Yard-Port (Regime 1)
            c1_req = corridor1_res.request()
            yield c1_req
            with bs_tracks.request() as bt:
                yield bt
                # Request yard_track before leaving BS
                with yard_track.request() as yt:
                    yield yt
                    yield from tp("BS", "Yard_Port")
                    corridor1_res.release(c1_req)

                    yield env.timeout(cfg.maneuver_time)
                    service = max(0.1, random.gauss(load_mean, load_std))
                    yield env.timeout(service + random.uniform(0, conflict))

            # 2. Outbound leg: Yard-Port → destination (Regime 3)
            c3_req = corridor2_res.request()
            yield c3_req
            yield from tp("Yard_Port", destination)
            corridor2_res.release(c3_req)

            # 3. Service at destination
            service = max(0.1, random.gauss(load_mean, load_std))
            yield env.timeout(service + random.uniform(0, conflict))

            # 4. Return leg 1: destination → Yard-Port (Regime 3)
            # FIX: request yard_track before departure from destination
            c3_req = corridor2_res.request()
            yield c3_req
            with yard_track.request() as yt:
                yield yt
                yield from tp(destination, "Yard_Port")
            corridor2_res.release(c3_req)

            # 5. Loco re-attachment at Yard-Port
            with yard_track.request() as yt:
                yield yt
                yield env.timeout(cfg.maneuver_time)

            # 6. Return leg 2: Yard-Port → BS (Regime 3)
            c3_req = corridor2_res.request()
            yield c3_req
            with bs_tracks.request() as bl:
                yield bl
                yield from tp("Yard_Port", "BS")
            corridor2_res.release(c3_req)

            stats.moves += 1

    elif cfg.name == "Scenario_A_v6_1_fixed":
        with bs_tracks.request() as bt:
            yield bt
            with support_station_res.request() as ss:
                yield ss
                yield env.timeout(cfg.maneuver_time)
                with bs_locos.request() as lr:
                    yield lr
                    yield from tp("BS", "Support_Stations")
                    yield from tp("Support_Stations", destination)
                    service = max(0.05, random.gauss(load_mean, load_std))
                    yield env.timeout(service + random.uniform(0, conflict))
                    yield from tp(destination, "Support_Stations")
                yield from tp("Support_Stations", "BS")
                yield env.timeout(cfg.maneuver_time)
        stats.moves += 1

    else:  # Scenario C
        if destination in ("Barko", "Steel_Hormozgan", "Sulfer", "old_steel_and_copper"):
            yard_name = cfg.yard_requirements[destination]
            with bs_tracks.request() as bt:
                yield bt
                yield env.timeout(cfg.maneuver_time)
            yield from tp("BS", yard_name)
            with yard_resources[yard_name].request() as yt:
                yield yt
                with yard_loco_res[yard_name].request() as yl:
                    yield yl
                    yield from tp(yard_name, destination)
                    service = max(0.05, random.gauss(load_mean, load_std))
                    yield env.timeout(service + random.uniform(0, conflict))
                    yield from tp(destination, yard_name)
            yield from tp(yard_name, "BS")
            stats.moves += 1
            with bs_tracks.request() as st:
                yield st
                yield env.timeout(cfg.maneuver_time)
        else:
            with bs_tracks.request() as bt:
                yield bt
            yield from tp("BS", "Yard_Port")
            with yard_track.request() as yt:
                yield yt
                with yard_loco.request() as yl:
                    yield yl
                    service = max(0.1, random.gauss(load_mean, load_std))
                    yield env.timeout(service + random.uniform(0, conflict))
                    yield env.timeout(cfg.maneuver_time)
            stats.moves += 1
            yield from tp("Yard_Port", "BS")
            with bs_tracks.request() as st:
                yield st
                yield env.timeout(cfg.maneuver_time)

    dwell = env.now - arrival_time
    stats.completed_overall += 1
    if arrival_time >= WARMUP_HOURS:
        stats.dwells.append(dwell)
        if dwell > DELAY_THRESH:
            stats.delays.append(dwell - DELAY_THRESH)
        stats.completed += 1
        stats.dest_counts[destination] = stats.dest_counts.get(destination, 0) + 1
        stats.dwells_by_dest.setdefault(destination, []).append(dwell)


def train_generator(env, cfg, rate, destinations, graph, edge_resources, bs_tracks, locos,
                    load_mean, load_std, conflict, stats, prefix,
                    switch_resources, edge_switches, yard_resources, yard_track, yard_loco,
                    support_station_res, dest_loco_res, yard_loco_res,
                    corridor1_res=None, corridor2_res=None, dedicated_port_res=None):
    """Generate trains as a Poisson process."""
    tid = 0
    while True:
        yield env.timeout(random.expovariate(rate))
        tid += 1
        stats.total_generated += 1
        env.process(train_process(env, cfg, f"{prefix}_{tid}", random.choice(destinations),
            graph, edge_resources, bs_tracks, locos, yard_track, yard_loco,
            load_mean, load_std, conflict, stats, switch_resources, edge_switches, yard_resources,
            support_station_res, dest_loco_res, yard_loco_res, corridor1_res, corridor2_res, dedicated_port_res))


def paired_train_generator(env, cfg, rate, within_gap, destinations, graph, edge_resources, bs_tracks, locos,
                            load_mean, load_std, conflict, stats, prefix,
                            switch_resources, edge_switches, yard_resources, yard_track, yard_loco,
                            support_station_res, dest_loco_res, yard_loco_res,
                            corridor1_res=None, corridor2_res=None, dedicated_port_res=None):
    """Generate pairs of port trains with a short gap between them."""
    cycle_time = 2.0 / rate
    between_gap_mean = max(0.5, cycle_time - within_gap)
    between_gap_std  = between_gap_mean * 0.232
    tid = 0
    while True:
        for i in range(2):
            tid += 1
            stats.total_generated += 1
            env.process(train_process(env, cfg, f"{prefix}_{tid}", random.choice(destinations),
                graph, edge_resources, bs_tracks, locos, yard_track, yard_loco,
                load_mean, load_std, conflict, stats, switch_resources, edge_switches, yard_resources,
                support_station_res, dest_loco_res, yard_loco_res, corridor1_res, corridor2_res, dedicated_port_res))
            if i == 0:
                yield env.timeout(max(0.05, random.gauss(within_gap, within_gap * 0.1)))
        yield env.timeout(max(0.5, random.gauss(between_gap_mean, between_gap_std)))


def run_replication(cfg, seed):
    """Run one replication of a scenario."""
    random.seed(seed); np.random.seed(seed)
    env = simpy.Environment()
    stats = RunStats()
    graph = cfg.graph_builder()
    sw_res, edge_sw, yard_res = cfg.switch_builder(env, cfg)
    edge_res = {(u, v): simpy.Resource(env, cfg.edge_capacities.get((u, v), 1)) for u, v in graph.edges}
    bs_tracks  = MonitoredResource(env, cfg.bs_track_cap)
    bs_locos   = MonitoredResource(env, cfg.bs_loco_cap)
    yard_track = MonitoredResource(env, cfg.yard_track_cap)
    yard_loco  = MonitoredResource(env, cfg.yard_loco_cap)
    yard_loco_res = {yn: MonitoredResource(env, cap) for yn, cap in cfg.yard_loco_capacities.items()}
    support_station_res = MonitoredResource(env, cfg.support_station_cap)
    dest_loco_res = {d: MonitoredResource(env, cfg.dest_loco_cap) for d in DEST_LIST}

    is_corridor_scenario = cfg.name in ("Baseline_v6_1_fixed", "Scenario_B_v6_1_fixed")
    corridor1_res = MonitoredResource(env, 1) if is_corridor_scenario else None
    corridor2_res = MonitoredResource(env, 1) if is_corridor_scenario else None
    dedicated_port_res = MonitoredResource(env, 1) if cfg.name == "Scenario_B_v6_1_fixed" else None

    common = dict(graph=graph, edge_resources=edge_res, bs_tracks=bs_tracks, locos=bs_locos,
                yard_track=yard_track, yard_loco=yard_loco, stats=stats,
                switch_resources=sw_res, edge_switches=edge_sw, yard_resources=yard_res,
                support_station_res=support_station_res, dest_loco_res=dest_loco_res,
                yard_loco_res=yard_loco_res, corridor1_res=corridor1_res, corridor2_res=corridor2_res,
                dedicated_port_res=dedicated_port_res)

    if cfg.use_paired_arrivals:
        env.process(paired_train_generator(env, cfg, rate=cfg.port_train_rate, within_gap=cfg.within_gap_hours,
            destinations=["Yard_Port"], load_mean=cfg.port_load_time[0], load_std=cfg.port_load_time[1],
            conflict=cfg.port_conflict, prefix="PORT", **common))
    else:
        env.process(train_generator(env, cfg, cfg.port_train_rate, ["Yard_Port"],
            load_mean=cfg.port_load_time[0], load_std=cfg.port_load_time[1],
            conflict=cfg.port_conflict, prefix="PORT", **common))

    env.process(train_generator(env, cfg, cfg.north_ind_rate, ["Barko", "Steel_Hormozgan"],
        load_mean=cfg.north_load_time[0], load_std=cfg.north_load_time[1],
        conflict=cfg.north_conflict, prefix="NORTH", **common))

    env.process(train_generator(env, cfg, cfg.south_ind_rate, ["Sulfer", "old_steel_and_copper"],
        load_mean=cfg.south_load_time[0], load_std=cfg.south_load_time[1],
        conflict=cfg.south_conflict, prefix="SOUTH", **common))

    env.run(until=SIM_HOURS)

    T = SIM_HOURS
    stats.bs_track_util   = bs_tracks.busy_time   / (cfg.bs_track_cap   * T) * 100
    stats.yard_track_util = yard_track.busy_time  / (cfg.yard_track_cap * T) * 100
    stats.bs_loco_util    = bs_locos.busy_time    / (cfg.bs_loco_cap    * T) * 100

    queue_int = bs_tracks.queue_integral + yard_track.queue_integral
    for res in yard_res.values():
        if isinstance(res, MonitoredResource):
            queue_int += res.queue_integral
    if isinstance(sw_res, dict):
        for sw in sw_res.values():
            if isinstance(sw, MonitoredResource):
                queue_int += sw.queue_integral
    stats.avg_queue_len = queue_int / T

    total_loco_busy = bs_locos.busy_time + yard_loco.busy_time
    total_loco_cap  = cfg.bs_loco_cap + cfg.yard_loco_cap
    for yl in yard_loco_res.values():
        total_loco_busy += yl.busy_time
        total_loco_cap  += yl.capacity
    stats.all_loco_util = (total_loco_busy / (total_loco_cap * T) * 100) if total_loco_cap > 0 else 0

    resource_objs = {
        "BS_track": bs_tracks,
        "BS_loco": bs_locos,
        "Yard_track": yard_track,
        "Yard_loco": yard_loco,
        "Support_Station": support_station_res,
    }
    if corridor1_res is not None:
        resource_objs["Regime1_BS_SW1_YardPort"] = corridor1_res
    if corridor2_res is not None:
        resource_objs["Regime2or3_YardPort_SW2_SW3_BS"] = corridor2_res
    if dedicated_port_res is not None:
        resource_objs["Regime2_DedicatedPortLine_BS_YardPort"] = dedicated_port_res
    if isinstance(sw_res, dict):
        for name, obj in sw_res.items():
            if isinstance(obj, MonitoredResource):
                resource_objs[f"Switch_{name}"] = obj
    if isinstance(yard_res, dict):
        for name, obj in yard_res.items():
            if isinstance(obj, MonitoredResource):
                resource_objs[f"Yard_{name}"] = obj
    for name, obj in yard_loco_res.items():
        resource_objs[f"YardLoco_{name}"] = obj

    for name, obj in resource_objs.items():
        stats.resource_util[name] = (obj.busy_time / (obj.capacity * T) * 100) if obj.capacity > 0 else 0.0
        stats.resource_wait[name] = (obj.queue_integral / obj.total_requests * 60) if obj.total_requests > 0 else 0.0

    return stats


def run_scenario(cfg, n=REPLICATIONS):
    """Run multiple replications and aggregate results."""
    rows, all_dwells, dest_counts_total, station_arrivals_total = [], [], {}, {}
    total_generated_sum = 0
    completed_overall_sum = 0
    dwells_by_dest_total = {}
    resource_util_sums = {}
    resource_wait_sums = {}
    eff_days = (SIM_HOURS - WARMUP_HOURS) / 24
    for i in range(n):
        s = run_replication(cfg, seed=RANDOM_SEED + i * 137)
        all_dwells.extend(s.dwells)
        for k, v in s.dest_counts.items():
            dest_counts_total[k] = dest_counts_total.get(k, 0) + v
        for k, v in s.station_arrivals.items():
            station_arrivals_total[k] = station_arrivals_total.get(k, 0) + v
        total_generated_sum += s.total_generated
        completed_overall_sum += s.completed_overall
        for k, v in s.dwells_by_dest.items():
            dwells_by_dest_total.setdefault(k, []).extend(v)
        for k, v in s.resource_util.items():
            resource_util_sums.setdefault(k, []).append(v)
        for k, v in s.resource_wait.items():
            resource_wait_sums.setdefault(k, []).append(v)
        rows.append({
            "avg_dwell":     np.mean(s.dwells) if s.dwells else 0,
            "p95_dwell":     np.percentile(s.dwells, 95) if s.dwells else 0,
            "thru_day":      s.completed / eff_days,
            "queue_len":     s.avg_queue_len,
            "all_loco_util": s.all_loco_util,
            "track_util":    s.bs_track_util,
            "yard_util":     s.yard_track_util,
        })
    dest_share = {k: v / sum(dest_counts_total.values()) for k, v in dest_counts_total.items()} if dest_counts_total else {}
    resource_util_avg = {k: float(np.mean(v)) for k, v in resource_util_sums.items()}
    resource_wait_avg = {k: float(np.mean(v)) for k, v in resource_wait_sums.items()}
    extras = dict(
        total_generated=total_generated_sum,
        completed_overall=completed_overall_sum,
        incomplete=total_generated_sum - completed_overall_sum,
        dwells_by_dest=dwells_by_dest_total,
        resource_util=resource_util_avg,
        resource_wait=resource_wait_avg,
    )
    return pd.DataFrame(rows), all_dwells, dest_share, dest_counts_total, station_arrivals_total, extras


# =============================================================================
# Capacity and conflict analysis
# =============================================================================
def compute_destination_capacities(cfg):
    """Compute theoretical capacity per destination based on service times and conflict."""
    capacities = {}
    mapping = {
        "Yard_Port": (cfg.port_load_time, cfg.port_conflict),
        "Barko": (cfg.north_load_time, cfg.north_conflict),
        "Steel_Hormozgan": (cfg.north_load_time, cfg.north_conflict),
        "old_steel_and_copper": (cfg.south_load_time, cfg.south_conflict),
        "Sulfer": (cfg.south_load_time, cfg.south_conflict),
    }
    for dest, ((mean, std), conflict) in mapping.items():
        effective_service = mean + conflict / 2
        trains_per_day = 24 / effective_service if effective_service > 0 else float('inf')
        wagons_per_day = trains_per_day * AVG_WAGONS.get(dest, 0)
        capacities[dest] = {
            "service_time": mean,
            "conflict": conflict,
            "effective_hours": effective_service,
            "trains_per_day": round(trains_per_day, 1),
            "wagons_per_day": round(wagons_per_day, 1)
        }
    return capacities


def get_roundtrip_resources(cfg):
    """For each destination, collect the set of resources (edges, switches, yards, tracks) used in a round trip."""
    dummy_env = simpy.Environment()
    graph = cfg.graph_builder()
    sw_res, edge_sw, yard_res = cfg.switch_builder(dummy_env, cfg)

    resources_per_dest = {}
    for dest in DEST_LIST:
        res_set = set()

        path_edges = get_path_edges(graph, "BS", dest)
        for u, v, _ in path_edges:
            res_set.add(("edge", (u, v)))
            for sw in edge_sw.get((u, v), []):
                res_set.add(("switch", sw))

        path_edges_back = get_path_edges(graph, dest, "BS")
        for u, v, _ in path_edges_back:
            res_set.add(("edge", (u, v)))
            for sw in edge_sw.get((u, v), []):
                res_set.add(("switch", sw))

        if cfg.yard_requirements and dest in cfg.yard_requirements:
            yard = cfg.yard_requirements[dest]
            res_set.add(("yard", yard))
        if cfg.name in ("Baseline_v6_1_fixed", "Scenario_B_v6_1_fixed"):
            if dest not in ("Barko", "Steel_Hormozgan"):
                res_set.add(("yard", "Yard_Port_yard"))
        elif cfg.name == "Scenario_C_v6_1_fixed":
            if dest == "Yard_Port":
                res_set.add(("yard", "Yard_Port_yard"))

        res_set.add(("bs_track", "BS_track"))
        resources_per_dest[dest] = res_set
    return resources_per_dest


def build_conflict_matrix(resources_per_dest):
    """Build a conflict matrix: True if two destinations share any resource."""
    dests = list(resources_per_dest.keys())
    conflict = {}
    for d1 in dests:
        conflict[d1] = {}
        for d2 in dests:
            if d1 == d2:
                conflict[d1][d2] = False
            else:
                common = resources_per_dest[d1] & resources_per_dest[d2]
                conflict[d1][d2] = len(common) > 0
    df_conflict = pd.DataFrame(conflict, index=dests)
    return df_conflict


def max_concurrent_routes(resources_per_dest):
    """Find the maximum number of destinations that can be served concurrently without resource conflict."""
    dests = list(resources_per_dest.keys())
    best = 0
    best_set = []
    for r in range(1, len(dests) + 1):
        for combo in itertools.combinations(dests, r):
            ok = True
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    if resources_per_dest[combo[i]] & resources_per_dest[combo[j]]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                if r > best:
                    best = r
                    best_set = combo
    return best, best_set


# =============================================================================
# Main execution
# =============================================================================
def main():
    print("Starting simulation of all scenarios...")
    eff_days = (SIM_HOURS - WARMUP_HOURS) / 24

    scenario_results = {}
    for cfg in ALL_SCENARIOS:
        total_rate = cfg.port_train_rate + cfg.north_ind_rate + cfg.south_ind_rate
        inbound_per_day = total_rate * 24
        print(f"\n  {cfg.short}: total round-trip rate = {total_rate:.4f}/h, "
            f"inbound trains/day ~= {inbound_per_day:.1f}")

        print(f"  Running {cfg.short} ...")
        df, dwells, dest_share, dest_counts, station_arrivals, extras = run_scenario(cfg)

        avg_wagons_per_train = sum(dest_share.get(d, 0) * AVG_WAGONS.get(d, 0) for d in DEST_LIST)
        thru_day = df["thru_day"].mean()
        wagons_day = thru_day * avg_wagons_per_train

        scenario_results[cfg.short] = {
            "config": cfg,
            "avg_dwell": df["avg_dwell"].mean(),
            "p95_dwell": df["p95_dwell"].mean(),
            "thru_day": thru_day,
            "wagons_day": wagons_day,
            "all_loco_util": df["all_loco_util"].mean(),
            "queue_len": df["queue_len"].mean(),
            "dest_share": dest_share,
            "dest_counts": dest_counts,
            "station_arrivals": station_arrivals,
            **extras,
        }

    print("\n=== 1. Destination loading/unloading capacities (Baseline) ===")
    dest_cap = compute_destination_capacities(BASELINE)
    cap_df = pd.DataFrame(dest_cap).T
    print(cap_df[["service_time", "conflict", "effective_hours", "trains_per_day", "wagons_per_day"]].to_string())

    print("\n=== 2. Conflict matrix and max concurrent routes ===")
    for name, res in scenario_results.items():
        cfg = res["config"]
        resources = get_roundtrip_resources(cfg)
        conflict_mat = build_conflict_matrix(resources)
        max_routes, routes = max_concurrent_routes(resources)
        print(f"\n{name}:")
        print("Conflict matrix (True = conflict):")
        print(conflict_mat.to_string())
        print(f"Maximum concurrent routes: {max_routes}  (set: {routes})")

    print(f"\n=== 3. Total trains that reached each station ({REPLICATIONS} reps, "
        f"post-warmup {eff_days:.0f}-day window) ===")
    all_stations = sorted({s for res in scenario_results.values() for s in res["station_arrivals"]})
    station_table = pd.DataFrame(
        {name: {s: res["station_arrivals"].get(s, 0) for s in all_stations}
        for name, res in scenario_results.items()}
    ).T
    station_table["TOTAL"] = station_table.sum(axis=1)
    print(station_table.to_string())
    print("\n(per day, averaged over the measurement window)")
    print((station_table.drop(columns="TOTAL") / eff_days).round(2).to_string())

    print("\n=== 4. Final comparison: Current state (Baseline) vs Ideal (Scenario C) ===")
    base = scenario_results["Baseline"]
    ideal = scenario_results["Scenario C"]

    base_max_routes, _ = max_concurrent_routes(get_roundtrip_resources(BASELINE))
    ideal_max_routes, _ = max_concurrent_routes(get_roundtrip_resources(SCENARIO_C))

    final_table = pd.DataFrame({
        "Metric": [
            "Average dwell time (h)",
            "95th percentile dwell time (h)",
            "Throughput (trains/day)",
            "Throughput (wagons/day)",
            "Max concurrent routes",
            "Main bottleneck"
        ],
        "Baseline": [
            f"{base['avg_dwell']:.2f}",
            f"{base['p95_dwell']:.2f}",
            f"{base['thru_day']:.1f}",
            f"{base['wagons_day']:.0f}",
            str(base_max_routes),
            "SW1 (main switch)"
        ],
        "Scenario C": [
            f"{ideal['avg_dwell']:.2f}",
            f"{ideal['p95_dwell']:.2f}",
            f"{ideal['thru_day']:.1f}",
            f"{ideal['wagons_day']:.0f}",
            str(ideal_max_routes),
            "Yards and independent lines"
        ]
    })
    print(final_table.to_string(index=False))

    return scenario_results


if __name__ == "__main__":
    main()