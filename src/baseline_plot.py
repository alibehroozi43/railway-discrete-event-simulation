import simpy
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.lines import Line2D
from dataclasses import dataclass
from enum import IntEnum
from collections import defaultdict
from typing import List, Optional

from revised_code_time_table import (
    RANDOM_SEED,
    SIM_HOURS,
    FIXED_TRAVEL_TIMES_H,
    BASELINE,
    BASE_PORT_RATE,
    BASE_NORTH_RATE,
    BASE_SOUTH_RATE,
)

# The mother file activates the non-interactive Agg backend.
# Switch back to an interactive backend for displaying the plot.
plt.switch_backend("TkAgg")

BS_TO_SW1 = FIXED_TRAVEL_TIMES_H[("BS", "SW1")]
SW1_TO_BS = FIXED_TRAVEL_TIMES_H[("SW1", "BS")]
SW1_TO_YARD = FIXED_TRAVEL_TIMES_H[("SW1", "Yard_Port")]
YARD_TO_SW2 = FIXED_TRAVEL_TIMES_H[("Yard_Port", "SW2")]
SW2_TO_YARD = FIXED_TRAVEL_TIMES_H[("SW2", "Yard_Port")]
SW2_TO_SW3 = FIXED_TRAVEL_TIMES_H[("SW2", "SW3")]
SW3_TO_SW2 = FIXED_TRAVEL_TIMES_H[("SW3", "SW2")]
SW3_TO_BS_EXIT = FIXED_TRAVEL_TIMES_H[("SW3", "BS")]

@dataclass(frozen=True)
class BaselinePlotInputs:
    # Corridor 1
    bs_to_sw1: float
    sw1_to_bs: float
    sw1_to_yard: float

    # Corridor 2
    yard_to_sw2: float
    sw2_to_yard: float
    sw2_to_sw3: float
    sw3_to_sw2: float
    sw3_to_bs_exit: float

    # Off-corridor branch travel
    sw1_to_barco: float
    barco_to_sw1: float
    sw1_to_steel: float
    steel_to_sw1: float

    sw2_to_copper: float
    copper_to_sw2: float

    sw3_to_sulfur: float
    sulfur_to_sw3: float

    # Operational parameters
    maneuver_time: float

    port_load_mean: float
    port_load_std: float

    north_load_mean: float
    north_load_std: float

    south_load_mean: float
    south_load_std: float

    port_conflict: float
    north_conflict: float
    south_conflict: float

    # Arrival rates
    port_rate: float
    north_rate: float
    south_rate: float

    # Capacities
    bs_capacity: int
    yard_port_capacity: int

def load_baseline_plot_inputs() -> BaselinePlotInputs:
    """
    Read Baseline parameters from the mother simulation.

    All time values are measured in hours.
    """

    inputs = BaselinePlotInputs(
        # Corridor 1
        bs_to_sw1=FIXED_TRAVEL_TIMES_H[
            ("BS", "SW1")
        ],
        sw1_to_bs=FIXED_TRAVEL_TIMES_H[
            ("SW1", "BS")
        ],
        sw1_to_yard=FIXED_TRAVEL_TIMES_H[
            ("SW1", "Yard_Port")
        ],

        # Corridor 2
        yard_to_sw2=FIXED_TRAVEL_TIMES_H[
            ("Yard_Port", "SW2")
        ],
        sw2_to_yard=FIXED_TRAVEL_TIMES_H[
            ("SW2", "Yard_Port")
        ],
        sw2_to_sw3=FIXED_TRAVEL_TIMES_H[
            ("SW2", "SW3")
        ],
        sw3_to_sw2=FIXED_TRAVEL_TIMES_H[
            ("SW3", "SW2")
        ],
        sw3_to_bs_exit=FIXED_TRAVEL_TIMES_H[
            ("SW3", "BS")
        ],

        # Barco branch
        sw1_to_barco=FIXED_TRAVEL_TIMES_H[
            ("SW1", "Barko")
        ],
        barco_to_sw1=FIXED_TRAVEL_TIMES_H[
            ("Barko", "SW1")
        ],

        # Steel Hormozgan branch
        sw1_to_steel=FIXED_TRAVEL_TIMES_H[
            ("SW1", "Steel_Hormozgan")
        ],
        steel_to_sw1=FIXED_TRAVEL_TIMES_H[
            ("Steel_Hormozgan", "SW1")
        ],

        # Copper branch
        sw2_to_copper=FIXED_TRAVEL_TIMES_H[
            ("SW2", "old_steel_and_copper")
        ],
        copper_to_sw2=FIXED_TRAVEL_TIMES_H[
            ("old_steel_and_copper", "SW2")
        ],

        # Sulfur branch
        sw3_to_sulfur=FIXED_TRAVEL_TIMES_H[
            ("SW3", "Sulfer")
        ],
        sulfur_to_sw3=FIXED_TRAVEL_TIMES_H[
            ("Sulfer", "SW3")
        ],

        # Maneuver time
        maneuver_time=BASELINE.maneuver_time,

        # Loading/unloading distributions
        port_load_mean=BASELINE.port_load_time[0],
        port_load_std=BASELINE.port_load_time[1],

        north_load_mean=BASELINE.north_load_time[0],
        north_load_std=BASELINE.north_load_time[1],

        south_load_mean=BASELINE.south_load_time[0],
        south_load_std=BASELINE.south_load_time[1],

        # Conflict delays
        port_conflict=BASELINE.port_conflict,
        north_conflict=BASELINE.north_conflict,
        south_conflict=BASELINE.south_conflict,

        # Train arrival rates
        port_rate=BASE_PORT_RATE,
        north_rate=BASE_NORTH_RATE,
        south_rate=BASE_SOUTH_RATE,

        # Station capacities
        bs_capacity=BASELINE.bs_track_cap,
        yard_port_capacity=BASELINE.yard_track_cap,
    )

    if not isinstance(
        inputs,
        BaselinePlotInputs,
    ):
        raise TypeError(
            "load_baseline_plot_inputs() must return "
            "a BaselinePlotInputs instance."
        )

    return inputs


def expected_uniform_delay(maximum_delay: float) -> float:
    """
    Return the expected value of Uniform(0, maximum_delay).

    Used only for deterministic integration testing.
    """
    if maximum_delay < 0:
        raise ValueError(
            "Maximum conflict delay cannot be negative."
        )

    return maximum_delay / 2.0


def build_deterministic_cycle_times(
    inputs: BaselinePlotInputs,
) -> dict:
    """
    Build deterministic expected cycle times for each destination.

    All values are in hours.

    Off-corridor cycles include:
    branch travel out
    + loading/unloading
    + expected conflict delay
    + branch travel back
    """

    cycles = {
        "barco_cycle": (
            inputs.sw1_to_barco
            + inputs.north_load_mean
            + expected_uniform_delay(
                inputs.north_conflict
            )
            + inputs.barco_to_sw1
        ),

        "steel_cycle": (
            inputs.sw1_to_steel
            + inputs.north_load_mean
            + expected_uniform_delay(
                inputs.north_conflict
            )
            + inputs.steel_to_sw1
        ),

        "yard_loading": (
            inputs.port_load_mean
            + expected_uniform_delay(
                inputs.port_conflict
            )
        ),

        "copper_cycle": (
            inputs.sw2_to_copper
            + inputs.south_load_mean
            + expected_uniform_delay(
                inputs.south_conflict
            )
            + inputs.copper_to_sw2
        ),

        "sulfur_cycle": (
            inputs.sw3_to_sulfur
            + inputs.south_load_mean
            + expected_uniform_delay(
                inputs.south_conflict
            )
            + inputs.sulfur_to_sw3
        ),
    }

    for cycle_name, cycle_time in cycles.items():
        if cycle_time <= 0:
            raise ValueError(
                f"{cycle_name} must be greater than zero."
            )

    return cycles

def sample_positive_normal(
    mean: float,
    std: float,
    minimum: float = 0.1,
) -> float:
    """
    Draw a positive service time from a normal distribution.
    """

    sampled_value = random.gauss(
        mean,
        std,
    )

    return max(
        minimum,
        sampled_value,
    )


def sample_conflict_delay(
    maximum_delay: float,
) -> float:
    """
    Draw a conflict delay from Uniform(0, maximum_delay).
    """

    if maximum_delay <= 0:
        return 0.0

    return random.uniform(
        0.0,
        maximum_delay,
    )


def sample_exponential_interarrival(
    rate: float,
) -> float:
    """
    Generate an exponential interarrival time.

    rate is measured in trains per hour.
    """

    if rate <= 0:
        raise ValueError(
            "Arrival rate must be greater than zero."
        )

    return random.expovariate(
        rate
    )

def create_barco_train(
    simulator: BaselineSimulator,
    train_id: int,
    inputs: BaselinePlotInputs,
) -> BarcoTrain:
    service_time = sample_positive_normal(
        mean=inputs.north_load_mean,
        std=inputs.north_load_std,
    )

    conflict_delay = sample_conflict_delay(
        inputs.north_conflict
    )

    barco_cycle_time = (
        inputs.sw1_to_barco
        + service_time
        + conflict_delay
        + inputs.barco_to_sw1
    )

    return BarcoTrain(
        simulator=simulator,
        train_id=train_id,
        travel_to_sw1=inputs.bs_to_sw1,
        barco_cycle_time=barco_cycle_time,
        travel_back=inputs.sw1_to_bs,
    )


def create_steel_train(
    simulator: BaselineSimulator,
    train_id: int,
    inputs: BaselinePlotInputs,
) -> SteelHormozganTrain:
    service_time = sample_positive_normal(
        mean=inputs.north_load_mean,
        std=inputs.north_load_std,
    )

    conflict_delay = sample_conflict_delay(
        inputs.north_conflict
    )

    steel_cycle_time = (
        inputs.sw1_to_steel
        + service_time
        + conflict_delay
        + inputs.steel_to_sw1
    )

    return SteelHormozganTrain(
        simulator=simulator,
        train_id=train_id,
        travel_to_sw1=inputs.bs_to_sw1,
        steel_cycle_time=steel_cycle_time,
        travel_back=inputs.sw1_to_bs,
    )


def create_yard_port_train(
    simulator: BaselineSimulator,
    train_id: int,
    inputs: BaselinePlotInputs,
) -> YardPortTrain:
    service_time = sample_positive_normal(
        mean=inputs.port_load_mean,
        std=inputs.port_load_std,
    )

    conflict_delay = sample_conflict_delay(
        inputs.port_conflict
    )

    loading_time = (
        service_time
        + conflict_delay
    )

    return YardPortTrain(
        simulator=simulator,
        train_id=train_id,
        bs_to_sw1_time=inputs.bs_to_sw1,
        sw1_to_yard_time=inputs.sw1_to_yard,
        loading_time=loading_time,
        yard_to_sw2_time=inputs.yard_to_sw2,
        sw2_to_sw3_time=inputs.sw2_to_sw3,
        sw3_to_bs_exit_time=inputs.sw3_to_bs_exit,
    )


def create_copper_train(
    simulator: BaselineSimulator,
    train_id: int,
    inputs: BaselinePlotInputs,
) -> CopperTrain:
    service_time = sample_positive_normal(
        mean=inputs.south_load_mean,
        std=inputs.south_load_std,
    )

    conflict_delay = sample_conflict_delay(
        inputs.south_conflict
    )

    copper_cycle_time = (
        inputs.sw2_to_copper
        + service_time
        + conflict_delay
        + inputs.copper_to_sw2
    )

    return CopperTrain(
        simulator=simulator,
        train_id=train_id,
        bs_to_sw1_time=inputs.bs_to_sw1,
        sw1_to_yard_time=inputs.sw1_to_yard,
        shunting_time=inputs.maneuver_time,
        yard_to_sw2_time=inputs.yard_to_sw2,
        copper_cycle_time=copper_cycle_time,
        sw2_to_yard_time=inputs.sw2_to_yard,
        main_loco_attach_time=inputs.maneuver_time,
        sw2_to_sw3_time=inputs.sw2_to_sw3,
        sw3_to_bs_exit_time=inputs.sw3_to_bs_exit,
    )


def create_sulfur_train(
    simulator: BaselineSimulator,
    train_id: int,
    inputs: BaselinePlotInputs,
) -> SulfurTrain:
    service_time = sample_positive_normal(
        mean=inputs.south_load_mean,
        std=inputs.south_load_std,
    )

    conflict_delay = sample_conflict_delay(
        inputs.south_conflict
    )

    sulfur_cycle_time = (
        inputs.sw3_to_sulfur
        + service_time
        + conflict_delay
        + inputs.sulfur_to_sw3
    )

    return SulfurTrain(
        simulator=simulator,
        train_id=train_id,
        bs_to_sw1_time=inputs.bs_to_sw1,
        sw1_to_yard_time=inputs.sw1_to_yard,
        shunting_time=inputs.maneuver_time,
        yard_to_sw2_time=inputs.yard_to_sw2,
        sw2_to_sw3_time=inputs.sw2_to_sw3,
        sulfur_cycle_time=sulfur_cycle_time,
        sw3_to_sw2_time=inputs.sw3_to_sw2,
        sw2_to_yard_time=inputs.sw2_to_yard,
        main_loco_attach_time=inputs.maneuver_time,
        sw3_to_bs_exit_time=inputs.sw3_to_bs_exit,
    )

def north_train_generator(
    simulator: BaselineSimulator,
    inputs: BaselinePlotInputs,
    train_ids: TrainIdGenerator,
    generation_horizon: float,
    generated_counts: dict,
):
    """
    Generate Barco and Steel Hormozgan trains.
    """

    while True:
        interarrival_time = (
            sample_exponential_interarrival(
                inputs.north_rate
            )
        )

        yield simulator.env.timeout(
            interarrival_time
        )

        if simulator.env.now > generation_horizon:
            break

        train_id = train_ids.next_id()

        destination = random.choice(
            [
                "Barco",
                "Steel",
            ]
        )

        if destination == "Barco":
            train = create_barco_train(
                simulator=simulator,
                train_id=train_id,
                inputs=inputs,
            )

            generated_counts["Barco"] += 1

        else:
            train = create_steel_train(
                simulator=simulator,
                train_id=train_id,
                inputs=inputs,
            )

            generated_counts["Steel"] += 1

        simulator.env.process(
            train.run()
        )

def south_train_generator(
    simulator: BaselineSimulator,
    inputs: BaselinePlotInputs,
    train_ids: TrainIdGenerator,
    generation_horizon: float,
    generated_counts: dict,
):
    """
    Generate Copper and Sulfur trains.
    """

    while True:
        interarrival_time = (
            sample_exponential_interarrival(
                inputs.south_rate
            )
        )

        yield simulator.env.timeout(
            interarrival_time
        )

        if simulator.env.now > generation_horizon:
            break

        train_id = train_ids.next_id()

        destination = random.choice(
            [
                "Copper",
                "Sulfur",
            ]
        )

        if destination == "Copper":
            train = create_copper_train(
                simulator=simulator,
                train_id=train_id,
                inputs=inputs,
            )

            generated_counts["Copper"] += 1

        else:
            train = create_sulfur_train(
                simulator=simulator,
                train_id=train_id,
                inputs=inputs,
            )

            generated_counts["Sulfur"] += 1

        simulator.env.process(
            train.run()
        )

def port_paired_train_generator(
    simulator: BaselineSimulator,
    inputs: BaselinePlotInputs,
    train_ids: TrainIdGenerator,
    generation_horizon: float,
    generated_counts: dict,
    within_pair_gap: float = 0.67,
):
    """
    Generate Yard Port trains in pairs.

    The logic follows the paired-arrival structure
    used in the mother simulation.
    """

    cycle_time = (
        2.0
        / inputs.port_rate
    )

    between_pair_gap_mean = max(
        0.5,
        cycle_time - within_pair_gap,
    )

    between_pair_gap_std = (
        between_pair_gap_mean
        * 0.232
    )

    while simulator.env.now <= generation_horizon:

        for pair_index in range(2):
            if simulator.env.now > generation_horizon:
                return

            train_id = train_ids.next_id()

            train = create_yard_port_train(
                simulator=simulator,
                train_id=train_id,
                inputs=inputs,
            )

            generated_counts["Yard Port"] += 1

            simulator.env.process(
                train.run()
            )

            if pair_index == 0:
                first_gap = max(
                    0.05,
                    random.gauss(
                        within_pair_gap,
                        within_pair_gap * 0.1,
                    ),
                )

                yield simulator.env.timeout(
                    first_gap
                )

        between_pair_gap = max(
            0.5,
            random.gauss(
                between_pair_gap_mean,
                between_pair_gap_std,
            ),
        )

        yield simulator.env.timeout(
            between_pair_gap
        )

def hours_to_minutes(value):
    return value * 60


def load_baseline_inputs():
    return {
        "travel": {
            "bs_to_sw1": hours_to_minutes(FIXED_TRAVEL_TIMES_H[("BS", "SW1")]),
            "sw1_to_bs": hours_to_minutes(FIXED_TRAVEL_TIMES_H[("SW1", "BS")]),
            "sw1_to_yard": hours_to_minutes(FIXED_TRAVEL_TIMES_H[("SW1", "Yard_Port")]),
            "yard_to_sw2": hours_to_minutes(FIXED_TRAVEL_TIMES_H[("Yard_Port", "SW2")]),
            "sw2_to_yard": hours_to_minutes(FIXED_TRAVEL_TIMES_H[("SW2", "Yard_Port")]),
            "sw2_to_sw3": hours_to_minutes(FIXED_TRAVEL_TIMES_H[("SW2", "SW3")]),
            "sw3_to_sw2": hours_to_minutes(FIXED_TRAVEL_TIMES_H[("SW3", "SW2")]),
            "sw3_to_bs_exit": hours_to_minutes(FIXED_TRAVEL_TIMES_H[("SW3", "BS")]),
        },
        "loads": {
            "yard_port_mean_h": BASELINE.port_load_time[0],
            "north_mean_h": BASELINE.north_load_time[0],
            "south_mean_h": BASELINE.south_load_time[0],
        },
        "rates": {
            "port_rate": BASE_PORT_RATE,
            "north_rate": BASE_NORTH_RATE,
            "south_rate": BASE_SOUTH_RATE,
        },
        "capacities": {
            "bs_track_cap": BASELINE.bs_track_cap,
            "yard_track_cap": BASELINE.yard_track_cap,
        },
        "maneuver_time_h": BASELINE.maneuver_time,
    }

class RailwayNode(IntEnum):
    BS_ENTRY = 0
    SW1 = 1
    YARD_PORT = 2
    SW2 = 3
    SW3 = 4
    BS_EXIT = 5


class Station(IntEnum):
    BS = 1
    YARD_PORT = 2


class EventType(IntEnum):
    ENTER = 1
    MOVE = 2
    WAIT = 3
    EXIT = 4


class SegmentType(IntEnum):
    MOVE = 1
    WAIT = 2


NODE_LABELS = {
    RailwayNode.BS_ENTRY: "BS Entry",
    RailwayNode.SW1: "SW1",
    RailwayNode.YARD_PORT: "Yard Port",
    RailwayNode.SW2: "SW2",
    RailwayNode.SW3: "SW3",
    RailwayNode.BS_EXIT: "BS Exit",
}


class Corridor(IntEnum):
    C1 = 1
    C2 = 2


CORRIDOR_PATHS = {
    Corridor.C1: [
        RailwayNode.BS_ENTRY,
        RailwayNode.SW1,
        RailwayNode.YARD_PORT,
    ],
    Corridor.C2: [
        RailwayNode.YARD_PORT,
        RailwayNode.SW2,
        RailwayNode.SW3,
        RailwayNode.BS_EXIT,
    ]
}


@dataclass
class Segment:
    train_id: int
    start_time: float
    end_time: float
    start_node: RailwayNode
    end_node: RailwayNode
    corridor: Corridor
    segment_type: SegmentType


@dataclass
class CorridorMovement:
    train_id: int
    start_node: RailwayNode
    end_node: RailwayNode
    start_time: float


class TrainIdGenerator:
    """
    Generates unique sequential train IDs.
    """

    def __init__(self, start: int = 1):
        self.current = start

    def next_id(self) -> int:
        train_id = self.current
        self.current += 1
        return train_id

class SegmentBuilder:
    def __init__(self):
        self.segments = []

    def _validate_nodes(self, corridor, start_node, end_node):
        path = CORRIDOR_PATHS[corridor]
        if start_node not in path:
            raise ValueError(f"{start_node.name} is not in {corridor.name}")
        if end_node not in path:
            raise ValueError(f"{end_node.name} is not in {corridor.name}")

    def add_move(self, train_id, corridor, start_node, end_node, start_time, end_time):
        self._validate_nodes(corridor, start_node, end_node)
        self.segments.append(
            Segment(
                train_id=train_id, start_time=start_time, end_time=end_time,
                start_node=start_node, end_node=end_node, corridor=corridor,
                segment_type=SegmentType.MOVE,
            )
        )

    def add_wait(self, train_id, corridor, node, start_time, end_time):
        self._validate_nodes(corridor, node, node)
        self.segments.append(
            Segment(
                train_id=train_id, start_time=start_time, end_time=end_time,
                start_node=node, end_node=node, corridor=corridor,
                segment_type=SegmentType.WAIT,
            )
        )

    def get_segments(self):
        return self.segments


class CorridorManager:
    def __init__(self):
        self.active_movements = {Corridor.C1: None, Corridor.C2: None}

    def is_free(self, corridor: Corridor) -> bool:
        return self.active_movements[corridor] is None

    def enter(self, corridor, train_id, start_node, end_node, current_time):
        if not self.is_free(corridor):
            current = self.active_movements[corridor]
            raise RuntimeError(f"{corridor.name} is occupied by Train {current.train_id}")
        
        self.active_movements[corridor] = CorridorMovement(
            train_id=train_id, start_node=start_node, end_node=end_node, start_time=current_time
        )

    def leave(self, corridor, train_id) -> CorridorMovement:
        movement = self.active_movements[corridor]
        if movement is None:
            raise RuntimeError(f"{corridor.name} is already free.")
        if movement.train_id != train_id:
            raise RuntimeError(f"Train {train_id} cannot release {corridor.name}.")
        
        self.active_movements[corridor] = None
        return movement


@dataclass
class StationSlot:
    slot_id: int
    train_id: Optional[int] = None

    @property
    def is_free(self) -> bool:
        return self.train_id is None


class StationManager:
    """
    Event-driven station capacity manager.

    SimPy Resource controls the real capacity.
    StationSlot objects are retained for debugging.
    """

    def __init__(
        self,
        env,
        bs_capacity: int = 2,
        yard_port_capacity: int = 2,
    ):
        if bs_capacity <= 0:
            raise ValueError(
                "BS capacity must be greater than zero."
            )

        if yard_port_capacity <= 0:
            raise ValueError(
                "Yard Port capacity must be greater than zero."
            )

        self.env = env

        self.slots = {
            Station.BS: [
                StationSlot(slot_id)
                for slot_id in range(
                    1,
                    bs_capacity + 1,
                )
            ],
            Station.YARD_PORT: [
                StationSlot(slot_id)
                for slot_id in range(
                    1,
                    yard_port_capacity + 1,
                )
            ],
        }

        self.resources = {
            Station.BS: simpy.Resource(
                env,
                capacity=bs_capacity,
            ),
            Station.YARD_PORT: simpy.Resource(
                env,
                capacity=yard_port_capacity,
            ),
        }

        self.active_requests = {}

    def available_slots(self, station: Station) -> int:
        return sum(
            slot.is_free
            for slot in self.slots[station]
        )

    def has_capacity(self, station: Station) -> bool:
        return self.available_slots(station) > 0

    def contains(
        self,
        station: Station,
        train_id: int,
    ) -> bool:
        return any(
            slot.train_id == train_id
            for slot in self.slots[station]
        )

    def request_entry(
        self,
        station: Station,
        train_id: int,
    ):
        """
        Wait event-driven until a station slot is available.

        Returns the allocated slot ID.
        """

        key = (
            station,
            train_id,
        )

        if key in self.active_requests:
            raise RuntimeError(
                f"Train {train_id} already has an active "
                f"reservation in {station.name}."
            )

        resource = self.resources[station]
        request = resource.request()

        # No polling. SimPy wakes the train when capacity is available.
        yield request

        free_slot = next(
            (
                slot
                for slot in self.slots[station]
                if slot.is_free
            ),
            None,
        )

        if free_slot is None:
            resource.release(request)

            raise RuntimeError(
                f"Resource/slot inconsistency in {station.name}."
            )

        free_slot.train_id = train_id
        self.active_requests[key] = request

        return free_slot.slot_id

    def release(
        self,
        station: Station,
        train_id: int,
    ):
        """
        Release a station slot and wake the next queued train.
        """

        key = (
            station,
            train_id,
        )

        request = self.active_requests.pop(
            key,
            None,
        )

        if request is None:
            raise RuntimeError(
                f"Train {train_id} has no reservation "
                f"in {station.name}."
            )

        occupied_slot = next(
            (
                slot
                for slot in self.slots[station]
                if slot.train_id == train_id
            ),
            None,
        )

        if occupied_slot is None:
            raise RuntimeError(
                f"Train {train_id} has no physical slot "
                f"in {station.name}."
            )

        occupied_slot.train_id = None

        self.resources[station].release(
            request
        )

    def status(self, station: Station):
        return [
            (
                slot.slot_id,
                slot.train_id,
            )
            for slot in self.slots[station]
        ]


class BaselineSimulator:
    def __init__(
        self,
        bs_capacity: int = 2,
        yard_port_capacity: int = 2,
    ):
        self.bs_capacity = bs_capacity
        self.yard_port_capacity = yard_port_capacity

        self.env = simpy.Environment()

        self.manager = CorridorManager()
        self.builder = SegmentBuilder()

        self.station_manager = StationManager(
            env=self.env,
            bs_capacity=bs_capacity,
            yard_port_capacity=yard_port_capacity,
        )

        self.corridor_resources = {
            Corridor.C1: simpy.Resource(
                self.env,
                capacity=1,
            ),
            Corridor.C2: simpy.Resource(
                self.env,
                capacity=1,
            ),
        }
        self.train_metadata = {}
    def get_corridor(
        self,
        corridor: Corridor,
    ):
        return self.corridor_resources[corridor]

    def add_train(
        self,
        process,
    ):
        return self.env.process(process)

    def run(
        self,
        until=None,
    ):
        self.env.run(until=until)

    def get_segments(self):
        return self.builder.get_segments()

    def reset(self):
        self.__init__(
            bs_capacity=self.bs_capacity,
            yard_port_capacity=self.yard_port_capacity,
        )

    def register_train_metadata(
        self,
        train_id: int,
        destination: str,
    ):
        """
        Store metadata for each train.
        """
        self.train_metadata[train_id] = {
            "destination": destination,
        }


    def get_train_destination(
        self,
        train_id: int,
    ) -> str:
        """
        Return the destination label of a train.
        """
        return self.train_metadata.get(
            train_id,
            {},
        ).get(
            "destination",
            "Unknown",
        )


    def get_train_metadata(self):
        return self.train_metadata

class TrainProcess:
    """
    Executes railway movements and waiting operations.
    """

    def __init__(self, simulator, train_id):
        self.sim = simulator
        self.env = simulator.env
        self.train_id = train_id

    def move(
        self,
        corridor: Corridor,
        start_node: RailwayNode,
        end_node: RailwayNode,
        travel_time: float,
    ):
        corridor_resource = self.sim.get_corridor(
            corridor
        )

        with corridor_resource.request() as request:
            yield request

            self.sim.manager.enter(
                corridor=corridor,
                train_id=self.train_id,
                start_node=start_node,
                end_node=end_node,
                current_time=self.env.now,
            )

            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor,
                start_node=start_node,
                end_node=end_node,
                start_time=self.env.now,
                end_time=self.env.now + travel_time,
            )

            yield self.env.timeout(
                travel_time
            )

            self.sim.manager.leave(
                corridor,
                self.train_id,
            )

    def wait(self, corridor: Corridor, node: RailwayNode, duration: float):
        self.sim.builder.add_wait(
            train_id=self.train_id, corridor=corridor, node=node,
            start_time=self.env.now, end_time=self.env.now + duration,
        )
        yield self.env.timeout(duration)

    def enter_station(
        self,
        station: Station,
    ):
        """
        Wait until station capacity becomes available,
        then reserve one station slot.
        """

        slot_id = yield from (
            self.sim.station_manager.request_entry(
                station=station,
                train_id=self.train_id,
            )
        )

        return slot_id


    def leave_station(
        self,
        station: Station,
    ):
        """
        Release the train's station slot.
        """

        self.sim.station_manager.release(
            station=station,
            train_id=self.train_id,
        )


class BaseTrain:
    def __init__(
        self,
        simulator,
        train_id,
        destination: str,
    ):
        self.sim = simulator
        self.env = simulator.env
        self.train_id = train_id
        self.destination = destination
        self.process = TrainProcess(
            simulator,
            train_id,
        )

        self.sim.register_train_metadata(
            train_id=train_id,
            destination=destination,
        )

    def run(self):
        raise NotImplementedError


class BarcoTrain(BaseTrain):
    """
    Executes one Barco train mission.

    Shared corridor
    ---------------
    Only BS Entry <-> SW1 belongs to the shared Corridor 1.

    The Barco branch starts after SW1 and is independent from
    the Steel Hormozgan branch.

    Therefore, Corridor 1 is requested separately for:
    1. Outbound movement
    2. Return movement
    """

    def __init__(
        self,
        simulator,
        train_id: int,
        travel_to_sw1: float = 8,
        barco_cycle_time: float = 12,
        travel_back: float = 8,
    ):
        super().__init__(
            simulator,
            train_id,
            destination="Barco",
        )
        self.travel_to_sw1 = travel_to_sw1
        self.barco_cycle_time = barco_cycle_time
        self.travel_back = travel_back

    def run(self):
        corridor = Corridor.C1
        resource = self.sim.get_corridor(
            corridor
        )

        # ===================================================
        # PHASE 1 — OUTBOUND SHARED CORRIDOR
        # BS Entry -> SW1
        # ===================================================

        with resource.request() as outbound_request:
            yield outbound_request

            self.sim.manager.enter(
                corridor=corridor,
                train_id=self.train_id,
                start_node=RailwayNode.BS_ENTRY,
                end_node=RailwayNode.SW1,
                current_time=self.env.now,
            )

            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor,
                start_node=RailwayNode.BS_ENTRY,
                end_node=RailwayNode.SW1,
                start_time=self.env.now,
                end_time=(
                    self.env.now
                    + self.travel_to_sw1
                ),
            )

            yield self.env.timeout(
                self.travel_to_sw1
            )

            # Train has cleared SW1 and entered the Barco branch.
            self.sim.manager.leave(
                corridor,
                self.train_id,
            )

        # ===================================================
        # PHASE 2 — BARCO BRANCH AND STATION OPERATION
        #
        # SW1 -> Barco -> operation -> SW1
        # Displayed as a horizontal line at SW1.
        #
        # Corridor 1 is FREE during this phase.
        # ===================================================

        self.sim.builder.add_wait(
            train_id=self.train_id,
            corridor=corridor,
            node=RailwayNode.SW1,
            start_time=self.env.now,
            end_time=(
                self.env.now
                + self.barco_cycle_time
            ),
        )

        yield self.env.timeout(
            self.barco_cycle_time
        )

        # ===================================================
        # PHASE 3 — RETURN SHARED CORRIDOR
        # SW1 -> BS Entry
        # ===================================================

        with resource.request() as return_request:
            yield return_request

            self.sim.manager.enter(
                corridor=corridor,
                train_id=self.train_id,
                start_node=RailwayNode.SW1,
                end_node=RailwayNode.BS_ENTRY,
                current_time=self.env.now,
            )

            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor,
                start_node=RailwayNode.SW1,
                end_node=RailwayNode.BS_ENTRY,
                start_time=self.env.now,
                end_time=(
                    self.env.now
                    + self.travel_back
                ),
            )

            yield self.env.timeout(
                self.travel_back
            )

            self.sim.manager.leave(
                corridor,
                self.train_id,
            )


class CopperTrain(BaseTrain):
    """
    Executes one complete Copper train mission.

    Operational route
    -----------------
    1. BS Entry -> SW1 -> Yard Port
    2. Shunting operation at Yard Port
    3. Yard Port -> SW2 -> Copper branch
    4. Loading/unloading at Copper
    5. Copper branch -> SW2 -> Yard Port
    6. Main locomotive attachment at Yard Port
    7. Yard Port -> SW2 -> SW3 -> BS Exit

    Visualization rule
    ------------------
    Copper station is outside the monitored corridors.

    Therefore, movement between SW2 and Copper, loading/unloading,
    and return from Copper to SW2 are represented by one horizontal
    WAIT segment at SW2.
    """

    def __init__(
        self,
        simulator,
        train_id: int,
        bs_to_sw1_time: float = 8,
        sw1_to_yard_time: float = 10,
        shunting_time: float = 15,
        yard_to_sw2_time: float = 10,
        copper_cycle_time: float = 30,
        sw2_to_yard_time: float = 10,
        main_loco_attach_time: float = 15,
        sw2_to_sw3_time: float = 10,
        sw3_to_bs_exit_time: float = 10,
    ):
        super().__init__(
            simulator,
            train_id,
            destination="Copper",
        )

        self.bs_to_sw1_time = bs_to_sw1_time
        self.sw1_to_yard_time = sw1_to_yard_time
        self.shunting_time = shunting_time
        self.yard_to_sw2_time = yard_to_sw2_time
        self.copper_cycle_time = copper_cycle_time
        self.sw2_to_yard_time = sw2_to_yard_time
        self.main_loco_attach_time = main_loco_attach_time
        self.sw2_to_sw3_time = sw2_to_sw3_time
        self.sw3_to_bs_exit_time = sw3_to_bs_exit_time


    def run(self):
        """
        Execute the complete Copper train mission.
        """

        # ===================================================
        # PHASE 1 — INBOUND THROUGH CORRIDOR 1
        # BS Entry -> SW1 -> Yard Port
        # ===================================================

        # Reserve Yard Port before entering Corridor 1.
        # If Yard Port is full, the train remains outside the corridor.
        yield from self.process.enter_station(
            Station.YARD_PORT
        )

        corridor_1 = Corridor.C1
        c1_resource = self.sim.get_corridor(corridor_1)

        with c1_resource.request() as c1_request:
            yield c1_request

            self.sim.manager.enter(
                corridor=corridor_1,
                train_id=self.train_id,
                start_node=RailwayNode.BS_ENTRY,
                end_node=RailwayNode.YARD_PORT,
                current_time=self.env.now,
            )

            # BS Entry -> SW1
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_1,
                start_node=RailwayNode.BS_ENTRY,
                end_node=RailwayNode.SW1,
                start_time=self.env.now,
                end_time=self.env.now + self.bs_to_sw1_time,
            )

            yield self.env.timeout(
                self.bs_to_sw1_time
            )

            # SW1 -> Yard Port
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_1,
                start_node=RailwayNode.SW1,
                end_node=RailwayNode.YARD_PORT,
                start_time=self.env.now,
                end_time=self.env.now + self.sw1_to_yard_time,
            )

            yield self.env.timeout(
                self.sw1_to_yard_time
            )

            # Train has fully entered Yard Port.
            self.sim.manager.leave(
                corridor_1,
                self.train_id,
            )

        # ===================================================
        # PHASE 2 — SHUNTING AT YARD PORT
        # ===================================================

        self.sim.builder.add_wait(
            train_id=self.train_id,
            corridor=Corridor.C2,
            node=RailwayNode.YARD_PORT,
            start_time=self.env.now,
            end_time=self.env.now + self.shunting_time,
        )

        yield self.env.timeout(
            self.shunting_time
        )

        # ===================================================
        # PHASE 3 — YARD PORT TO COPPER
        # Monitored section: Yard Port -> SW2
        # ===================================================

        corridor_2 = Corridor.C2
        c2_resource = self.sim.get_corridor(corridor_2)

        with c2_resource.request() as c2_request:
            yield c2_request

            self.sim.manager.enter(
                corridor=corridor_2,
                train_id=self.train_id,
                start_node=RailwayNode.YARD_PORT,
                end_node=RailwayNode.SW2,
                current_time=self.env.now,
            )

            # Yard Port -> SW2
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.YARD_PORT,
                end_node=RailwayNode.SW2,
                start_time=self.env.now,
                end_time=self.env.now + self.yard_to_sw2_time,
            )

            yield self.env.timeout(
                self.yard_to_sw2_time
            )

            # The train has completely left the monitored corridor
            # and entered the Copper branch.
            self.sim.manager.leave(
                corridor_2,
                self.train_id,
            )

        # The train has completely left Yard Port,
        # so its Yard Port slot can now be released.
        self.process.leave_station(
            Station.YARD_PORT
        )

        # ===================================================
        # PHASE 4 — COPPER OPERATION
        # SW2 -> Copper -> loading/unloading -> SW2
        #
        # This entire off-corridor operation is represented
        # as a horizontal line at SW2.
        # ===================================================

        self.sim.builder.add_wait(
            train_id=self.train_id,
            corridor=Corridor.C2,
            node=RailwayNode.SW2,
            start_time=self.env.now,
            end_time=self.env.now + self.copper_cycle_time,
        )

        yield self.env.timeout(
            self.copper_cycle_time
        )

        # ===================================================
        # PHASE 5 — RETURN TO YARD PORT
        # SW2 -> Yard Port
        # ===================================================

        # The train must reserve Yard Port before leaving Copper.
        yield from self.process.enter_station(
            Station.YARD_PORT
        )

        with c2_resource.request() as return_request:
            yield return_request

            self.sim.manager.enter(
                corridor=corridor_2,
                train_id=self.train_id,
                start_node=RailwayNode.SW2,
                end_node=RailwayNode.YARD_PORT,
                current_time=self.env.now,
            )

            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.SW2,
                end_node=RailwayNode.YARD_PORT,
                start_time=self.env.now,
                end_time=self.env.now + self.sw2_to_yard_time,
            )

            yield self.env.timeout(
                self.sw2_to_yard_time
            )

            # The train has completely entered Yard Port.
            self.sim.manager.leave(
                corridor_2,
                self.train_id,
            )

        # ===================================================
        # PHASE 6 — MAIN LOCOMOTIVE ATTACHMENT
        # ===================================================

        self.sim.builder.add_wait(
            train_id=self.train_id,
            corridor=Corridor.C2,
            node=RailwayNode.YARD_PORT,
            start_time=self.env.now,
            end_time=self.env.now + self.main_loco_attach_time,
        )

        yield self.env.timeout(
            self.main_loco_attach_time
        )

        # ===================================================
        # PHASE 7 — FINAL EXIT THROUGH CORRIDOR 2
        # Yard Port -> SW2 -> SW3 -> BS Exit
        # ===================================================

        with c2_resource.request() as exit_request:
            yield exit_request

            self.sim.manager.enter(
                corridor=corridor_2,
                train_id=self.train_id,
                start_node=RailwayNode.YARD_PORT,
                end_node=RailwayNode.BS_EXIT,
                current_time=self.env.now,
            )

            # Yard Port -> SW2
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.YARD_PORT,
                end_node=RailwayNode.SW2,
                start_time=self.env.now,
                end_time=self.env.now + self.yard_to_sw2_time,
            )

            yield self.env.timeout(
                self.yard_to_sw2_time
            )

            # The train has now completely left Yard Port.
            self.process.leave_station(
                Station.YARD_PORT
            )

            # SW2 -> SW3
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.SW2,
                end_node=RailwayNode.SW3,
                start_time=self.env.now,
                end_time=self.env.now + self.sw2_to_sw3_time,
            )

            yield self.env.timeout(
                self.sw2_to_sw3_time
            )

            # SW3 -> BS Exit
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.SW3,
                end_node=RailwayNode.BS_EXIT,
                start_time=self.env.now,
                end_time=self.env.now + self.sw3_to_bs_exit_time,
            )

            yield self.env.timeout(
                self.sw3_to_bs_exit_time
            )

            # Train has completely exited Corridor 2.
            self.sim.manager.leave(
                corridor_2,
                self.train_id,
            )


class SulfurTrain(BaseTrain):
    """
    Executes one complete Sulfur train mission.

    Operational route
    -----------------
    1. BS Entry -> SW1 -> Yard Port
    2. Shunting operation at Yard Port
    3. Yard Port -> SW2 -> SW3 -> Sulfur branch
    4. Loading/unloading at Sulfur
    5. Sulfur branch -> SW3 -> SW2 -> Yard Port
    6. Main locomotive attachment at Yard Port
    7. Yard Port -> SW2 -> SW3 -> BS Exit

    Visualization rule
    ------------------
    Sulfur station is outside the monitored corridors.

    Therefore, travel from SW3 to Sulfur, loading/unloading,
    and return from Sulfur to SW3 are represented by one
    horizontal WAIT segment at SW3.
    """

    def __init__(
        self,
        simulator,
        train_id: int,
        bs_to_sw1_time: float = 8,
        sw1_to_yard_time: float = 10,
        shunting_time: float = 15,
        yard_to_sw2_time: float = 10,
        sw2_to_sw3_time: float = 10,
        sulfur_cycle_time: float = 30,
        sw3_to_sw2_time: float = 10,
        sw2_to_yard_time: float = 10,
        main_loco_attach_time: float = 15,
        sw3_to_bs_exit_time: float = 10,
    ):
        super().__init__(
            simulator,
            train_id,
            destination="Sulfur",
        )

        self.bs_to_sw1_time = bs_to_sw1_time
        self.sw1_to_yard_time = sw1_to_yard_time
        self.shunting_time = shunting_time
        self.yard_to_sw2_time = yard_to_sw2_time
        self.sw2_to_sw3_time = sw2_to_sw3_time
        self.sulfur_cycle_time = sulfur_cycle_time
        self.sw3_to_sw2_time = sw3_to_sw2_time
        self.sw2_to_yard_time = sw2_to_yard_time
        self.main_loco_attach_time = main_loco_attach_time
        self.sw3_to_bs_exit_time = sw3_to_bs_exit_time


    def run(self):
        """
        Execute the complete Sulfur train mission.
        """

        corridor_1 = Corridor.C1
        corridor_2 = Corridor.C2

        c1_resource = self.sim.get_corridor(corridor_1)
        c2_resource = self.sim.get_corridor(corridor_2)

        # ===================================================
        # PHASE 1 — RESERVE YARD PORT
        # ===================================================

        yield from self.process.enter_station(
            Station.YARD_PORT
        )

        # ===================================================
        # PHASE 2 — INBOUND THROUGH CORRIDOR 1
        # BS Entry -> SW1 -> Yard Port
        # ===================================================

        with c1_resource.request() as c1_request:
            yield c1_request

            self.sim.manager.enter(
                corridor=corridor_1,
                train_id=self.train_id,
                start_node=RailwayNode.BS_ENTRY,
                end_node=RailwayNode.YARD_PORT,
                current_time=self.env.now,
            )

            # BS Entry -> SW1
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_1,
                start_node=RailwayNode.BS_ENTRY,
                end_node=RailwayNode.SW1,
                start_time=self.env.now,
                end_time=self.env.now + self.bs_to_sw1_time,
            )

            yield self.env.timeout(
                self.bs_to_sw1_time
            )

            # SW1 -> Yard Port
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_1,
                start_node=RailwayNode.SW1,
                end_node=RailwayNode.YARD_PORT,
                start_time=self.env.now,
                end_time=self.env.now + self.sw1_to_yard_time,
            )

            yield self.env.timeout(
                self.sw1_to_yard_time
            )

            self.sim.manager.leave(
                corridor_1,
                self.train_id,
            )

        # ===================================================
        # PHASE 3 — SHUNTING AT YARD PORT
        # ===================================================

        self.sim.builder.add_wait(
            train_id=self.train_id,
            corridor=corridor_2,
            node=RailwayNode.YARD_PORT,
            start_time=self.env.now,
            end_time=self.env.now + self.shunting_time,
        )

        yield self.env.timeout(
            self.shunting_time
        )

        # ===================================================
        # PHASE 4 — YARD PORT TO SULFUR BRANCH
        # Yard Port -> SW2 -> SW3
        # ===================================================

        with c2_resource.request() as outbound_request:
            yield outbound_request

            self.sim.manager.enter(
                corridor=corridor_2,
                train_id=self.train_id,
                start_node=RailwayNode.YARD_PORT,
                end_node=RailwayNode.SW3,
                current_time=self.env.now,
            )

            # Yard Port -> SW2
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.YARD_PORT,
                end_node=RailwayNode.SW2,
                start_time=self.env.now,
                end_time=self.env.now + self.yard_to_sw2_time,
            )

            yield self.env.timeout(
                self.yard_to_sw2_time
            )

            # SW2 -> SW3
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.SW2,
                end_node=RailwayNode.SW3,
                start_time=self.env.now,
                end_time=self.env.now + self.sw2_to_sw3_time,
            )

            yield self.env.timeout(
                self.sw2_to_sw3_time
            )

            # Train has completely entered the Sulfur branch.
            self.sim.manager.leave(
                corridor_2,
                self.train_id,
            )

        # Yard Port is physically clear now.
        self.process.leave_station(
            Station.YARD_PORT
        )

        # ===================================================
        # PHASE 5 — SULFUR OPERATION
        #
        # SW3 -> Sulfur -> loading/unloading -> SW3
        # is displayed as one horizontal segment at SW3.
        # ===================================================

        self.sim.builder.add_wait(
            train_id=self.train_id,
            corridor=corridor_2,
            node=RailwayNode.SW3,
            start_time=self.env.now,
            end_time=self.env.now + self.sulfur_cycle_time,
        )

        yield self.env.timeout(
            self.sulfur_cycle_time
        )

        # ===================================================
        # PHASE 6 — RESERVE YARD PORT BEFORE RETURN
        # ===================================================

        yield from self.process.enter_station(
            Station.YARD_PORT
        )

        # ===================================================
        # PHASE 7 — RETURN TO YARD PORT
        # SW3 -> SW2 -> Yard Port
        # ===================================================

        with c2_resource.request() as return_request:
            yield return_request

            self.sim.manager.enter(
                corridor=corridor_2,
                train_id=self.train_id,
                start_node=RailwayNode.SW3,
                end_node=RailwayNode.YARD_PORT,
                current_time=self.env.now,
            )

            # SW3 -> SW2
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.SW3,
                end_node=RailwayNode.SW2,
                start_time=self.env.now,
                end_time=self.env.now + self.sw3_to_sw2_time,
            )

            yield self.env.timeout(
                self.sw3_to_sw2_time
            )

            # SW2 -> Yard Port
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.SW2,
                end_node=RailwayNode.YARD_PORT,
                start_time=self.env.now,
                end_time=self.env.now + self.sw2_to_yard_time,
            )

            yield self.env.timeout(
                self.sw2_to_yard_time
            )

            self.sim.manager.leave(
                corridor_2,
                self.train_id,
            )

        # ===================================================
        # PHASE 8 — MAIN LOCOMOTIVE ATTACHMENT
        # ===================================================

        self.sim.builder.add_wait(
            train_id=self.train_id,
            corridor=corridor_2,
            node=RailwayNode.YARD_PORT,
            start_time=self.env.now,
            end_time=self.env.now + self.main_loco_attach_time,
        )

        yield self.env.timeout(
            self.main_loco_attach_time
        )

        # ===================================================
        # PHASE 9 — FINAL EXIT THROUGH CORRIDOR 2
        # Yard Port -> SW2 -> SW3 -> BS Exit
        # ===================================================

        with c2_resource.request() as exit_request:
            yield exit_request

            self.sim.manager.enter(
                corridor=corridor_2,
                train_id=self.train_id,
                start_node=RailwayNode.YARD_PORT,
                end_node=RailwayNode.BS_EXIT,
                current_time=self.env.now,
            )

            # Yard Port -> SW2
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.YARD_PORT,
                end_node=RailwayNode.SW2,
                start_time=self.env.now,
                end_time=self.env.now + self.yard_to_sw2_time,
            )

            yield self.env.timeout(
                self.yard_to_sw2_time
            )

            # Train has now fully left Yard Port.
            self.process.leave_station(
                Station.YARD_PORT
            )

            # SW2 -> SW3
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.SW2,
                end_node=RailwayNode.SW3,
                start_time=self.env.now,
                end_time=self.env.now + self.sw2_to_sw3_time,
            )

            yield self.env.timeout(
                self.sw2_to_sw3_time
            )

            # SW3 -> BS Exit
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.SW3,
                end_node=RailwayNode.BS_EXIT,
                start_time=self.env.now,
                end_time=self.env.now + self.sw3_to_bs_exit_time,
            )

            yield self.env.timeout(
                self.sw3_to_bs_exit_time
            )

            self.sim.manager.leave(
                corridor_2,
                self.train_id,
            )

class YardPortTrain(BaseTrain):
    """
    Executes one complete Yard Port train mission.

    Operational route
    -----------------
    1. BS Entry -> SW1 -> Yard Port
    2. Loading/unloading at Yard Port
    3. Yard Port -> SW2 -> SW3 -> BS Exit

    Railway rules
    -------------
    - Yard Port capacity is 2.
    - The train must reserve a Yard Port slot before entering Corridor 1.
    - If Yard Port is full, the train waits before entering Corridor 1.
    - The Yard Port slot remains occupied during loading/unloading.
    - The slot is released only after the train completely leaves Yard Port.
    - Corridor 1 and Corridor 2 each have capacity 1.
    """

    def __init__(
        self,
        simulator,
        train_id: int,
        bs_to_sw1_time: float = 8,
        sw1_to_yard_time: float = 10,
        loading_time: float = 30,
        yard_to_sw2_time: float = 10,
        sw2_to_sw3_time: float = 10,
        sw3_to_bs_exit_time: float = 10,
    ):
        super().__init__(
            simulator,
            train_id,
            destination="Yard Port",
        )

        self.bs_to_sw1_time = bs_to_sw1_time
        self.sw1_to_yard_time = sw1_to_yard_time
        self.loading_time = loading_time
        self.yard_to_sw2_time = yard_to_sw2_time
        self.sw2_to_sw3_time = sw2_to_sw3_time
        self.sw3_to_bs_exit_time = sw3_to_bs_exit_time


    def run(self):
        """
        Execute the complete Yard Port mission.
        """

        corridor_1 = Corridor.C1
        corridor_2 = Corridor.C2

        c1_resource = self.sim.get_corridor(corridor_1)
        c2_resource = self.sim.get_corridor(corridor_2)

        # ===================================================
        # PHASE 1 — RESERVE YARD PORT
        # ===================================================

        yield from self.process.enter_station(
            Station.YARD_PORT
        )

        # ===================================================
        # PHASE 2 — INBOUND THROUGH CORRIDOR 1
        # BS Entry -> SW1 -> Yard Port
        # ===================================================

        with c1_resource.request() as c1_request:
            yield c1_request

            self.sim.manager.enter(
                corridor=corridor_1,
                train_id=self.train_id,
                start_node=RailwayNode.BS_ENTRY,
                end_node=RailwayNode.YARD_PORT,
                current_time=self.env.now,
            )

            # BS Entry -> SW1
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_1,
                start_node=RailwayNode.BS_ENTRY,
                end_node=RailwayNode.SW1,
                start_time=self.env.now,
                end_time=self.env.now + self.bs_to_sw1_time,
            )

            yield self.env.timeout(
                self.bs_to_sw1_time
            )

            # SW1 -> Yard Port
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_1,
                start_node=RailwayNode.SW1,
                end_node=RailwayNode.YARD_PORT,
                start_time=self.env.now,
                end_time=self.env.now + self.sw1_to_yard_time,
            )

            yield self.env.timeout(
                self.sw1_to_yard_time
            )

            # Train has fully entered Yard Port.
            self.sim.manager.leave(
                corridor_1,
                self.train_id,
            )

        # ===================================================
        # PHASE 3 — LOADING / UNLOADING AT YARD PORT
        # ===================================================

        self.sim.builder.add_wait(
            train_id=self.train_id,
            corridor=corridor_2,
            node=RailwayNode.YARD_PORT,
            start_time=self.env.now,
            end_time=self.env.now + self.loading_time,
        )

        yield self.env.timeout(
            self.loading_time
        )

        # ===================================================
        # PHASE 4 — FINAL EXIT THROUGH CORRIDOR 2
        # Yard Port -> SW2 -> SW3 -> BS Exit
        # ===================================================

        with c2_resource.request() as c2_request:
            yield c2_request

            self.sim.manager.enter(
                corridor=corridor_2,
                train_id=self.train_id,
                start_node=RailwayNode.YARD_PORT,
                end_node=RailwayNode.BS_EXIT,
                current_time=self.env.now,
            )

            # Yard Port -> SW2
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.YARD_PORT,
                end_node=RailwayNode.SW2,
                start_time=self.env.now,
                end_time=self.env.now + self.yard_to_sw2_time,
            )

            yield self.env.timeout(
                self.yard_to_sw2_time
            )

            # The train has completely left Yard Port.
            self.process.leave_station(
                Station.YARD_PORT
            )

            # SW2 -> SW3
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.SW2,
                end_node=RailwayNode.SW3,
                start_time=self.env.now,
                end_time=self.env.now + self.sw2_to_sw3_time,
            )

            yield self.env.timeout(
                self.sw2_to_sw3_time
            )

            # SW3 -> BS Exit
            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor_2,
                start_node=RailwayNode.SW3,
                end_node=RailwayNode.BS_EXIT,
                start_time=self.env.now,
                end_time=self.env.now + self.sw3_to_bs_exit_time,
            )

            yield self.env.timeout(
                self.sw3_to_bs_exit_time
            )

            # Train has completely exited Corridor 2.
            self.sim.manager.leave(
                corridor_2,
                self.train_id,
            )

class SteelHormozganTrain(BaseTrain):
    """
    Executes one Steel Hormozgan train mission.

    Barco and Steel Hormozgan use different branches after SW1.

    Corridor 1 is occupied only while the train is moving through
    the shared BS Entry <-> SW1 section.
    """

    def __init__(
        self,
        simulator,
        train_id: int,
        travel_to_sw1: float = 8,
        steel_cycle_time: float = 16,
        travel_back: float = 8,
    ):
        super().__init__(
            simulator,
            train_id,
            destination="Steel Hormozgan",
        )

        self.travel_to_sw1 = travel_to_sw1
        self.steel_cycle_time = steel_cycle_time
        self.travel_back = travel_back

    def run(self):
        corridor = Corridor.C1
        resource = self.sim.get_corridor(
            corridor
        )

        # ===================================================
        # PHASE 1 — OUTBOUND SHARED CORRIDOR
        # BS Entry -> SW1
        # ===================================================

        with resource.request() as outbound_request:
            yield outbound_request

            self.sim.manager.enter(
                corridor=corridor,
                train_id=self.train_id,
                start_node=RailwayNode.BS_ENTRY,
                end_node=RailwayNode.SW1,
                current_time=self.env.now,
            )

            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor,
                start_node=RailwayNode.BS_ENTRY,
                end_node=RailwayNode.SW1,
                start_time=self.env.now,
                end_time=(
                    self.env.now
                    + self.travel_to_sw1
                ),
            )

            yield self.env.timeout(
                self.travel_to_sw1
            )

            # Train has entered the Steel branch.
            self.sim.manager.leave(
                corridor,
                self.train_id,
            )

        # ===================================================
        # PHASE 2 — STEEL HORMOZGAN BRANCH OPERATION
        #
        # Corridor 1 is free during this phase.
        # ===================================================

        self.sim.builder.add_wait(
            train_id=self.train_id,
            corridor=corridor,
            node=RailwayNode.SW1,
            start_time=self.env.now,
            end_time=(
                self.env.now
                + self.steel_cycle_time
            ),
        )

        yield self.env.timeout(
            self.steel_cycle_time
        )

        # ===================================================
        # PHASE 3 — RETURN SHARED CORRIDOR
        # SW1 -> BS Entry
        # ===================================================

        with resource.request() as return_request:
            yield return_request

            self.sim.manager.enter(
                corridor=corridor,
                train_id=self.train_id,
                start_node=RailwayNode.SW1,
                end_node=RailwayNode.BS_ENTRY,
                current_time=self.env.now,
            )

            self.sim.builder.add_move(
                train_id=self.train_id,
                corridor=corridor,
                start_node=RailwayNode.SW1,
                end_node=RailwayNode.BS_ENTRY,
                start_time=self.env.now,
                end_time=(
                    self.env.now
                    + self.travel_back
                ),
            )

            yield self.env.timeout(
                self.travel_back
            )

            self.sim.manager.leave(
                corridor,
                self.train_id,
            )


class MovementDiagram:
    """
    Draw a cleaner railway movement diagram.

    Improvements
    ------------
    - Colors are assigned by destination, not by train ID.
    - MOVE and WAIT segments use different line styles.
    - Lines are thinner.
    - The legend is based on destinations, not all train IDs.
    - Optional train ID labels can be shown at the end of each line.
    """

    DESTINATION_COLORS = {
        "Barco": "tab:blue",
        "Steel Hormozgan": "tab:orange",
        "Yard Port": "tab:green",
        "Copper": "tab:red",
        "Sulfur": "tab:purple",
        "Unknown": "gray",
    }

    def __init__(
        self,
        segments,
        train_metadata=None,
    ):
        self.segments = sorted(
            segments,
            key=lambda segment: (
                segment.train_id,
                segment.start_time,
                segment.end_time,
            ),
        )

        self.train_metadata = (
            train_metadata
            if train_metadata is not None
            else {}
        )

    @staticmethod
    def node_y(node: RailwayNode) -> int:
        return int(node)

    def group_by_train(self):
        trains = defaultdict(list)

        for segment in self.segments:
            trains[segment.train_id].append(segment)

        for train_id in trains:
            trains[train_id].sort(
                key=lambda segment: (
                    segment.start_time,
                    segment.end_time,
                )
            )

        return trains

    def get_train_destination(
        self,
        train_id: int,
    ) -> str:
        return self.train_metadata.get(
            train_id,
            {},
        ).get(
            "destination",
            "Unknown",
        )

    def get_train_color(
        self,
        train_id: int,
    ) -> str:
        destination = self.get_train_destination(
            train_id
        )

        return self.DESTINATION_COLORS.get(
            destination,
            self.DESTINATION_COLORS["Unknown"],
        )

    def validate_train_continuity(
        self,
        train_id,
        train_segments,
    ):
        tolerance = 1e-9

        for previous, current in zip(
            train_segments,
            train_segments[1:],
        ):
            if current.start_time < previous.end_time - tolerance:
                raise RuntimeError(
                    f"Train {train_id} has overlapping segments: "
                    f"{previous.end_time} > {current.start_time}"
                )

            if previous.end_node != current.start_node:
                raise RuntimeError(
                    f"Train {train_id} has an invalid spatial jump: "
                    f"{previous.end_node.name} -> "
                    f"{current.start_node.name}"
                )

    def draw_train(
        self,
        ax,
        train_id,
        train_segments,
        show_train_ids=False,
    ):
        """
        Draw one complete train path.

        All MOVE, WAIT and implicit waiting segments are shown
        using solid lines. Color represents destination.
        """

        self.validate_train_continuity(
            train_id,
            train_segments,
        )

        color = self.get_train_color(
            train_id
        )

        previous_segment = None

        for segment in train_segments:
            # Draw unrecorded waiting between two consecutive segments.
            if previous_segment is not None:
                gap = (
                    segment.start_time
                    - previous_segment.end_time
                )

                if gap > 1e-9:
                    waiting_y = self.node_y(
                        previous_segment.end_node
                    )

                    ax.plot(
                        [
                            previous_segment.end_time,
                            segment.start_time,
                        ],
                        [
                            waiting_y,
                            waiting_y,
                        ],
                        color=color,
                        linewidth=0.75,
                        linestyle="-",
                        alpha=0.55,
                        zorder=1,
                    )

            x_values = [
                segment.start_time,
                segment.end_time,
            ]

            y_values = [
                self.node_y(segment.start_node),
                self.node_y(segment.end_node),
            ]

            # All segments are solid.
            # MOVE segments are slightly stronger than WAIT segments.
            if segment.segment_type == SegmentType.MOVE:
                line_width = 1.05
                line_alpha = 0.90
                z_order = 3
            else:
                line_width = 0.80
                line_alpha = 0.62
                z_order = 2

            ax.plot(
                x_values,
                y_values,
                color=color,
                linewidth=line_width,
                linestyle="-",
                alpha=line_alpha,
                zorder=z_order,
            )

            previous_segment = segment

        # Only show start and final points.
        first_segment = train_segments[0]
        last_segment = train_segments[-1]

        ax.scatter(
            [first_segment.start_time],
            [self.node_y(first_segment.start_node)],
            color=color,
            s=12,
            alpha=0.90,
            zorder=4,
        )

        ax.scatter(
            [last_segment.end_time],
            [self.node_y(last_segment.end_node)],
            color=color,
            s=12,
            alpha=0.90,
            zorder=4,
        )

        if show_train_ids:
            ax.annotate(
                f"T{train_id}",
                (
                    last_segment.end_time,
                    self.node_y(last_segment.end_node),
                ),
                xytext=(3, 1),
                textcoords="offset points",
                fontsize=6,
                color=color,
                va="bottom",
            )

    def build_destination_legend(self):
        legend_handles = []

        ordered_destinations = [
            "Barco",
            "Steel Hormozgan",
            "Yard Port",
            "Copper",
            "Sulfur",
        ]

        for destination in ordered_destinations:
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=self.DESTINATION_COLORS[destination],
                    lw=1.8,
                    linestyle="-",
                    label=destination,
                )
            )

        return legend_handles

    def build_style_legend(self):
        return [
            Line2D(
                [0],
                [0],
                color="black",
                lw=1.2,
                linestyle="-",
                label="MOVE segment",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                lw=1.0,
                linestyle="--",
                label="Recorded WAIT segment",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                lw=0.9,
                linestyle=":",
                label="Implicit waiting gap",
            ),
        ]

    def calculate_summary(
        self,
        generation_horizon=None,
    ):
        """
        Calculate summary information for the plotted simulation.
        """

        trains = self.group_by_train()

        processed_train_ids = set(
            trains.keys()
        )

        completed_train_ids = set()

        for train_id, train_segments in trains.items():
            if not train_segments:
                continue

            last_segment = train_segments[-1]

            if last_segment.end_node in {
                RailwayNode.BS_ENTRY,
                RailwayNode.BS_EXIT,
            }:
                completed_train_ids.add(
                    train_id
                )

        destination_counts = defaultdict(int)

        for train_id in processed_train_ids:
            destination = self.get_train_destination(
                train_id
            )

            destination_counts[
                destination
            ] += 1

        generated_count = len(
            self.train_metadata
        )

        processed_count = len(
            processed_train_ids
        )

        completed_count = len(
            completed_train_ids
        )

        incomplete_count = (
            generated_count
            - completed_count
        )

        return {
            "generated": generated_count,
            "processed": processed_count,
            "completed": completed_count,
            "incomplete": incomplete_count,
            "segments": len(self.segments),
            "destination_counts": dict(
                destination_counts
            ),
            "generation_horizon": generation_horizon,
        }


    def build_information_text(
        self,
        summary,
    ):
        """
        Build the simulation information displayed inside the plot.
        """

        lines = [
            f"Generated trains: {summary['generated']}",
            f"Processed trains: {summary['processed']}",
            f"Completed trains: {summary['completed']}",
            f"Incomplete/active: {summary['incomplete']}",
            f"Recorded segments: {summary['segments']}",
            "",
            "Trains by destination:",
        ]

        destination_order = [
            "Yard Port",
            "Barco",
            "Steel Hormozgan",
            "Copper",
            "Sulfur",
        ]

        for destination in destination_order:
            count = summary[
                "destination_counts"
            ].get(
                destination,
                0,
            )

            lines.append(
                f"  {destination}: {count}"
            )

        if summary["generation_horizon"] is not None:
            lines.extend(
                [
                    "",
                    (
                        "Generation horizon: "
                        f"{summary['generation_horizon']:.0f} h"
                    ),
                ]
            )

        return "\n".join(
            lines
        )

    def plot(
        self,
        show_train_ids=False,
        x_limit=None,
        generation_horizon=None,
        show_information=False,
    ):
        """
        Draw the destination-based train movement diagram.

        Destination legend is kept.
        Summary information box is removed.
        """

        fig, ax = plt.subplots(
            figsize=(16, 9)
        )

        trains = self.group_by_train()

        for train_id in sorted(trains):
            self.draw_train(
                ax=ax,
                train_id=train_id,
                train_segments=trains[train_id],
                show_train_ids=show_train_ids,
            )

        ax.set_yticks(
            [int(node) for node in RailwayNode]
        )

        ax.set_yticklabels(
            [NODE_LABELS[node] for node in RailwayNode]
        )

        ax.set_xlabel(
            "Simulation Time (hours)",
            fontsize=12,
        )

        ax.set_ylabel(
            "Railway Node",
            fontsize=12,
        )

        ax.set_title(
            "Baseline Train Movement Diagram",
            fontsize=16,
        )

        ax.grid(
            True,
            linestyle="--",
            linewidth=0.7,
            alpha=0.28,
            zorder=0,
        )

        if x_limit is not None:
            ax.set_xlim(
                0,
                x_limit,
            )

        ax.margins(
            x=0.01,
            y=0.06,
        )

        # Keep destination legend only
        destination_legend = ax.legend(
            handles=self.build_destination_legend(),
            title="Destination",
            loc="upper left",
            bbox_to_anchor=(1.01, 1.00),
            frameon=True,
            fontsize=9,
            title_fontsize=10,
        )

        ax.add_artist(
            destination_legend
        )

        plt.tight_layout(
            rect=[0, 0, 0.86, 1]
        )

        output_path = (
            "baseline_movement_diagram.png"
        )

        fig.savefig(
            output_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"Movement diagram saved to: "
            f"{output_path}"
        )

        plt.show()


def start_train_after_delay(
    simulator: BaselineSimulator,
    train,
    delay: float,
):
    """
    Start a train after a specified simulation delay.
    """

    if delay < 0:
        raise ValueError("Train delay cannot be negative.")

    yield simulator.env.timeout(delay)

    yield simulator.env.process(
        train.run()
    )

def validate_corridor_overlaps(segments):
    """
    Check that MOVE segments do not overlap inside each corridor.

    WAIT segments are ignored because they may represent operations
    outside the physical corridor.
    """

    tolerance = 1e-9

    for corridor in Corridor:
        movements = sorted(
            [
                segment
                for segment in segments
                if (
                    segment.corridor == corridor
                    and segment.segment_type == SegmentType.MOVE
                )
            ],
            key=lambda segment: (
                segment.start_time,
                segment.end_time,
            ),
        )

        for previous, current in zip(
            movements,
            movements[1:],
        ):
            if current.start_time < previous.end_time - tolerance:
                raise RuntimeError(
                    f"Illegal overlap in {corridor.name}: "
                    f"Train {previous.train_id} "
                    f"[{previous.start_time}, {previous.end_time}] "
                    f"and Train {current.train_id} "
                    f"[{current.start_time}, {current.end_time}]"
                )

    print(
        "Validation passed: "
        "no overlapping MOVE segments were found."
    )

def station_test_process(
    simulator: BaselineSimulator,
    train_id: int,
    arrival_time: float,
    holding_time: float,
):
    yield simulator.env.timeout(
        arrival_time
    )

    print(
        f"t={simulator.env.now:.2f} | "
        f"Train {train_id} requests Yard Port"
    )

    slot_id = yield from (
        simulator.station_manager.request_entry(
            station=Station.YARD_PORT,
            train_id=train_id,
        )
    )

    print(
        f"t={simulator.env.now:.2f} | "
        f"Train {train_id} enters slot {slot_id}"
    )

    yield simulator.env.timeout(
        holding_time
    )

    simulator.station_manager.release(
        station=Station.YARD_PORT,
        train_id=train_id,
    )

    print(
        f"t={simulator.env.now:.2f} | "
        f"Train {train_id} leaves Yard Port"
    )


if __name__ == "__main__":
    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    inputs = load_baseline_plot_inputs()

    generation_horizon = 24.0

    # Additional time allows late trains to finish.
    simulation_end = 60.0

    sim = BaselineSimulator(
        bs_capacity=inputs.bs_capacity,
        yard_port_capacity=inputs.yard_port_capacity,
    )

    train_ids = TrainIdGenerator()

    generated_counts = {
        "Yard Port": 0,
        "Barco": 0,
        "Steel": 0,
        "Copper": 0,
        "Sulfur": 0,
    }

    sim.add_train(
        port_paired_train_generator(
            simulator=sim,
            inputs=inputs,
            train_ids=train_ids,
            generation_horizon=generation_horizon,
            generated_counts=generated_counts,
            within_pair_gap=BASELINE.within_gap_hours,
        )
    )

    sim.add_train(
        north_train_generator(
            simulator=sim,
            inputs=inputs,
            train_ids=train_ids,
            generation_horizon=generation_horizon,
            generated_counts=generated_counts,
        )
    )

    sim.add_train(
        south_train_generator(
            simulator=sim,
            inputs=inputs,
            train_ids=train_ids,
            generation_horizon=generation_horizon,
            generated_counts=generated_counts,
        )
    )

    sim.run(
        until=simulation_end
    )

    print(
        "\n=== Generated trains during 24 hours ==="
    )

    total_generated = 0

    for destination, count in generated_counts.items():
        print(
            f"{destination}: {count}"
        )

        total_generated += count

    print(
        f"Total generated: {total_generated}"
    )

    segments = sim.get_segments()

    print(
        f"Generated segments: {len(segments)}"
    )

    assert total_generated > 0
    assert len(segments) > 0

    print(
        "Twenty-four-hour generation test passed."
    )

    # ===================================================
    # TRAIN METADATA TEST
    # Must be checked after trains are generated.
    # ===================================================

    print(
        "\n=== Train metadata ==="
    )

    metadata = sim.get_train_metadata()

    for train_id, meta in sorted(
        metadata.items()
    ):
        print(
            train_id,
            meta,
        )

    assert len(metadata) > 0

    assert len(metadata) == total_generated, (
        f"Metadata count ({len(metadata)}) does not match "
        f"generated train count ({total_generated})."
    )

    valid_destinations = {
        "Yard Port",
        "Barco",
        "Steel Hormozgan",
        "Copper",
        "Sulfur",
    }

    for train_id, meta in metadata.items():
        destination = meta.get(
            "destination"
        )

        assert destination in valid_destinations, (
            f"Train {train_id} has invalid destination: "
            f"{destination}"
        )

    print(
        "Train metadata registration passed."
    )

    # ===================================================
    # CORRIDOR VALIDATION
    # ===================================================

    validate_corridor_overlaps(
        segments
    )

    # ===================================================
    # YARD PORT FINAL STATUS
    # ===================================================

    final_yard_status = (
        sim.station_manager.status(
            Station.YARD_PORT
        )
    )

    print(
        "Final Yard Port status:",
        final_yard_status,
    )

    if not all(
        train_id is None
        for _, train_id in final_yard_status
    ):
        print(
            "Warning: one or more trains are still active "
            "at the end of the simulation."
        )

    # ===================================================
    # DRAW CLEAN DESTINATION-BASED DIAGRAM
    # ===================================================
    diagram = MovementDiagram(
        segments=segments,
        train_metadata=metadata,
    )

    summary = diagram.calculate_summary(
        generation_horizon=24,
    )

    print(
        "\n=== Simulation summary ==="
    )

    print(
        f"Generated trains: "
        f"{summary['generated']}"
    )

    print(
        f"Processed trains: "
        f"{summary['processed']}"
    )

    print(
        f"Completed trains: "
        f"{summary['completed']}"
    )

    print(
        f"Incomplete/active trains: "
        f"{summary['incomplete']}"
    )

    print(
        f"Recorded segments: "
        f"{summary['segments']}"
    )

    print(
        "Destination counts:",
        summary["destination_counts"],
    )

    diagram.plot(
        show_train_ids=False,
        x_limit=24,
        show_information=False,
    )