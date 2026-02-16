from path_planning.multi_agent_planner.centralized.cbs.cbs import Environment, CBS
from path_planning.multi_agent_planner.centralized.icbs.icbs import IEnvironment, ICBS
from typing import Tuple


def get_centralized(centralized_alg_name:str) -> Tuple[CBS,Environment]:
    centralized_alg_name = centralized_alg_name.lower()
    if centralized_alg_name == "cbs":
        return CBS,Environment
    elif centralized_alg_name == "icbs":
        return ICBS,IEnvironment
    else:
        raise ValueError(f"Invalid algorithm: {centralized_alg_name}")