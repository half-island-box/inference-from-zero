class ProcessGroupManager:
    """Tracks the tensor-parallel process group size."""

    def __init__(self) -> None:
        self.tp_world_size: int = 1

    def init(self, tp_world_size: int) -> None:
        self.tp_world_size = tp_world_size


process_group_manager = ProcessGroupManager()
