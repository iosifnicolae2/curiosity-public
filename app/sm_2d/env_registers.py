from gym import register
from gym_minigrid.minigrid import MiniGridEnv, Grid, Goal


class EmptyEnvNoLimit(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=30000,
            # Set this to True for maximum speed
            see_through_walls=True,
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def _reward(self):
        """
        Compute the reward to be given upon success
        """

        return 0.99 ** self.step_count


class EmptyEnvNoLimit6X6(EmptyEnvNoLimit):
    def __init__(self):
        super().__init__(size=6)


register(
    id='MiniGridNoLimit-Empty-6x6-v0',
    entry_point='app.sm_2d.env_registers:EmptyEnvNoLimit6X6'
)
