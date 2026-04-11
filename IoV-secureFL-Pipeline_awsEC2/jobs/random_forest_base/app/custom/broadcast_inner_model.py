"""
BroadcastInnerModel — a lightweight NVFlare controller that broadcasts the
globally aggregated Stage 1 (inner) model to all clients before Stage 2 training.

This fixes the prob_ATTACK distribution mismatch: without this step, Stage 2
ScatterAndGather sends an empty shareable (start_round=0, allow_empty_global_weights=True)
so clients fall back to their local 20-tree model. With this controller, every client
receives the full globally aggregated inner model (100 trees from all 5 sites) before
generating the prob_ATTACK feature for Stage 2.
"""
from nvflare.apis.impl.controller import Controller
from nvflare.apis.controller_spec import Task
from nvflare.apis.fl_context import FLContext
from nvflare.apis.signal import Signal
from nvflare.apis.shareable import Shareable


class BroadcastInnerModel(Controller):
    def __init__(
        self,
        persistor_id: str = "persistor_inner",
        shareable_generator_id: str = "shareable_generator_inner",
        task_name: str = "set_global_inner",
        task_timeout: int = 120,
        min_clients: int = 1,
        wait_time_after_min_received: int = 10,
    ):
        super().__init__()
        self._persistor_id = persistor_id
        self._shareable_generator_id = shareable_generator_id
        self._task_name = task_name
        self._task_timeout = task_timeout
        self._min_clients = min_clients
        self._wait_time_after_min_received = wait_time_after_min_received
        self._persistor = None

    def start_controller(self, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()
        self._persistor = engine.get_component(self._persistor_id)
        if self._persistor is None:
            self.system_panic(f"Persistor '{self._persistor_id}' not found", fl_ctx)

    def control_flow(self, abort_signal: Signal, fl_ctx: FLContext):
        engine = fl_ctx.get_engine()

        # Load the globally aggregated Stage 1 model saved by persistor_inner
        global_weights = self._persistor.load(fl_ctx)
        if global_weights is None:
            self.system_panic("Global inner model not available — run train_inner first", fl_ctx)
            return

        # Convert ModelLearnable → Shareable using XGBModelShareableGenerator
        shareable_gen = engine.get_component(self._shareable_generator_id)
        if shareable_gen is None:
            self.system_panic(f"'{self._shareable_generator_id}' component not found", fl_ctx)
            return
        data_shareable = shareable_gen.learnable_to_shareable(global_weights, fl_ctx)

        print(f"[BroadcastInnerModel] Broadcasting global inner model to all clients via '{self._task_name}'")

        task = Task(
            name=self._task_name,
            data=data_shareable,
            timeout=self._task_timeout,
        )
        self.broadcast_and_wait(
            task=task,
            fl_ctx=fl_ctx,
            min_responses=self._min_clients,
            wait_time_after_min_received=self._wait_time_after_min_received,
            abort_signal=abort_signal,
        )

    def stop_controller(self, fl_ctx: FLContext):
        pass
