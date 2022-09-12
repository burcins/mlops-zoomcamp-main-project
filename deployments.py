# from prefect.flow_runners import SubprocessFlowRunner
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import IntervalSchedule
from datetime import timedelta
import main




deployment = Deployment.build_from_flow(
    flow=main.main(),
    name="wine_quality_clf",
    schedule=IntervalSchedule(interval=timedelta(days=7)),
    work_queue_name="model_experiments"
)

deployment.apply()

