import logging

import weave
from weave.evaluation.eval_imperative import EvaluationLogger
from weave.evaluation.eval_imperative import _active_evaluation_loggers
from weave.flow import leaderboard
from weave.trace import urls as weave_urls
from weave.trace.ref_util import get_ref

logger = logging.getLogger(__name__)

LEADERBOARD_NAME = "Inspect AI Leaderboard"
LEADERBOARD_DESCRIPTION = (
    "Leaderboard comparing model performance on the various tasks.\n\n"
    "This leaderboard was automatically generated from Inspect AI evaluations logged to Weave."
)

def build_columns_from_eval_logger(eval_logger: EvaluationLogger) -> list[leaderboard.LeaderboardColumn]:
    """
    Build leaderboard columns for a single evaluation logger.
    Processes only the 'output' scorer from the evaluation output.
    """
    eval_output = eval_logger._evaluate_call and (eval_logger._evaluate_call.output or {})
    output_scorer = eval_output.get("output", {})
    lb_columns = []
    for metric_name, matric_values in output_scorer.items():
        for m_value in matric_values:
            if "err" in m_value:
                continue
            
            try:
                lb_columns.append(
                        leaderboard.LeaderboardColumn(
                            evaluation_object_ref=get_ref(
                                eval_logger._pseudo_evaluation
                            ).uri(),
                            scorer_name="output",
                            summary_metric_path=f"{metric_name}.{m_value}",
                            should_minimize=False,
                        )
                    )
            except Exception as e:
                logger.debug(
                    f"Could not create column for {metric_name}: {e}"
                )
                continue

    return lb_columns

def create_leaderboard(
    name: str = LEADERBOARD_NAME,
    description: str = LEADERBOARD_DESCRIPTION,
) -> None:
    """
    Create a leaderboard from eval loggers in the WeaveEvaluationHooks instance.
    
    Requires the weave client to be initialized.

    Args:
        name: Leaderboard name.
        description: Leaderboard description.
    """ 
    assert weave.get_client() is not None, "Weave client not initialized"
       
    try:
        leaderboard_columns = []
        for eval_logger in _active_evaluation_loggers:
            leaderboard_columns.extend(build_columns_from_eval_logger(eval_logger))

        spec = leaderboard.Leaderboard(
            name=name,
            description=description,
            columns=leaderboard_columns,
        )
        ref = weave.publish(spec)
        url = weave_urls.leaderboard_path(
            ref.entity,
            ref.project,
            ref.name,
        )

        print(f"View Leaderboard in Weave: {url}")

    except Exception as e:
        logger.error(f"Failed to create leaderboard: {e}", exc_info=True)
        print(f"Failed to create leaderboard: {e}")
