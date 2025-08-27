import logging

import weave
from weave.evaluation.eval_imperative import _active_evaluation_loggers
from weave.flow import leaderboard
from weave.trace import urls as weave_urls
from weave.trace.ref_util import get_ref

logger = logging.getLogger(__name__)


def create_leaderboard(weave_client=None) -> None:
    """
    Create a leaderboard from eval loggers in the WeaveEvaluationHooks instance.

    Args:
        weave_client: Weave client instance. If None, will use the current context's client.
    """
    try:

        lb_columns = []
        for eval_logger in _active_evaluation_loggers:
            eval_output = eval_logger._evaluate_call.output
            for scorer_name, metric_dict in eval_output.items():
                if scorer_name == "output":
                    for metric_name, matric_values in metric_dict.items():
                        for m_value in matric_values:
                            if "err" not in m_value:
                                try:
                                    lb_columns.append(
                                        leaderboard.LeaderboardColumn(
                                            evaluation_object_ref=get_ref(
                                                eval_logger._pseudo_evaluation
                                            ).uri(),
                                            scorer_name=scorer_name,
                                            summary_metric_path=f"{metric_name}.{m_value}",
                                            should_minimize=False,
                                        )
                                    )
                                except Exception as e:
                                    logger.debug(
                                        f"Could not create column for {scorer_name}: {e}"
                                    )
                                    continue

        leaderboard_name = f"Inspect AI Leaderboard"
        leaderboard_description = f"""
Leaderboard comparing model performance on the various tasks. 

This leaderboard was automatically generated from Inspect AI evaluations logged to Weave.
""".strip()

        spec = leaderboard.Leaderboard(
            name=leaderboard_name,
            description=leaderboard_description,
            columns=lb_columns,
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
