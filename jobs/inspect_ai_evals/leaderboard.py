import wandb
import weave
from weave.evaluation.eval_imperative import EvaluationLogger
from weave.evaluation.eval_imperative import _active_evaluation_loggers
from weave.flow import leaderboard
from weave.trace import urls as weave_urls
from weave.trace.ref_util import get_ref


LEADERBOARD_REF = "Inspect-AI-Leaderboard"
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
        # Build new columns from the active evaluation loggers in this run
        new_columns: list[leaderboard.LeaderboardColumn] = []
        for eval_logger in _active_evaluation_loggers:
            new_columns.extend(build_columns_from_eval_logger(eval_logger))

        # Pull any existing leaderboard (latest version) and merge columns
        existing_columns: list[leaderboard.LeaderboardColumn] = []
        try:
            existing = weave.ref(LEADERBOARD_REF).get()
            cols = getattr(existing, "columns", None)
            if cols:
                existing_columns = list(cols)
        except Exception:
            # No existing leaderboard with this name, or not retrievable – start fresh
            existing_columns = []

        merged_columns = list(
            {
                (column.evaluation_object_ref, column.scorer_name, column.summary_metric_path, column.should_minimize): column
                for column in (existing_columns or []) + new_columns
            }.values()
        )

        spec = leaderboard.Leaderboard(
            name=name,
            description=description,
            columns=merged_columns,
        )
        ref = weave.publish(spec, name=name)
        url = weave_urls.leaderboard_path(
            ref.entity,
            ref.project,
            ref.name,
        )

        print(f"View Leaderboard in Weave: {url}")

    except Exception as e:
        wandb.termerror(f"Failed to create leaderboard: {e}")