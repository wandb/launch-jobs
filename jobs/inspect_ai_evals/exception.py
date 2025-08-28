from datasets.exceptions import DatasetNotFoundError
import logging

logger = logging.getLogger(__name__)

def handle_exception(e: Exception) -> None:
    if isinstance(e, DatasetNotFoundError):
        print("Dataset not found:")
        print(f"Hint: If this is a gated dataset, try setting your Hugging Face token in the 'hf_token' field")
        print(f"Full exception:\n{e}")
        print("--------------------------------")
        logger.error(f"Dataset not found: {e}", exc_info=True)
        raise e
    
    raise e