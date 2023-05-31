import optuna


def objective(trial):
    database = trial.suggest_categorical("database-123", ["small", "medium", "large"])
    randomize = None  # init for printing

    param1 = trial.suggest_int("param1", 0, 10)
    sleep = trial.suggest_float("sleep", 0.1, 0.3)

    if database in ["small", "large"]:
        batch_size = trial.suggest_int("batch_size", 16, 64)

        # maybe test randomization when the batch_size is small
        randomize = trial.suggest_categorical("randomize", [True, False])
    else:
        batch_size = trial.suggest_int("batch_size", 64, 256)

    print(f"{database=} {batch_size=} {randomize=}")

    return -1


def sampler():
    return optuna.samplers.NSGAIISampler(
        population_size=100,
        crossover_prob=0.2,
        seed=1000000,
    )


"""Example of an alternate sampler"""
# def sampler():
#     return optuna.samplers.QMCSampler(
#         qmc_type="halton",
#         scramble=True
#     )


def pruner():
    return optuna.pruners.SuccessiveHalvingPruner()
