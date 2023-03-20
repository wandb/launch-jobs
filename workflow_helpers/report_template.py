import wandb.apis.reports as wr

report = wr.Report(
    project="github-actions-report",
    title="Hello from Github Actions!",
    description="This report was generated from a github action on the marketplace",
    blocks=[
        wr.P(
            [
                "To learn more, check out the action ",
                wr.Link(
                    "on the github marketplace",
                    url="https://github.com/marketplace/actions/generate-weights-biases-report",
                ),
            ]
        ),
        wr.H1("Here are some charts"),
        wr.P("We wrote everything declaratively!"),
        wr.PanelGrid(
            runsets=[
                wr.Runset(project="lineage-example"),
                wr.Runset(project="report-api-quickstart"),
            ],
            panels=[
                wr.LinePlot(y="val_acc", title="Validation Accuracy over Time"),
                wr.BarPlot(metrics="acc"),
                wr.MediaBrowser(media_keys="img", num_columns=1),
            ],
        ),
        wr.H1("Here is some artifact lineage"),
        wr.WeaveBlockArtifact(
            "megatruong", "lineage-example", artifact="model-1", tab="lineage"
        ),
    ],
).save()
