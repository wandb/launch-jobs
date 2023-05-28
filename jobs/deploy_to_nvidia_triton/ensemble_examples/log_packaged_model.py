import wandb

def log_packaged_model(components):
    with wandb.init(project='red-bull-triton', job_type="package_components") as run:
        art = wandb.Artifact('ensemble_model', type='ensemble_model')
        # art.add_dir('model_repository/ensemble_model')
        art.add_dir('ensemble_model')
        arts = [run.use_artifact(f"{component}:latest") for component in components]
        add_refs(art, *arts)
        run.log_artifact(art)
        run.log_code()

def add_refs(target, *sources):
    for source in sources:
        for name in source.manifest.entries:
            ref = source.get_path(name)
            art_name, art_ver = source.name.split(':v')
            namespaced_fname = f"{art_name}/{name}"
            target.add_reference(ref, namespaced_fname)


ensemble_components = ['detection_preprocessing', 'text_detection', 'detection_postprocessing', 'text_recognition', 'recognition_postprocessing']
log_packaged_model(ensemble_components)
