import os

def get_recent_model(dir_path):
    models = [item.name.split("-")[-1].split(".")[:-1] for item in os.scandir(dir_path)]
    models_full_times = ["".join(t) for t in models]

    most_recent = max(models_full_times)
    most_recent_idx = models_full_times.index(most_recent)

    model_path = f"model-{".".join(models[most_recent_idx])}.pt"

    return model_path