from clearml import Task
import matplotlib.pyplot as plt


def get_experiment_ids_from_url(url: str) -> list[str]:
    """Extract experiment IDs from a ClearML compare-experiments URL.

    Args:
        url (str): ClearML URL containing experiment IDs

    Returns:
        list[str]: List of experiment IDs
    """
    # Find the ids= parameter and extract everything after it until the next / or end of string
    if "ids=" not in url:
        raise ValueError("URL does not contain experiment IDs in the expected format")

    ids_section = url.split("ids=")[1].split("/")[0]
    experiment_ids = ids_section.split(",")
    return experiment_ids


def get_loss_data(task_ids):
    loss_data = {}
    for task_id in task_ids:
        task = Task.get_task(task_id=task_id)
        scalar_logs = task.get_reported_scalars()

        x_values = scalar_logs["loss"]["loss"]["x"]
        y_values = scalar_logs["loss"]["loss"]["y"]

        task_name = task.name.replace("model.", "")
        loss_data[task_id] = {"name": task_name, "steps": x_values, "loss": y_values}
    return loss_data


def calculate_ema(data, smoothing=0.97):
    ema = [data[0]]
    for value in data[1:]:
        ema.append(ema[-1] * smoothing + value * (1 - smoothing))
    return ema


def plot_loss_data(loss_data, plot_last: int = 1000):
    plt.figure(figsize=(10, 6))
    for _, data in loss_data.items():
        steps = data["steps"][-plot_last:]
        loss = data["loss"][-plot_last:]

        loss_ema = calculate_ema(loss, smoothing=0.97)

        (line,) = plt.plot(steps, loss, alpha=0.5, label=f"{data['name']}")
        color = line.get_color()
        plt.plot(steps, loss_ema, color=color)

    plt.title("Loss Over Steps for Each Task")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend(
        title="Experiments",
        bbox_to_anchor=(0.5, -0.1),
        loc="upper center",
        fontsize="small",
        title_fontsize="small",
        ncol=2,
    )
    plt.minorticks_on()
    plt.grid(which="both", linestyle="--", linewidth=0.5)
    plt.show()
