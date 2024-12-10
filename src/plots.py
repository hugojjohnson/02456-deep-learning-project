import matplotlib.pyplot as plt

def plot_wind(data, start=0, end=-1, label='Wind Load Blade 1 (Mz1)'):
    plt.plot(data[start:end])
    plt.xlabel('Time (index)')
    plt.ylabel(label)
    plt.title('Wind Load Blade over time')
    plt.show()


def evaluate_predictions(predicted, actual, mse, start=0, end=-1, name='[model name]', label='[insert label]', dots=False):
    plt.plot(actual, label='Actual Mz1')
    if dots:
        plt.plot(predicted, label='Predicted Mz1', marker='o', linestyle='')
    else:
        plt.plot(predicted, label='Predicted Mz1')
    plt.xlabel('Samples')
    plt.ylabel('Mz1')
    plt.legend()
    plt.title("Evaluation of " + name)
    # Add text beneath the plot
    text = f"Mean Squared Error: {mse:.3g}"
    plt.figtext(0, 0, text, ha='left', va='top', fontsize=12, wrap=True)
    plt.show()

# from pathlib import Path

# import typer
# from loguru import logger
# from tqdm import tqdm

# from src.config import FIGURES_DIR, PROCESSED_DATA_DIR

# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     output_path: Path = FIGURES_DIR / "plot.png",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Generating plot from data...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Plot generation complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
